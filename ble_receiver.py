"""
ble_receiver.py — PawDynamics BLE 수신기
=========================================
PAW-LF / PAW-RF / PAW-LH / PAW-RH 4대에서
동시에 BLE Notification 을 수신하고
희소 인코딩된 패킷을 160점 행렬로 복원한다.

사용법:
    python ble_receiver.py

의존성:
    pip install bleak

설계자료 7장 기반:
    - MTU 247 바이트
    - 희소 인코딩 (index + value 쌍)
    - 타임스탬프 포함
    - 4대 동시 연결
"""

import asyncio
import struct
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable
from bleak import BleakClient, BleakScanner

# ── 로깅 설정 ─────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("BLE_RECEIVER")

# ── 상수 ──────────────────────────────────────
# 보드 광고 이름 (펌웨어 ble_transport.c 와 일치)
# [2발 모드] LF/RF 2대만 사용 — LH/RH 는 이번 실험에서 제외, 스캔 대상에서도 뺌
PAW_NAMES = ["PAW-LF", "PAW-RF"]

# GATT UUID (펌웨어 ble_transport.c 와 일치)
PAW_SVC_UUID = "00001234-0000-1000-8000-00805f9b34fb"
PAW_CHR_UUID = "00005678-0000-1000-8000-00805f9b34fb"

# 행렬 크기
MATRIX_ROWS = 16
MATRIX_COLS = 10
MATRIX_SIZE = MATRIX_ROWS * MATRIX_COLS   # 160

# 패킷 헤더 구조 (struct 포맷)
# timestamp_ms(4) + position(1) + node_count(1) +
# pitch(4) + roll(4) + accel_x(4) + accel_y(4) + accel_z(4) + battery_pct(1)
HEADER_FMT  = "<IBBfffffB"
HEADER_SIZE = struct.calcsize(HEADER_FMT)   # 27바이트

# 노드 1개 구조: index(1) + value(2) = 3바이트
NODE_FMT  = "<BH"
NODE_SIZE = struct.calcsize(NODE_FMT)       # 3바이트


# ── 데이터 클래스 ──────────────────────────────
@dataclass
class ImuData:
    """IMU 자세 데이터"""
    pitch_deg: float = 0.0    # 앞뒤 기울기 (°)
    roll_deg:  float = 0.0    # 좌우 기울기 (°)
    accel_x:   float = 0.0    # 선형 가속도 X (m/s²)
    accel_y:   float = 0.0    # 선형 가속도 Y (m/s²)
    accel_z:   float = 0.0    # 선형 가속도 Z (m/s²)


@dataclass
class PawFrame:
    """
    보드 1장에서 수신한 한 프레임 데이터.
    수신 직후 희소 → 행렬 복원 완료 상태.
    """
    position:     int                          # 0=LF 1=RF 2=LH 3=RH
    timestamp_ms: int                          # 펌웨어 기준 타임스탬프 (ms)
    recv_time:    float = 0.0                  # PC 수신 시각 (time.monotonic)
    matrix:       list  = field(default_factory=lambda: [
                              [0] * MATRIX_COLS for _ in range(MATRIX_ROWS)
                          ])                  # [16][10] 복원된 행렬
    imu:          ImuData = field(default_factory=ImuData)
    battery_pct:  int   = 100
    node_count:   int   = 0                   # 전송된 희소 노드 수


# ── 패킷 파싱 ──────────────────────────────────
def parse_packet(data: bytes) -> Optional[PawFrame]:
    """
    BLE 수신 바이트를 PawFrame 으로 파싱.

    패킷 구조 (펌웨어 ble_packet_t 와 동일):
        [헤더 27B] + [노드 × 3B]

    Returns:
        PawFrame 또는 파싱 실패 시 None
    """
    if len(data) < HEADER_SIZE:
        log.warning(f"패킷 너무 짧음: {len(data)}B < {HEADER_SIZE}B")
        return None

    # 헤더 파싱
    (timestamp_ms, position, node_count,
     pitch, roll, ax, ay, az, bat_pct) = struct.unpack_from(HEADER_FMT, data, 0)

    frame = PawFrame(
        position     = position,
        timestamp_ms = timestamp_ms,
        recv_time    = time.monotonic(),
        imu          = ImuData(pitch_deg=pitch, roll_deg=roll,
                               accel_x=ax, accel_y=ay, accel_z=az),
        battery_pct  = bat_pct,
        node_count   = node_count,
    )

    # 희소 노드 → 행렬 복원
    offset = HEADER_SIZE
    for _ in range(node_count):
        if offset + NODE_SIZE > len(data):
            log.warning("패킷 길이 부족 — 노드 파싱 중단")
            break
        idx, val = struct.unpack_from(NODE_FMT, data, offset)
        offset += NODE_SIZE
        r = idx // MATRIX_COLS
        c = idx  % MATRIX_COLS
        if 0 <= r < MATRIX_ROWS and 0 <= c < MATRIX_COLS:
            frame.matrix[r][c] = val

    return frame


# ── BLE 수신기 ─────────────────────────────────
class PawReceiver:
    """
    PAW-LF/RF/LH/RH 4대 동시 BLE 수신기.

    사용법:
        receiver = PawReceiver(on_frame=my_callback)
        asyncio.run(receiver.run())

    콜백:
        on_frame(frame: PawFrame) 이 프레임마다 호출됨
    """

    def __init__(self, on_frame: Callable[[PawFrame], None] = None):
        self.on_frame  = on_frame
        self.clients:  Dict[str, BleakClient] = {}   # name → client
        self.addresses: Dict[str, str] = {}           # name → address
        self._running  = False

    # ── 스캔 ────────────────────────────────────
    async def scan(self, timeout: float = 10.0) -> Dict[str, str]:
        """
        PAW-* 이름의 BLE 장치를 스캔하여 주소를 반환.

        Returns:
            {"PAW-LF": "AA:BB:...", ...}
        """
        log.info(f"BLE 스캔 시작 ({timeout}초)...")
        found = {}

        def detection_cb(device, _):
            if device.name and device.name in PAW_NAMES:
                found[device.name] = device.address
                log.info(f"발견: {device.name} ({device.address})")

        # [패치] 원본의 `await BleakScanner.start()` 는 인스턴스 없이 호출되어 TypeError → 삭제
        scanner = BleakScanner(detection_callback=detection_cb)
        await scanner.start()
        await asyncio.sleep(timeout)
        await scanner.stop()

        log.info(f"스캔 완료: {list(found.keys())}")
        missing = [n for n in PAW_NAMES if n not in found]
        if missing:
            log.warning(f"[패치] 발견되지 않은 보드: {missing} — 전원/광고 상태 확인")
        return found

    # ── Notification 콜백 ────────────────────────
    def _make_notification_cb(self, name: str):
        """각 보드별 Notification 수신 콜백 생성"""
        def callback(sender, data: bytearray):
            frame = parse_packet(bytes(data))
            if frame is None:
                return
            log.debug(f"{name}: {frame.node_count}노드 "
                      f"pitch={frame.imu.pitch_deg:.1f}° "
                      f"bat={frame.battery_pct}%")
            if self.on_frame:
                self.on_frame(frame)
        return callback

    # ── 연결 ────────────────────────────────────
    async def connect_all(self, addresses: Dict[str, str]):
        """주소 딕셔너리의 모든 보드에 동시 연결"""
        self.addresses = addresses
        tasks = [self._connect_one(name, addr)
                 for name, addr in addresses.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for name, result in zip(addresses.keys(), results):
            if isinstance(result, Exception):
                log.error(f"{name} 연결 실패: {result}")
            else:
                log.info(f"{name} 연결 성공 ✓")

    async def _connect_one(self, name: str, address: str):
        """단일 보드 연결 + Notification 등록"""
        client = BleakClient(address, timeout=15.0)
        await client.connect()
        self.clients[name] = client

        # [패치] bleak 에는 request_mtu() 가 없음. MTU 는 OS 가 협상하고 결과만 읽을 수 있다.
        try:
            log.info(f"{name}: MTU={client.mtu_size} "
                     f"(패킷 최대 {client.mtu_size - 3}B, 27+3×노드수 가 이를 넘으면 잘림)")
        except Exception:
            pass

        # Notification 등록
        await client.start_notify(PAW_CHR_UUID,
                                  self._make_notification_cb(name))
        log.info(f"{name}: Notification 등록 완료")

    # ── 재연결 ──────────────────────────────────
    async def _reconnect_loop(self, name: str, address: str):
        """
        연결이 끊기면 5초 간격으로 재연결 시도.
        설계자료 7-3: 끊김을 전제로 설계
        """
        while self._running:
            client = self.clients.get(name)
            # [패치] 최초 연결에 실패한 보드는 self.clients 에 아예 등록되지 않아
            # client 가 None 이 되고, 예전 조건(if client and ...)에서는 영원히
            # 재연결을 시도하지 않았다. None 도 "연결 안 됨"으로 취급해야 한다.
            if client is None or not client.is_connected:
                log.warning(f"{name} 끊김 → 재연결 시도...")
                try:
                    await self._connect_one(name, address)
                    log.info(f"{name} 재연결 성공 ✓")
                except Exception as e:
                    log.error(f"{name} 재연결 실패: {e}")
            await asyncio.sleep(5.0)

    # ── 메인 실행 ────────────────────────────────
    async def run(self, scan_timeout: float = 10.0):
        """
        스캔 → 연결 → 수신 루프 실행.
        Ctrl+C 로 종료.
        """
        self._running = True

        # 스캔
        addresses = await self.scan(scan_timeout)
        if not addresses:
            log.error("PAW-* 장치를 찾을 수 없음")
            return

        # 연결
        await self.connect_all(addresses)

        # 재연결 감시 태스크 시작
        reconnect_tasks = [
            asyncio.create_task(
                self._reconnect_loop(name, addr)
            )
            for name, addr in addresses.items()
        ]

        log.info("=== 수신 중 (Ctrl+C 로 종료) ===")
        try:
            while self._running:
                await asyncio.sleep(1.0)
                # 연결 상태 주기 로그
                status = {n: c.is_connected
                          for n, c in self.clients.items()}
                log.debug(f"연결 상태: {status}")
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            for task in reconnect_tasks:
                task.cancel()
            for client in self.clients.values():
                if client.is_connected:
                    await client.disconnect()
            log.info("=== 수신 종료 ===")

    async def stop(self):
        """수신 중단"""
        self._running = False


# ── 테스트용 더미 수신기 (보드 없이 테스트) ────
class DummyPawReceiver:
    """
    보드 없이 테스트할 때 사용하는 더미 수신기.
    1학기 CSV 데이터를 읽어서 PawFrame 으로 변환 후
    on_frame 콜백을 50Hz 로 호출한다.
    """

    def __init__(self, on_frame: Callable[[PawFrame], None] = None,
                 csv_path: str = None):
        self.on_frame = on_frame
        self.csv_path = csv_path
        self._running = False

    async def run(self, hz: float = 50.0):
        """더미 프레임 50Hz 로 송출"""
        import random
        self._running = True
        period = 1.0 / hz
        frame_idx = 0

        log.info("=== 더미 수신기 시작 (보드 없이 테스트) ===")
        while self._running:
            t0 = time.monotonic()

            for pos in range(2):   # [2발 모드] LF/RF 만
                # 랜덤 더미 행렬 생성
                mat = [[random.randint(0, 500) for _ in range(MATRIX_COLS)]
                       for _ in range(MATRIX_ROWS)]
                frame = PawFrame(
                    position     = pos,
                    timestamp_ms = int(time.monotonic() * 1000),
                    recv_time    = time.monotonic(),
                    matrix       = mat,
                    imu          = ImuData(
                        pitch_deg = random.uniform(-5, 5),
                        roll_deg  = random.uniform(-3, 3),
                    ),
                    battery_pct  = 85,
                    node_count   = 30,
                )
                if self.on_frame:
                    self.on_frame(frame)

            frame_idx += 1
            elapsed = time.monotonic() - t0
            sleep_t = max(0, period - elapsed)
            await asyncio.sleep(sleep_t)


# ── 단독 실행 ──────────────────────────────────
if __name__ == "__main__":
    received_count = [0]

    def on_frame(frame: PawFrame):
        received_count[0] += 1
        pos_names = ["LF", "RF"]   # [2발 모드]
        name = pos_names[frame.position] if frame.position < 2 else "??"
        print(f"[{name}] ts={frame.timestamp_ms}ms "
              f"nodes={frame.node_count} "
              f"pitch={frame.imu.pitch_deg:.1f}° "
              f"bat={frame.battery_pct}%")

    # 보드가 없으면 더미 수신기로 테스트
    USE_DUMMY = True   # 실제 보드 사용 시 False 로 변경

    if USE_DUMMY:
        receiver = DummyPawReceiver(on_frame=on_frame)
        asyncio.run(receiver.run(hz=10))   # 테스트용 10Hz
    else:
        receiver = PawReceiver(on_frame=on_frame)
        asyncio.run(receiver.run())
