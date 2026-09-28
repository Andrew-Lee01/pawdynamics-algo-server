"""
frame_sync.py — 발 프레임 동기화
====================================
[2발 모드] NUM_PAWS(=2, LF/RF)대 보드에서 각각 오는 PawFrame 을
타임스탬프 기준으로 정렬하여 동기화된 묶음(SyncedFrame)을 만든다.
원래 4발용으로 설계됐으나 이번 실험은 LH/RH 를 쓰지 않으므로
NUM_PAWS 상수만 2로 맞춰 그대로 사용한다.

설계자료 7-2 기반:
    - BLE 패킷 도착 시각은 10~50ms 지터가 있음
    - 강아지 입각기 0.2~0.3초 → 최대 25% 오차 가능
    - 타임스탬프 기반으로 정렬해야 정확한 SI 계산 가능

동기화 방법:
    1. 각 보드의 타임스탬프 오프셋 계산
       (왕복 시간 ÷ 2 = 단방향 지연 추정)
    2. 보정된 타임스탬프로 4발 프레임 묶기
    3. SYNC_WINDOW_MS 안에 들어온 프레임을 한 묶음으로 처리
    4. 빠진 발이 있으면 이전 프레임으로 보간
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import deque

from ble_receiver import PawFrame, MATRIX_ROWS, MATRIX_COLS, ImuData

log = logging.getLogger("FRAME_SYNC")

# ── 상수 ──────────────────────────────────────
# [패치] 50Hz 고정 틱 방식: 20ms 마다 4발에서 틱 시각에 가장 가까운 프레임을 1개씩 뽑는다.
TICK_MS = 20
# 틱 시각과 이 값 이상 벌어진 프레임은 "이 틱에는 없음"(결측) 으로 본다.
# 보드마다 20ms 루프 위상이 제각각이라 정상 프레임도 최대 10ms 는 벌어진다.
MATCH_TOL_MS = 15
# 한 발이라도 안 와서 틱을 못 내보낼 때, 이 시간 넘게 지나면 결측 처리하고 진행
SYNC_TIMEOUT_MS = 100
# 오프셋 추정에 쓰는 최근 샘플 수 (50Hz 기준 4초)
OFFSET_WINDOW = 200

# 프레임 버퍼 최대 크기 (발당)
BUFFER_MAX = 30

# [2발 모드] 파우다이나믹스 2학기: LF(0)/RF(1) 2대만 사용, LH/RH 제외
NUM_PAWS = 2

# 보간 최대 허용 횟수 — 이 이상 빠지면 해당 발 결측으로 처리
MAX_INTERPOLATE = 5


# ── 동기화된 4발 묶음 ──────────────────────────
@dataclass
class SyncedFrame:
    """
    4발이 동기화된 한 보행 프레임.
    sync_time 을 기준으로 가장 가까운 프레임들을 묶음.
    """
    sync_time:   float                          # PC 기준 동기화 시각 (monotonic)
    frames:      Dict[int, PawFrame] = field(   # position → PawFrame
                     default_factory=dict)
    is_complete: bool = False                   # 4발 모두 있으면 True
    missing:     List[int] = field(             # 빠진 발 위치 목록
                     default_factory=list)

    @property
    def lf(self) -> Optional[PawFrame]:
        return self.frames.get(0)

    @property
    def rf(self) -> Optional[PawFrame]:
        return self.frames.get(1)

    @property
    def lh(self) -> Optional[PawFrame]:
        return self.frames.get(2)

    @property
    def rh(self) -> Optional[PawFrame]:
        return self.frames.get(3)

    def get_matrix(self, position: int) -> List[List[int]]:
        """발 위치의 행렬 반환 (없으면 0 행렬)"""
        f = self.frames.get(position)
        if f:
            return f.matrix
        return [[0] * MATRIX_COLS for _ in range(MATRIX_ROWS)]

    def get_imu(self, position: int) -> ImuData:
        """발 위치의 IMU 데이터 반환 (없으면 기본값)"""
        f = self.frames.get(position)
        return f.imu if f else ImuData()


# ── 타임스탬프 오프셋 추정 ──────────────────────
class TimestampSync:
    """
    보드 타임스탬프 ↔ PC 시각 오프셋 추정.
    설계자료 7-2: 왕복 시간 ÷ 2 = 단방향 지연

    보드 타임스탬프(ms)를 PC monotonic 시각으로 변환:
        pc_time = board_ts_ms / 1000 + offset
    """

    def __init__(self):
        # position → offset (초) 누적 평균
        self._offsets: Dict[int, float] = {}
        self._counts:  Dict[int, int]   = {}
        self._samples: Dict[int, deque] = {}

    def update(self, position: int, board_ts_ms: int, recv_pc_time: float):
        """
        수신된 프레임으로 오프셋 갱신.
        offset = PC_recv_time - board_ts_sec

        [패치] 지수이동평균 → 최근 OFFSET_WINDOW 개 중 최솟값.
        BLE 지연은 늦게 도착하는 방향으로만 흔들리므로 가장 빨리 도착한
        패킷이 실제 시계 차이에 가장 가깝다. 평균을 쓰면 지터(10~50ms)가
        오프셋에 그대로 섞여 발 사이 시각이 1~3프레임 어긋난다.
        """
        measured = recv_pc_time - board_ts_ms / 1000.0
        buf = self._samples.setdefault(position, deque(maxlen=OFFSET_WINDOW))
        buf.append(measured)
        self._offsets[position] = min(buf)
        self._counts[position]  = self._counts.get(position, 0) + 1

    def board_to_pc(self, position: int, board_ts_ms: int) -> float:
        """보드 타임스탬프를 PC monotonic 시각으로 변환"""
        offset = self._offsets.get(position, 0.0)
        return board_ts_ms / 1000.0 + offset

    def is_ready(self) -> bool:
        """모든 보드의 오프셋이 추정됐으면 True"""
        return len(self._offsets) == NUM_PAWS and all(
            v >= 10 for v in self._counts.values()
        )


# ── 프레임 동기화기 ────────────────────────────
class FrameSynchronizer:
    """
    4발 BLE 프레임 동기화기.

    on_frame(PawFrame) 을 BLE 수신 콜백으로 등록하면
    내부에서 버퍼링 후 동기화된 SyncedFrame 을
    on_synced(SyncedFrame) 콜백으로 전달한다.

    사용법:
        sync = FrameSynchronizer(on_synced=my_callback)
        receiver = PawReceiver(on_frame=sync.on_frame)
    """

    def __init__(self, on_synced: callable = None):
        self.on_synced = on_synced
        # 발별 프레임 버퍼 (deque — 자동 크기 제한)
        self._buffers: Dict[int, deque] = {
            i: deque(maxlen=BUFFER_MAX) for i in range(NUM_PAWS)
        }
        self._ts_sync   = TimestampSync()
        self._last_sync = time.monotonic()
        self._interp_cnt: Dict[int, int] = {i: 0 for i in range(NUM_PAWS)}
        self._last_frames: Dict[int, Optional[PawFrame]] = {
            i: None for i in range(NUM_PAWS)
        }
        self._tick: Optional[float] = None          # 다음에 내보낼 틱의 PC 시각(초)
        self._miss_cnt: Dict[int, int] = {i: 0 for i in range(NUM_PAWS)}   # 틱에서 결측난 횟수

    # ── BLE 수신 콜백 ───────────────────────────
    def on_frame(self, frame: PawFrame):
        """BLE 수신 시 호출. 버퍼에 넣고 동기화 시도."""
        pos = frame.position
        if pos < 0 or pos >= NUM_PAWS:
            return

        # 오프셋 갱신
        self._ts_sync.update(pos, frame.timestamp_ms, frame.recv_time)

        # 버퍼에 추가
        self._buffers[pos].append(frame)

        # 동기화 시도 — 틱이 더 안 나올 때까지 반복
        while True:
            synced = self._try_sync()
            if synced is None:
                break
            if self.on_synced:
                self.on_synced(synced)

    # ── 동기화 시도 ─────────────────────────────
    def _t(self, pos: int, frame: PawFrame) -> float:
        return self._ts_sync.board_to_pc(pos, frame.timestamp_ms)

    def _try_sync(self) -> Optional[SyncedFrame]:
        """
        [패치] 50Hz 고정 틱 동기화.

        원본은 프레임이 도착할 때마다 묶음을 내보내 초당 200개(4배)가 생성되고
        대부분이 이전 프레임 복제본이었다. gait_realtime_v2 는 FPS=50 을 가정
        (주기 = 프레임수/50) 하므로 그러면 주기·입각기 시간이 전부 틀어진다.

        방식:
          1. 틱(20ms 간격) 시각 T 를 정한다.
          2. 4발 모두 T 이후 프레임이 도착했을 때(또는 타임아웃) 발마다
             T 에 가장 가까운 프레임 1개를 뽑고, 그 프레임과 더 오래된 프레임은 버린다.
          3. T ± MATCH_TOL_MS 안에 프레임이 없는 발은 결측 → 직전 프레임으로 보간.
        """
        bufs = self._buffers
        have = [p for p in range(NUM_PAWS) if bufs[p]]
        if not have:
            return None
        tick_s = TICK_MS / 1000.0
        tol_s  = MATCH_TOL_MS / 1000.0
        tmo_s  = SYNC_TIMEOUT_MS / 1000.0

        newest = max(self._t(p, bufs[p][-1]) for p in have)

        # 첫 틱: 4발 모두 도착했으면 가장 늦게 시작한 발 기준, 아니면 타임아웃 후 시작
        if self._tick is None:
            firsts = {p: self._t(p, bufs[p][0]) for p in have}
            if len(have) == NUM_PAWS:
                self._tick = max(firsts.values())
            elif newest - min(firsts.values()) > tmo_s * 2:
                self._tick = max(firsts.values())
            else:
                return None

        T = self._tick
        latest = {p: self._t(p, bufs[p][-1]) for p in have}
        all_reached = len(have) == NUM_PAWS and all(v >= T for v in latest.values())
        if not all_reached and newest < T + tmo_s:
            return None            # 아직 기다림

        synced = SyncedFrame(sync_time=T)
        for pos in range(NUM_PAWS):
            best, best_dt = None, None
            for f in bufs[pos]:
                dt = abs(self._t(pos, f) - T)
                if best_dt is None or dt < best_dt:
                    best, best_dt = f, dt
            if best is not None and best_dt <= tol_s:
                synced.frames[pos] = best
                self._last_frames[pos] = best
                self._interp_cnt[pos] = 0
                # 뽑힌 프레임과 그보다 오래된 프레임 제거
                while bufs[pos] and bufs[pos][0] is not best:
                    bufs[pos].popleft()
                if bufs[pos]:
                    bufs[pos].popleft()
            else:
                synced.missing.append(pos)
                self._miss_cnt[pos] += 1
                # 틱보다 확실히 오래된 프레임은 버림
                while bufs[pos] and self._t(pos, bufs[pos][0]) < T - tol_s:
                    bufs[pos].popleft()
                if (self._last_frames[pos] is not None
                        and self._interp_cnt[pos] < MAX_INTERPOLATE):
                    synced.frames[pos] = self._last_frames[pos]
                    self._interp_cnt[pos] += 1

        synced.is_complete = (len(synced.missing) == 0)
        self._tick = T + tick_s
        return synced

    def get_sync_status(self) -> Dict:
        """동기화 상태 반환 (디버깅용)"""
        return {
            "offsets_ready": self._ts_sync.is_ready(),
            "buffer_sizes":  {p: len(b) for p, b in self._buffers.items()},
            "interp_counts": self._interp_cnt.copy(),
            "miss_counts":   self._miss_cnt.copy(),
        }


# ── 단독 실행 테스트 ───────────────────────────
if __name__ == "__main__":
    import asyncio
    from ble_receiver import DummyPawReceiver

    sync_count = [0]

    def on_synced(synced: SyncedFrame):
        sync_count[0] += 1
        pos_names = ["LF", "RF"]   # [2발 모드]
        avgs = {}
        for pos in range(NUM_PAWS):
            mat = synced.get_matrix(pos)
            flat = [v for row in mat for v in row]
            avgs[pos_names[pos]] = sum(flat) / len(flat)

        print(f"[SYNC #{sync_count[0]:04d}] "
              f"완전={synced.is_complete} "
              f"missing={synced.missing} "
              f"평균압력={avgs}")

    sync = FrameSynchronizer(on_synced=on_synced)
    dummy = DummyPawReceiver(on_frame=sync.on_frame)

    print("=== frame_sync 테스트 시작 ===")
    asyncio.run(dummy.run(hz=10))
