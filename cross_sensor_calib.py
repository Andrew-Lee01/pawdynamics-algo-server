"""
cross_sensor_calib.py — 센서 상호 보정 [2발 모드: LF/RF]
==========================================
설계자료 8-2절 기반.

목적:
    LF/RF 두 센서가 같은 하중에서 같은 합계를 출력하도록
    센서 간 스케일 계수를 맞춘다.
    이 보정 없이는 SI(대칭성 지수) 를 신뢰할 수 없다.

방법:
    1. 두 발에 동일한 무게를 동시에 올려놓고 측정
    2. 각 발의 전체 합계를 비교
    3. 기준 발(LF) 대비 나머지 발의 스케일 계수 계산
    4. sensor_gain.csv 저장

보정 적용:
    corrected_sum = raw_sum * gain[pos]

사용법:
    python cross_sensor_calib.py
"""

import asyncio
import numpy as np
import csv
import logging
from pathlib import Path

from ble_receiver import DummyPawReceiver, PawReceiver, MATRIX_ROWS, MATRIX_COLS
from frame_sync import FrameSynchronizer, SyncedFrame, NUM_PAWS
from node_calibration import load_calibration, apply_calibration

log = logging.getLogger("CROSS_CALIB")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# ── 설정 ──────────────────────────────────────
CALIB_WEIGHTS_KG = [0.5, 1.0, 2.0]
CALIB_REPEATS    = 3
CALIB_FRAMES     = 100    # 2초분

SAVE_PATH = Path("calib/sensor_gain.csv")
POS_NAMES = ["LF", "RF"]
REF_POS   = 0   # 기준 발 (LF)

USE_DUMMY = False  # 실제 보드 사용 시 False


# ── 상호 보정 수집기 ───────────────────────────
class CrossSensorCalibrator:

    def __init__(self):
        # position → 합계 누적 리스트
        self._sums: dict = {i: [] for i in range(NUM_PAWS)}
        self._collecting = False
        # 노드별 보정계수 로드
        self._node_coeffs = [load_calibration(pos) for pos in range(NUM_PAWS)]

    def on_synced(self, synced: SyncedFrame):
        if not self._collecting:
            return
        for pos in range(NUM_PAWS):
            mat = synced.get_matrix(pos)
            # 노드별 보정 먼저 적용
            corrected = apply_calibration(mat, self._node_coeffs[pos])
            self._sums[pos].append(corrected.sum())

    def start_collection(self):
        self._collecting = True
        for pos in range(NUM_PAWS):
            self._sums[pos].clear()

    def stop_collection(self):
        self._collecting = False

    def collected_count(self) -> int:
        return min(len(self._sums[i]) for i in range(NUM_PAWS))

    def compute_gains(self) -> dict:
        """
        기준 발(LF) 대비 각 발의 이득 계산.

        gain[pos] = mean_sum[REF_POS] / mean_sum[pos]
        → 적용: corrected = raw_sum * gain[pos]
        """
        means = {}
        for pos in range(NUM_PAWS):
            if self._sums[pos]:
                means[pos] = np.mean(self._sums[pos])
            else:
                means[pos] = 1.0

        ref_mean = means[REF_POS]
        gains = {}
        for pos in range(NUM_PAWS):
            if means[pos] > 1e-6:
                gains[pos] = ref_mean / means[pos]
            else:
                gains[pos] = 1.0
            # 이상값 클램프 (0.7 ~ 1.3)
            gains[pos] = float(np.clip(gains[pos], 0.7, 1.3))

        log.info("센서 간 이득:")
        for pos in range(NUM_PAWS):
            log.info(f"  {POS_NAMES[pos]}: mean={means[pos]:.1f} "
                     f"gain={gains[pos]:.4f}")
        return gains


# ── 저장/불러오기 ──────────────────────────────
def save_gains(gains: dict):
    SAVE_PATH.parent.mkdir(exist_ok=True)
    with open(SAVE_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["position", "name", "gain"])
        for pos in range(NUM_PAWS):
            writer.writerow([pos, POS_NAMES[pos], f"{gains[pos]:.6f}"])
    log.info(f"이득 저장: {SAVE_PATH}")


def load_gains() -> dict:
    """저장된 이득 불러오기. 파일 없으면 1.0 반환."""
    if not SAVE_PATH.exists():
        log.warning(f"이득 파일 없음: {SAVE_PATH} → 1.0 사용")
        return {i: 1.0 for i in range(NUM_PAWS)}
    gains = {}
    with open(SAVE_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gains[int(row["position"])] = float(row["gain"])
    log.info(f"이득 로드: {SAVE_PATH}")
    return gains


# ── 메인 ──────────────────────────────────────
async def run_cross_calibration():
    calibrator = CrossSensorCalibrator()
    sync = FrameSynchronizer(on_synced=calibrator.on_synced)

    if USE_DUMMY:
        receiver = DummyPawReceiver(on_frame=sync.on_frame)
        recv_task = asyncio.create_task(receiver.run(hz=50))
    else:
        receiver = PawReceiver(on_frame=sync.on_frame)
        recv_task = asyncio.create_task(receiver.run())
        await asyncio.sleep(15.0)

    all_gains = []

    for weight in CALIB_WEIGHTS_KG:
        for rep in range(CALIB_REPEATS):
            input(f"\n[상호보정] LF/RF 두 발 모두에 {weight}kg 올리고 Enter...")

            log.info(f"{weight}kg 측정 시작...")
            calibrator.start_collection()

            while calibrator.collected_count() < CALIB_FRAMES:
                await asyncio.sleep(0.1)

            calibrator.stop_collection()
            gains = calibrator.compute_gains()
            all_gains.append(gains)

    # 평균 이득 계산 및 저장
    final_gains = {}
    for pos in range(NUM_PAWS):
        final_gains[pos] = float(np.mean([g[pos] for g in all_gains]))
    save_gains(final_gains)

    recv_task.cancel()
    try:
        await recv_task
    except asyncio.CancelledError:
        pass

    log.info("센서 상호 보정 완료!")


if __name__ == "__main__":
    print("=== PawDynamics 센서 상호 보정 (LF/RF) ===")
    print("⚠ node_calibration.py 를 먼저 실행해야 합니다")
    if USE_DUMMY:
        print("※ 더미 모드")
    asyncio.run(run_cross_calibration())
