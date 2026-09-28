"""
node_calibration.py — 노드별 보정표 생성
==========================================
설계자료 8-1절 기반.

목적:
    센서 개체 차이로 인한 노드별 감도 편차(최대 50%)를
    균일 하중 실측으로 보정계수를 만들어 없앤다.

사용법:
    python node_calibration.py

실행 순서 [2발 모드: LF/RF]:
    1. 보드 연결 대기
    2. 균일 하중(0.5kg, 1kg, 2kg) 각 3회 측정
    3. 각 노드의 기대값 대비 비율을 계수로 저장
    4. calib_LF.npy / calib_RF.npy 저장

보정 적용:
    corrected = raw_voltage * calib_coeff[r][c]

보드 없이 테스트:
    USE_DUMMY = True 로 설정 (더미 데이터로 계수 생성)
"""

import asyncio
import numpy as np
import time
import logging
from pathlib import Path

from ble_receiver import PawReceiver, DummyPawReceiver, PawFrame, MATRIX_ROWS, MATRIX_COLS
from frame_sync import FrameSynchronizer, SyncedFrame, NUM_PAWS

log = logging.getLogger("NODE_CALIB")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# ── 설정 ──────────────────────────────────────
# 보정 조건 (설계자료 8-1)
CALIB_WEIGHTS_KG = [0.5, 1.0, 2.0]   # 순서대로 올려놓음
CALIB_REPEATS    = 3                   # 각 무게 반복 횟수
CALIB_FRAMES     = 50                  # 무게당 수집 프레임 수 (50Hz × 1초)

# 저장 경로
SAVE_DIR = Path("calib")
SAVE_DIR.mkdir(exist_ok=True)

# 보드 위치 이름
POS_NAMES = ["LF", "RF"]

# 보드 없이 테스트할지 여부
USE_DUMMY = False  # 실제 보드 사용 시 False


# ── 보정 수집기 ────────────────────────────────
class NodeCalibrator:
    """
    균일 하중으로 노드별 보정계수를 생성한다.

    보정계수 계산:
        expected = 전체 노드 평균값
        coeff[r][c] = expected / node_mean[r][c]
        → 모든 노드가 같은 값을 출력하도록 스케일
    """

    def __init__(self):
        # position → [frames_list]
        self._collected: dict = {i: [] for i in range(NUM_PAWS)}
        self._collecting = False

    def on_synced(self, synced: SyncedFrame):
        """동기화된 프레임 수집"""
        if not self._collecting:
            return
        for pos in range(NUM_PAWS):
            mat = synced.get_matrix(pos)
            self._collected[pos].append(mat)

    def start_collection(self):
        self._collecting = True
        for pos in range(NUM_PAWS):
            self._collected[pos].clear()

    def stop_collection(self):
        self._collecting = False

    def collected_count(self, pos: int) -> int:
        return len(self._collected[pos])

    def compute_coefficients(self, pos: int) -> np.ndarray:
        """
        수집된 프레임으로 보정계수 행렬 계산.

        Returns:
            coeff[16][10] — 각 노드에 곱할 계수
        """
        if not self._collected[pos]:
            log.warning(f"pos={pos}: 수집 데이터 없음 → 1.0 계수 반환")
            return np.ones((MATRIX_ROWS, MATRIX_COLS))

        # 프레임 평균 → 노드별 평균 행렬
        frames = np.array(self._collected[pos], dtype=float)
        node_mean = frames.mean(axis=0)   # [16][10]

        # 전체 평균 (균일 하중이라면 모든 노드가 비슷해야 함)
        global_mean = node_mean.mean()

        if global_mean < 1e-6:
            log.warning(f"pos={pos}: 전체 평균이 너무 작음 (센서 미접촉?)")
            return np.ones((MATRIX_ROWS, MATRIX_COLS))

        # 보정계수: 전체평균 / 노드평균
        # 값이 작은 노드 → 계수 크게 (증폭)
        # 값이 큰 노드  → 계수 작게 (감쇠)
        coeff = global_mean / np.where(node_mean > 1e-6, node_mean, global_mean)

        # 이상값 클램프 (0.5 ~ 2.0 범위)
        coeff = np.clip(coeff, 0.5, 2.0)

        log.info(f"pos={pos} 보정계수: "
                 f"min={coeff.min():.3f} max={coeff.max():.3f} "
                 f"mean={coeff.mean():.3f}")
        return coeff


# ── 메인 보정 루틴 ─────────────────────────────
async def run_calibration():
    calibrator = NodeCalibrator()
    sync = FrameSynchronizer(on_synced=calibrator.on_synced)

    # 수신기 선택
    if USE_DUMMY:
        log.info("더미 수신기 사용 (보드 없이 테스트)")
        receiver = DummyPawReceiver(on_frame=sync.on_frame)
        recv_task = asyncio.create_task(receiver.run(hz=50))
    else:
        receiver = PawReceiver(on_frame=sync.on_frame)
        recv_task = asyncio.create_task(receiver.run())
        # 보드 연결 대기
        log.info("보드 연결 대기 중...")
        await asyncio.sleep(15.0)

    # 무게별 보정 수집
    all_coeffs = {pos: [] for pos in range(NUM_PAWS)}

    for weight in CALIB_WEIGHTS_KG:
        for rep in range(CALIB_REPEATS):
            input(f"\n[보정] LF/RF 모두에 {weight}kg 올려놓고 Enter 를 누르세요 "
                  f"(반복 {rep+1}/{CALIB_REPEATS})...")

            log.info(f"{weight}kg 측정 시작 ({CALIB_FRAMES}프레임)...")
            calibrator.start_collection()

            # 프레임 수집 대기
            while calibrator.collected_count(0) < CALIB_FRAMES:
                await asyncio.sleep(0.1)

            calibrator.stop_collection()
            log.info(f"{weight}kg 측정 완료")

            # 각 발 계수 계산 후 누적
            for pos in range(NUM_PAWS):
                coeff = calibrator.compute_coefficients(pos)
                all_coeffs[pos].append(coeff)

    # 전체 무게·반복의 평균 계수 계산 및 저장
    log.info("\n=== 최종 보정계수 저장 ===")
    for pos in range(NUM_PAWS):
        final_coeff = np.mean(all_coeffs[pos], axis=0)
        save_path = SAVE_DIR / f"calib_{POS_NAMES[pos]}.npy"
        np.save(save_path, final_coeff)
        log.info(f"  저장: {save_path} "
                 f"(min={final_coeff.min():.3f} "
                 f"max={final_coeff.max():.3f})")

    log.info("노드별 보정 완료!")

    # 정리
    recv_task.cancel()
    try:
        await recv_task
    except asyncio.CancelledError:
        pass


# ── 보정계수 불러오기 (다른 모듈에서 사용) ──────
def load_calibration(pos: int) -> np.ndarray:
    """
    저장된 보정계수 불러오기.
    파일 없으면 1.0 (보정 없음) 반환.

    Args:
        pos: 0=LF 1=RF

    Returns:
        coeff[16][10]
    """
    path = SAVE_DIR / f"calib_{POS_NAMES[pos]}.npy"
    if path.exists():
        coeff = np.load(path)
        log.info(f"보정계수 로드: {path}")
        return coeff
    else:
        log.warning(f"보정 파일 없음: {path} → 1.0 사용")
        return np.ones((MATRIX_ROWS, MATRIX_COLS))


def apply_calibration(matrix: list, coeff: np.ndarray) -> np.ndarray:
    """
    보정계수를 행렬에 적용.

    Args:
        matrix: [16][10] raw 행렬
        coeff:  [16][10] 보정계수

    Returns:
        보정된 [16][10] numpy 배열
    """
    mat = np.array(matrix, dtype=float)
    return mat * coeff


# ── 단독 실행 ──────────────────────────────────
if __name__ == "__main__":
    print("=== PawDynamics 노드별 보정 (LF/RF 2발 모드) ===")
    print(f"보정 조건: {CALIB_WEIGHTS_KG}kg × {CALIB_REPEATS}회")
    print(f"저장 경로: {SAVE_DIR.resolve()}")
    if USE_DUMMY:
        print("※ 더미 모드 (실제 보드 없이 테스트)")
    asyncio.run(run_calibration())
