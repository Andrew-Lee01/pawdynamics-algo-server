"""
cop_analyzer.py — COP 궤적 분석 [2발 모드: LF/RF]
==================================
1학기에 없던 신규 분석 모듈.

COP(Center of Pressure, 압력 중심)의 궤적을 분석하여
보행 이상의 공간적 패턴을 추출한다.

주요 분석:
    1. COP 궤적 길이 (총 이동 거리)
    2. COP 이탈도 (정상 궤적 중심에서 벗어난 정도)
    3. COP 속도 (이동 속도 변화)
    4. LF+RF 합산 CoP (전체 무게 중심)

입력:  SyncedFrame 리스트 (한 보행 주기)
출력:  COP 특징 딕셔너리

사용법:
    from cop_analyzer import CopAnalyzer
    cop = CopAnalyzer()
    features = cop.analyze(synced_frames)
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from frame_sync import SyncedFrame, NUM_PAWS
from node_calibration import apply_calibration, load_calibration

log = logging.getLogger("COP")

# ── 상수 ──────────────────────────────────────
MATRIX_ROWS  = 16
MATRIX_COLS  = 10
NODE_PITCH   = 5.0    # 노드 간격 (mm)
FPS          = 50     # 샘플링 주파수 (Hz)
LF, RF = 0, 1   # [2발 모드]
POS_NAMES = ["LF", "RF"]


# ── COP 포인트 ─────────────────────────────────
@dataclass
class CopPoint:
    """단일 시각의 COP 위치"""
    row_mm: float   # 행 방향 위치 (mm)
    col_mm: float   # 열 방향 위치 (mm)
    time:   float   # 시각 (s)
    grf:    float   # 해당 시각 GRF


# ── COP 분석기 ────────────────────────────────
class CopAnalyzer:
    """
    4발 COP 궤적 분석기.

    한 보행 주기 동안의 SyncedFrame 리스트를 받아
    발별 / 전체 COP 특징을 계산한다.
    """

    def __init__(self):
        self._node_coeffs = [load_calibration(pos) for pos in range(NUM_PAWS)]

    # ── 행렬 → COP 계산 ──────────────────────────
    def _matrix_to_cop(self, mat: list, pos: int
                        ) -> Optional[CopPoint]:
        """
        행렬에서 COP 위치 계산.

        COP_row = Σ(압력 × 행 번호) / Σ(압력)
        COP_col = Σ(압력 × 열 번호) / Σ(압력)
        """
        arr = apply_calibration(mat, self._node_coeffs[pos])
        total = arr.sum()
        if total < 1e-6:
            return None

        rows = np.arange(MATRIX_ROWS).reshape(-1, 1)
        cols = np.arange(MATRIX_COLS).reshape(1, -1)
        cop_row = float((arr * rows).sum() / total)
        cop_col = float((arr * cols).sum() / total)

        return CopPoint(
            row_mm = cop_row * NODE_PITCH,
            col_mm = cop_col * NODE_PITCH,
            time   = 0.0,   # analyze() 에서 채움
            grf    = float(total),
        )

    # ── 궤적 특징 계산 ────────────────────────────
    def _trajectory_features(self, traj: List[CopPoint],
                              pos: int) -> Dict:
        """COP 궤적 특징 추출"""
        pn = POS_NAMES[pos]
        if len(traj) < 2:
            return {f"{pn}_COP_Length(mm)":   0.0,
                    f"{pn}_COP_Deviation(mm)": 0.0,
                    f"{pn}_COP_Vel_Mean(mm/s)":0.0,
                    f"{pn}_COP_Vel_Std(mm/s)": 0.0,
                    f"{pn}_COP_Row_Range(mm)": 0.0,
                    f"{pn}_COP_Col_Range(mm)": 0.0,
                    f"{pn}_COP_Row_Mean(mm)":  0.0,
                    f"{pn}_COP_Col_Mean(mm)":  0.0}

        rows = np.array([p.row_mm for p in traj])
        cols = np.array([p.col_mm for p in traj])
        times = np.array([p.time  for p in traj])

        # 궤적 총 길이
        drows = np.diff(rows)
        dcols = np.diff(cols)
        dists = np.sqrt(drows**2 + dcols**2)
        length = float(dists.sum())

        # 이탈도: 중심에서 각 점까지의 평균 거리
        center_r = rows.mean()
        center_c = cols.mean()
        deviations = np.sqrt((rows - center_r)**2 + (cols - center_c)**2)
        deviation = float(deviations.mean())

        # 속도
        dt = np.diff(times)
        dt = np.where(dt < 1e-6, 1/FPS, dt)
        velocities = dists / dt
        vel_mean = float(velocities.mean())
        vel_std  = float(velocities.std())

        return {
            f"{pn}_COP_Length(mm)":    round(length, 2),
            f"{pn}_COP_Deviation(mm)": round(deviation, 2),
            f"{pn}_COP_Vel_Mean(mm/s)":round(vel_mean, 2),
            f"{pn}_COP_Vel_Std(mm/s)": round(vel_std, 2),
            f"{pn}_COP_Row_Range(mm)": round(float(np.ptp(rows)), 2),
            f"{pn}_COP_Col_Range(mm)": round(float(np.ptp(cols)), 2),
            f"{pn}_COP_Row_Mean(mm)":  round(float(rows.mean()), 2),
            f"{pn}_COP_Col_Mean(mm)":  round(float(cols.mean()), 2),
        }

    # ── 4발 합산 CoP ─────────────────────────────
    def _total_cop_features(self,
                             traj_dict: Dict[int, List[CopPoint]]) -> Dict:
        """
        4발 COP 를 GRF 가중 평균하여 전체 무게 중심 궤적 계산.
        보행 중 전체 균형 흔들림 분석.
        """
        # 가장 짧은 궤적 길이로 맞춤
        min_len = min(len(t) for t in traj_dict.values() if t)
        if min_len < 2:
            return {"Total_COP_Length(mm)":   0.0,
                    "Total_COP_Deviation(mm)": 0.0,
                    "Total_CoP_Vel_Std(mm/s)": 0.0}

        total_rows, total_cols = [], []
        for i in range(min_len):
            w_sum = 0.0
            r_sum = 0.0
            c_sum = 0.0
            for pos, traj in traj_dict.items():
                if i < len(traj):
                    pt = traj[i]
                    w_sum += pt.grf
                    r_sum += pt.row_mm * pt.grf
                    c_sum += pt.col_mm * pt.grf
            if w_sum > 1e-6:
                total_rows.append(r_sum / w_sum)
                total_cols.append(c_sum / w_sum)

        if len(total_rows) < 2:
            return {"Total_COP_Length(mm)":   0.0,
                    "Total_COP_Deviation(mm)": 0.0,
                    "Total_CoP_Vel_Std(mm/s)": 0.0}

        rows = np.array(total_rows)
        cols = np.array(total_cols)
        dists = np.sqrt(np.diff(rows)**2 + np.diff(cols)**2)
        center_r = rows.mean()
        center_c = cols.mean()
        devs = np.sqrt((rows - center_r)**2 + (cols - center_c)**2)

        dt = 1.0 / FPS
        velocities = dists / dt

        return {
            "Total_COP_Length(mm)":    round(float(dists.sum()), 2),
            "Total_COP_Deviation(mm)": round(float(devs.mean()), 2),
            "Total_CoP_Vel_Std(mm/s)": round(float(velocities.std()), 2),
        }

    # ── 메인 분석 ────────────────────────────────
    def analyze(self, synced_frames: List[SyncedFrame]) -> Dict:
        """
        한 보행 주기의 SyncedFrame 리스트로 COP 특징 계산.

        Args:
            synced_frames: 한 주기의 SyncedFrame 리스트

        Returns:
            COP 특징 딕셔너리 (~24개 특징)
        """
        if not synced_frames:
            return {}

        # 발별 COP 궤적 구성
        traj_dict: Dict[int, List[CopPoint]] = {i: [] for i in range(NUM_PAWS)}
        dt = 1.0 / FPS

        for frame_idx, synced in enumerate(synced_frames):
            t = frame_idx * dt
            for pos in range(NUM_PAWS):
                mat = synced.get_matrix(pos)
                pt  = self._matrix_to_cop(mat, pos)
                if pt:
                    pt.time = t
                    traj_dict[pos].append(pt)

        # 발별 특징
        features = {}
        for pos in range(NUM_PAWS):
            feat = self._trajectory_features(traj_dict[pos], pos)
            features.update(feat)

        # 전체(LF+RF) 합산 CoP
        total_feat = self._total_cop_features(traj_dict)
        features.update(total_feat)

        return features


# ── 단독 실행 테스트 ───────────────────────────
if __name__ == "__main__":
    import asyncio
    from ble_receiver import DummyPawReceiver
    from frame_sync import FrameSynchronizer

    cop     = CopAnalyzer()
    buffer  = []
    count   = [0]

    def on_synced(synced: SyncedFrame):
        buffer.append(synced)
        # 50프레임 (1초) 모이면 분석
        if len(buffer) >= 50:
            features = cop.analyze(buffer)
            count[0] += 1
            print(f"\n[COP 분석 #{count[0]}]")
            for k, v in features.items():
                print(f"  {k}: {v}")
            buffer.clear()

    sync  = FrameSynchronizer(on_synced=on_synced)
    dummy = DummyPawReceiver(on_frame=sync.on_frame)

    print("=== cop_analyzer 테스트 ===")
    asyncio.run(dummy.run(hz=50))
