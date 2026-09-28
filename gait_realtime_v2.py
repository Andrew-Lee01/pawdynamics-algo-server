"""
gait_realtime_v2.py — 전처리 파이프라인 v2 [2발 모드: LF/RF]
=============================================
1학기 gait_realtime.py 확장판.

변경사항:
    - LF/RF 2발 동시 처리 (LH/RH 는 이번 실험에서 제외)
    - IMU 지형 보정 적용 (GRF × cos(pitch), LF 기준)
    - 특징: 1학기 LF/RF 기반 25개 + IMU 특징
      · LH/RH 전용 특징(전후 SI, 대각선 SI, 체중분배율, 뒷발 SI)은
        발이 2개뿐이라 의미가 없어 제거했습니다.

입력:  SyncedFrame (frame_sync.py)
출력:  feature_dict (gait_ensemble_v2.py 입력)

사용법:
    from gait_realtime_v2 import GaitPreprocessorV2
    preprocessor = GaitPreprocessorV2(robot_weight_kg=15.0)
    features = preprocessor.process(synced_frame)
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from scipy.interpolate import interp1d

from frame_sync import SyncedFrame, NUM_PAWS
from node_calibration import load_calibration, apply_calibration
from cross_sensor_calib import load_gains
from imu_terrain_calib import get_realtime_correction

log = logging.getLogger("GAIT_V2")

# ── 상수 ──────────────────────────────────────
MATRIX_ROWS    = 16
MATRIX_COLS    = 10
ROBOT_WEIGHT   = 15.0    # kg (기본값, 초기화 시 변경 가능)
KGF_SCALE      = 1.0     # 노드별 환산 합계에 곱할 실측 보정계수 (kgf_check.py 로 무게 2점 이상 측정 후 설정)
N_TIMEPOINTS   = 100     # 시간 정규화 포인트 수 (1학기와 동일)
SENSOR_AREA_CM2 = 0.25   # 노드 1개 면적 (5mm × 5mm)
FPS            = 50      # 샘플링 주파수 (Hz)
STANCE_THRESH  = 0.02    # 입각기 판별 임계값 (%BW)
# [2026-09-24] SnowForce3 경로는 무부하 노이즈가 없어서(ESP32 LF와 다름) 3.0은 너무 높음
# [임시방편, 2026-09-23] 원래 0.05 였으나 LF 상시 노이즈(무부하 GRF 0.87~0.95%BW,
# 시간에 따라 더 변할 수 있음)가 이 값을 넘어 "항상 입각"으로 오판되는 문제 발견 →
# 노이즈보다 확실히 위, 실제 로봇 보행 하중보다는 훨씬 아래인 3.0%BW 로 올림.
# 근본 해결 아님 — 노이즈 노드((7,0)/(14,4)/(14,9)/(15,9) 등) 배선 점검이 진짜 원인.
# 노이즈가 이 값보다 더 커지면 다시 같은 문제가 재발할 수 있음.

# 발 위치 인덱스 [2발 모드]
LF, RF = 0, 1
POS_NAMES = ["LF", "RF"]


# ── 단일 발 처리 결과 ──────────────────────────
@dataclass
class SinglePawResult:
    """한 발의 전처리 결과"""
    position:       int
    time_series:    np.ndarray    # [100] %BW 정규화 파형
    grf_peak:       float         # 최대 GRF (%BW)
    grf_mean:       float         # 평균 GRF (%BW)
    loading_rate:   float         # 하중률 (kgf/s)
    stance_ratio:   float         # 입각기 비율 (%)
    contact_area:   float         # 접촉 면적 (cm²)
    cop_row:        float         # COP 행 방향 위치
    cop_col:        float         # COP 열 방향 위치
    cop_row_range:  float         # COP 행 방향 이동 범위
    cop_col_range:  float         # COP 열 방향 이동 범위
    foot_angle:     float         # 발 각도 추정 (°)
    stride_length:  float         # 보폭 추정 (mm)
    cycle_duration: float         # 보행 주기 (s)
    stance_duration:float         # 입각기 지속 시간 (s)
    swing_duration: float         # 유각기 지속 시간 (s)
    is_valid:       bool = True   # 유효한 주기인지


# ── 전처리기 v2 ────────────────────────────────
class GaitPreprocessorV2:
    """
    LF/RF 2발 + IMU 전처리 파이프라인. [2발 모드]

    1. 보정 적용 (노드별 + 센서간 + 지형)
    2. 체중 정규화 (%BW)
    3. 입각기 검출
    4. 시간 정규화 (100포인트)
    5. 특징 추출
    """

    def __init__(self, robot_weight_kg: float = ROBOT_WEIGHT):
        self.weight = robot_weight_kg
        # 보정 파일 로드
        self._node_coeffs = [load_calibration(pos) for pos in range(NUM_PAWS)]
        self._gains        = load_gains()
        # 보행 주기 버퍼 (입각기 검출용)
        self._stance_flags: Dict[int, bool]  = {i: False for i in range(NUM_PAWS)}
        self._cycle_frames: Dict[int, List]  = {i: []    for i in range(NUM_PAWS)}
        self._completed:    Dict[int, Optional[SinglePawResult]] = {
            i: None for i in range(NUM_PAWS)
        }
        log.info(f"GaitPreprocessorV2 초기화 (체중={robot_weight_kg}kg)")

    # ── 행렬 → GRF 합계 ─────────────────────────
    def _matrix_to_grf(self, mat: list, pos: int,
                        pitch_deg: float = 0.0) -> float:
        """
        행렬 합계 → kgf 변환 → 지형 보정 → %BW 변환

        Args:
            mat:       [16][10] raw 행렬
            pos:       발 위치
            pitch_deg: IMU pitch (지형 보정용)
        Returns:
            GRF in %BW
        """
        # 노드별 보정
        corrected = apply_calibration(mat, self._node_coeffs[pos])
        # ADC → 전압 → kgf 변환 (설계자료 4-4 Kitronyx 곡선 근사)
        # 노드값은 0~4095(=0~1.65V) 스케일이므로 합계가 아니라 노드마다 환산한 뒤 더한다.
        # (합계를 전압으로 보면 노드 몇 개만 켜져도 4095 를 넘어 7kgf 로 포화됨)
        voltage = corrected / 4095.0 * 1.65   # V, [16][10]
        kgf_nodes = np.interp(voltage, self._V_TABLE, self._KGF_TABLE,
                              left=0.0, right=self._KGF_TABLE[-1])
        # 센서 간 이득 보정 + 실측 스케일(KGF_SCALE)
        kgf = float(kgf_nodes.sum()) * self._gains.get(pos, 1.0) * KGF_SCALE
        # 지형 보정 (GRF × cos(pitch))
        terrain_coeff = get_realtime_correction(pitch_deg)
        kgf_corrected = kgf * terrain_coeff
        # %BW 변환
        return kgf_corrected / self.weight * 100.0

    # (전압, kgf) 보간 테이블 — 설계자료 4-4 표
    _V_TABLE   = np.array([0.11, 0.42, 1.24, 1.91, 2.18, 2.34, 2.42])
    _KGF_TABLE = np.array([0.00, 0.25, 0.50, 1.00, 2.00, 4.00, 7.00])

    def _voltage_to_kgf(self, voltage: float) -> float:
        """
        Kitronyx 저항 곡선 기반 선형 보간 (노드 1개의 전압 → kgf).
        설계자료 4-4 표 참조.
        """
        return float(np.interp(voltage, self._V_TABLE, self._KGF_TABLE,
                               left=0.0, right=self._KGF_TABLE[-1]))

    # ── COP 계산 ─────────────────────────────────
    def _compute_cop(self, mat: list) -> tuple:
        """행렬에서 COP(압력 중심) 위치 계산"""
        arr = np.array(mat, dtype=float)
        total = arr.sum()
        if total < 1e-6:
            return MATRIX_ROWS / 2, MATRIX_COLS / 2
        rows = np.arange(MATRIX_ROWS).reshape(-1, 1)
        cols = np.arange(MATRIX_COLS).reshape(1, -1)
        cop_row = (arr * rows).sum() / total
        cop_col = (arr * cols).sum() / total
        return cop_row, cop_col

    # ── 입각기 검출 ──────────────────────────────
    def _detect_stance(self, grf_pct: float) -> bool:
        """GRF %BW 가 임계값 이상이면 입각기"""
        return grf_pct > STANCE_THRESH

    # ── 단일 발 특징 추출 ────────────────────────
    def _extract_single_features(
            self, frames: List[float], pos: int) -> SinglePawResult:
        """
        1학기 특징 추출 구조 유지 + 확장.

        Args:
            frames: GRF %BW 값 리스트 (가변 길이)
            pos:    발 위치
        Returns:
            SinglePawResult
        """
        arr = np.array(frames, dtype=float)
        n   = len(arr)
        if n < 3:
            return SinglePawResult(
                position=pos, time_series=np.zeros(N_TIMEPOINTS),
                grf_peak=0, grf_mean=0, loading_rate=0,
                stance_ratio=0, contact_area=0,
                cop_row=0, cop_col=0, cop_row_range=0, cop_col_range=0,
                foot_angle=0, stride_length=0,
                cycle_duration=0, stance_duration=0, swing_duration=0,
                is_valid=False)

        # 시간 정규화 (100포인트)
        x_orig = np.linspace(0, 100, n)
        x_new  = np.linspace(0, 100, N_TIMEPOINTS)
        f      = interp1d(x_orig, arr, kind="linear")
        ts     = f(x_new)

        # 입각기 구간
        stance_mask = arr > STANCE_THRESH
        stance_n    = stance_mask.sum()
        cycle_dur   = n / FPS
        stance_dur  = stance_n / FPS
        swing_dur   = max(0, cycle_dur - stance_dur)
        stance_rat  = stance_n / n * 100.0

        # 하중 특징
        grf_peak = float(arr.max())
        grf_mean = float(arr[stance_mask].mean()) if stance_n > 0 else 0.0

        # 하중률 (앞 20% 구간 기울기)
        load_end = max(1, int(n * 0.2))
        loading_rate = (arr[load_end] - arr[0]) / (load_end / FPS + 1e-9)

        return SinglePawResult(
            position       = pos,
            time_series    = ts,
            grf_peak       = grf_peak,
            grf_mean       = grf_mean,
            loading_rate   = loading_rate,
            stance_ratio   = stance_rat,
            contact_area   = stance_n * SENSOR_AREA_CM2,
            cop_row        = 0.0,   # COP 는 cop_analyzer.py 에서 계산
            cop_col        = 0.0,
            cop_row_range  = 0.0,
            cop_col_range  = 0.0,
            foot_angle     = 0.0,
            stride_length  = stride_dur * 200 if (stride_dur := cycle_dur) else 0,
            cycle_duration = cycle_dur,
            stance_duration= stance_dur,
            swing_duration = swing_dur,
        )

    # ── 대칭성 지수 ──────────────────────────────
    @staticmethod
    def symmetry_index(a: float, b: float) -> float:
        """
        SI = |a - b| / ((a + b) / 2) × 100
        0% = 완전 대칭, 높을수록 비대칭
        """
        denom = (a + b) / 2.0
        if denom < 1e-9:
            return 0.0
        return abs(a - b) / denom * 100.0

    # ── 메인 특징 추출 ────────────────────────────
    def extract_features(
            self, results: Dict[int, SinglePawResult],
            imu_pitch: float = 0.0,
            imu_roll:  float = 0.0,
            imu_accel_rms: float = 0.0) -> Dict:
        """
        LF/RF 2발 결과에서 특징 추출. [2발 모드]

        Returns:
            feature_dict (gait_ensemble_v2.py 입력 형식)
        """
        def get(pos): return results.get(pos)

        lf = get(LF); rf = get(RF)

        features = {}

        # ── 1학기 기존 25개 특징 (LF/RF 기반) ──────
        for pos, r in [(LF, lf), (RF, rf)]:
            pn = POS_NAMES[pos]
            if r:
                features[f"{pn}_Cycle_Duration(s)"]    = r.cycle_duration
                features[f"{pn}_Stance_Duration(s)"]   = r.stance_duration
                features[f"{pn}_Swing_Duration(s)"]    = r.swing_duration
                features[f"{pn}_Stance_Ratio(%)"]      = r.stance_ratio
                features[f"{pn}_GRF_Peak(kgf)"]        = r.grf_peak
                features[f"{pn}_Loading_Rate(kgf/s)"]  = r.loading_rate
                features[f"{pn}_Contact_Area(cm2)"]    = r.contact_area
                features[f"{pn}_COP_Row_Range(mm)"]    = r.cop_row_range
                features[f"{pn}_COP_Col_Range(mm)"]    = r.cop_col_range
                features[f"{pn}_Foot_Angle_est(deg)"]  = r.foot_angle
                features[f"{pn}_Stride_Length_est(mm)"]= r.stride_length
            else:
                for k in ["Cycle_Duration(s)", "Stance_Duration(s)",
                          "Swing_Duration(s)", "Stance_Ratio(%)",
                          "GRF_Peak(kgf)", "Loading_Rate(kgf/s)",
                          "Contact_Area(cm2)", "COP_Row_Range(mm)",
                          "COP_Col_Range(mm)", "Foot_Angle_est(deg)",
                          "Stride_Length_est(mm)"]:
                    features[f"{pn}_{k}"] = 0.0

        # 좌우 대칭성 (앞발)
        lf_peak = lf.grf_peak if lf else 0.0
        rf_peak = rf.grf_peak if rf else 0.0
        lf_st   = lf.stance_duration if lf else 0.0
        rf_st   = rf.stance_duration if rf else 0.0
        features["SI_GRF_Peak(%)"]        = self.symmetry_index(lf_peak, rf_peak)
        features["SI_Stance_Duration(%)"] = self.symmetry_index(lf_st, rf_st)
        features["Force_Diff_peak(kgf)"]  = abs(lf_peak - rf_peak)

        # [2발 모드] LH/RH 전용 특징(전후 SI, 대각선 SI, 체중분배율, 뒷발 SI)은
        # 발이 2개뿐이라 정의할 수 없어 제거했습니다. LH/RH 가 생기면 복원하세요.

        # IMU 특징
        features["IMU_Pitch(deg)"]    = imu_pitch
        features["IMU_Roll(deg)"]     = imu_roll
        features["IMU_Accel_RMS"]     = imu_accel_rms

        # 시계열 저장 (HMM/DTW 용)
        for pos, r in results.items():
            if r:
                features[f"{POS_NAMES[pos]}_time_series"] = r.time_series.tolist()
            else:
                features[f"{POS_NAMES[pos]}_time_series"] = [0.0] * N_TIMEPOINTS

        return features

    # ── 프레임 단위 처리 ─────────────────────────
    def process_frame(self, synced: SyncedFrame) -> Optional[Dict]:
        """
        동기화된 프레임 1개 처리.

        입각기가 완성된 발이 있으면 특징 딕셔너리 반환.
        아직 주기가 완성되지 않았으면 None 반환.
        """
        imu = synced.get_imu(LF)
        pitch = imu.pitch_deg
        accel_rms = float(np.sqrt(
            imu.accel_x**2 + imu.accel_y**2 + imu.accel_z**2
        ))

        for pos in range(NUM_PAWS):
            mat = synced.get_matrix(pos)
            grf = self._matrix_to_grf(mat, pos, pitch)
            in_stance = self._detect_stance(grf)

            if in_stance:
                self._cycle_frames[pos].append(grf)
                self._stance_flags[pos] = True
            elif self._stance_flags[pos]:
                # 입각기 → 유각기 전환 → 주기 완성
                frames = self._cycle_frames[pos]
                if len(frames) >= 3:   # 최소 3프레임 이상 [2026-09-24] 빠른 보행 탭 대응, 5→3
                    result = self._extract_single_features(frames, pos)
                    self._completed[pos] = result
                self._cycle_frames[pos] = []
                self._stance_flags[pos] = False

        # LF/RF 모두 완성됐으면 특징 반환
        if all(self._completed[p] is not None for p in range(NUM_PAWS)):
            features = self.extract_features(
                self._completed,
                imu_pitch     = pitch,
                imu_roll      = imu.roll_deg,
                imu_accel_rms = accel_rms,
            )
            # 완성된 결과 초기화
            for pos in range(NUM_PAWS):
                self._completed[pos] = None
            return features

        return None


# ── 단독 실행 테스트 ───────────────────────────
if __name__ == "__main__":
    import asyncio
    from ble_receiver import DummyPawReceiver
    from frame_sync import FrameSynchronizer

    preprocessor = GaitPreprocessorV2(robot_weight_kg=15.0)
    feature_count = [0]

    def on_synced(synced):
        feat = preprocessor.process_frame(synced)
        if feat:
            feature_count[0] += 1
            si = feat.get("SI_GRF_Peak(%)", 0)
            print(f"[특징 #{feature_count[0]:03d}] "
                  f"SI_GRF={si:.1f}% "
                  f"pitch={feat.get('IMU_Pitch(deg)', 0):.1f}°")

    sync = FrameSynchronizer(on_synced=on_synced)
    dummy = DummyPawReceiver(on_frame=sync.on_frame)

    print("=== gait_realtime_v2 테스트 (LF/RF 2발 모드) ===")
    asyncio.run(dummy.run(hz=50))
