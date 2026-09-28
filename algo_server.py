# -*- coding: utf-8 -*-
"""algo_server.py — process_and_score(): 오늘 만든 알고리즘(GaitPreprocessorV2 +
   EnsembleV2: ML+HMM+DTW 앙상블)을 조원이 만들고 있는 폰↔서버 파이프라인이
   바로 import 해서 쓸 수 있는 함수 하나로 감싼 모듈.

   (FastAPI 부분은 조원이 자기 서버 코드에서 직접 감쌀 것이므로 여기엔 없음.
   이 파일은 `from algo_server import process_and_score` 로만 쓰면 된다.)

핵심 난제와 해결:
    조원의 서버는 "그 순간의 매트릭스 스냅샷 하나"를 매 틱마다 넘겨줄 텐데,
    우리 알고리즘(GaitPreprocessorV2)은 "입각→유각 한 주기 전체"가 쌓여야
    특징을 뽑는다.
    → process_and_score() 를 호출할 때마다 프레임 하나씩 파이프라인에 흘려
      넣고, 한 주기가 막 완성된 호출에서만 새 판정을 계산해 캐시해두고,
      그 사이 호출들은 "가장 최근 판정"을 그대로 돌려준다. 그래서 이 함수는
      항상(첫 호출부터) 유효한 dict 를 반환한다 — None 이 오는 경우는 없다.

입력 매트릭스 값 스케일 자동 판별:
    앱의 BleShoeProtocol.decodeMatrix() 는 raw 바이트(0~255)를 255.0 으로
    나눠 0~1 로 준다. 반면 우리 파이프라인(kgf 환산표)은 0~4095(12bit) 스케일을
    전제로 한다. 어느 쪽 스케일로 들어오는지 호출부마다 다를 수 있어, 매트릭스
    최댓값이 1.5 이하면 0~1(정규화됨)로 보고 4080(=255×16)을 곱해 되돌리고,
    그보다 크면 이미 raw 카운트 값으로 보고 그대로 쓴다.

출력 형식(조원 요청 규격):
    {
      "score": 92.0,        # 종합 점수 0~100, 높을수록 좌우 대칭적(정상)
      "status": "normal",    # "normal" / "warning" / "abnormal"
      "asymmetry": 0.05,      # 앙상블 비대칭 점수 0~1, 낮을수록 정상
      "detail": {...}         # ML/HMM/DTW 개별 점수 등 참고용
    }
"""
import logging
from typing import Dict, List

from ble_receiver import ImuData
from gait_realtime_v2 import GaitPreprocessorV2, LF, RF
from cop_analyzer import CopAnalyzer
from gait_ensemble_v2 import EnsembleV2, ROBOT_WEIGHT_KG

log = logging.getLogger("ALGO_SERVER")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

MATRIX_SCALE = 4080.0   # 0~1(정규화) → 0~4080≈4095(우리 파이프라인). 255×16 근사치.
NORMALIZED_MAX = 1.5    # 이 값 이하면 0~1 정규화된 입력으로 판단


# ── SyncedFrame 흉내: GaitPreprocessorV2.process_frame() 이 요구하는
#    .get_matrix(pos) / .get_imu(pos) 두 메서드만 있으면 된다 ──────────
class SimpleSyncedFrame:
    def __init__(self, matrices: Dict[int, list], imu: ImuData):
        self._matrices = matrices
        self._imu = imu

    def get_matrix(self, pos: int) -> list:
        return self._matrices.get(pos, [[0] * 10 for _ in range(16)])

    def get_imu(self, pos: int) -> ImuData:
        return self._imu


def _normalize_matrix(mat: List[List[float]]) -> List[List[int]]:
    """0~1 정규화 값이든 이미 0~4095 raw 값이든, 최댓값을 보고 스케일을
    자동 판별해 0~4095 정수 매트릭스로 맞춘다."""
    flat_max = max((v for row in mat for v in row), default=0.0)
    scale = MATRIX_SCALE if flat_max <= NORMALIZED_MAX else 1.0
    return [[int(round(min(4095.0, max(0.0, v * scale))))
              for v in row] for row in mat]


# ── 알고리즘 상태 (프로세스 전체에서 하나만 유지 — 로봇 1대 가정) ──────
preprocessor = GaitPreprocessorV2(ROBOT_WEIGHT_KG)
cop_analyzer = CopAnalyzer()
ensemble     = EnsembleV2()
_cop_buffer: List[SimpleSyncedFrame] = []

# 아직 한 주기도 안 끝났을 때 돌려줄 중립 기본값
_last_formatted: Dict = {
    "score": 50.0,
    "status": "normal",
    "asymmetry": 0.5,
    "detail": {"note": "첫 보행 주기(입각→유각) 완성 전 — 기본값"},
}

# final(0~1, 높을수록 비정상)에 따른 3단계 상태 구분 임계값.
# final>0.5 는 predict() 안에서 이미 ABNORMAL 로 확정되므로, 그 아래(0.3~0.5)는
# "정상이긴 하지만 앙상블 점수가 애매하게 높다"는 경계 구간으로 warning 처리.
WARNING_THRESH = 0.3


def _format_result(result: Dict) -> Dict:
    final = result["score"]  # 0~1, 높을수록 비정상
    if result["verdict"] == "ABNORMAL":
        status = "abnormal"
    elif final >= WARNING_THRESH:
        status = "warning"
    else:
        status = "normal"

    return {
        "score":     float(result["symmetry"]),  # 0~100, 높을수록 정상
        "status":    status,
        "asymmetry": final,                        # 0~1, 낮을수록 정상
        "detail": {
            "ml_score":            result["ml_score"],
            "hmm_score":           result["hmm_score"],
            "dtw_score":           result["dtw_score"],
            "si_grf_pct":          result["si_grf"],
            "imu_pitch_deg":       result["imu_pitch"],
            "message":             result["message"],
            "paw":                 result["paw"],
            "cycle":               result["cycle"],
            "consecutive_abnormal": result["consecutive"],
            "forced_abnormal":     result["forced_ab"],
            "verdict_raw":         result["verdict"],
            "timestamp":           result["timestamp"],
        },
    }


def process_and_score(lf_matrix: List[List[float]],
                       rf_matrix: List[List[float]],
                       pitch_deg: float = 0.0) -> Dict:
    """
    프레임 하나(양발 압력 매트릭스 16x10 + IMU pitch)를 파이프라인에 흘려넣는다.
    항상 유효한 dict 를 반환한다 — 한 주기가 막 완성됐으면 새로 계산한 값을,
    아니면 가장 최근 판정(또는 아직 없으면 중립 기본값)을 돌려준다.
    """
    global _last_formatted

    synced = SimpleSyncedFrame(
        matrices={LF: _normalize_matrix(lf_matrix), RF: _normalize_matrix(rf_matrix)},
        imu=ImuData(pitch_deg=pitch_deg),
    )
    _cop_buffer.append(synced)

    features = preprocessor.process_frame(synced)
    if features is not None:
        cop_features = cop_analyzer.analyze(_cop_buffer)
        _cop_buffer.clear()
        features.update(cop_features)

        result = ensemble.predict(features, cop_trajs=None)
        _last_formatted = _format_result(result)

    return _last_formatted


if __name__ == "__main__":
    # 간단한 동작 확인용(FastAPI 없이): 압력 매트릭스를 흉내 낸 값을 몇 번 흘려
    # process_and_score() 가 매번 유효한 dict 를 반환하는지만 확인한다.
    dummy_lf = [[0.5] * 10 for _ in range(16)]
    dummy_rf = [[0.1] * 10 for _ in range(16)]
    for i in range(5):
        r = process_and_score(dummy_lf, dummy_rf, pitch_deg=0.0)
        log.info(f"[{i}] score={r['score']} status={r['status']} "
                 f"asymmetry={r['asymmetry']}")
