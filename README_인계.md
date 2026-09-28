# PawDynamics — 알고리즘 모듈(`process_and_score`) 인계

이 폴더 통째로 보내면 됩니다. **`algo_server.py` 파일 하나만 보내면 동작하지
않습니다** — 그 파일이 안에서 import 하는 다른 파이썬 파일들과, 학습된
모델/보정 데이터 파일들이 같이 있어야 합니다. 이 폴더에는 그 전부가 이미
포함되어 있고, 실제로 실행해서 확인까지 했습니다.

## 쓰는 법

```python
from algo_server import process_and_score

result = process_and_score(lf_matrix, rf_matrix, pitch_deg=0.0)
# lf_matrix, rf_matrix: 16x10 리스트(0~1 정규화 값이든 0~4095 raw 값이든 자동 판별)
# pitch_deg: IMU pitch(도), 없으면 0.0

# result 예시:
# {
#   "score": 92.0,        # 0~100, 높을수록 좌우 대칭(정상)
#   "status": "normal",    # "normal" / "warning" / "abnormal"
#   "asymmetry": 0.05,      # 0~1, 낮을수록 정상
#   "detail": {...}         # ML/HMM/DTW 개별 점수 등 참고용
# }
```

`process_and_score()`는 매 호출마다 항상 유효한 dict를 반환합니다(`None`이
오는 경우 없음). 입각→유각 한 보행 주기가 아직 안 끝났으면 "가장 최근 판정"
(또는 첫 호출 전엔 중립 기본값 `score=50, status=normal`)을 그대로 돌려줍니다.

## 설치해야 하는 패키지

```bash
pip install numpy scipy pandas scikit-learn hmmlearn joblib bleak websockets
```

(FastAPI로 감쌀 경우 `pip install fastapi uvicorn`도 추가로 필요합니다.)

## 폴더 안 파일 설명

| 파일 | 역할 |
|---|---|
| `algo_server.py` | **여기서 `process_and_score()` 만 갖다 쓰면 됨** |
| `gait_ensemble_v2.py` | ML+HMM+DTW 앙상블 (핵심 판정 로직) |
| `gait_realtime_v2.py` | 압력 매트릭스+IMU → %BW GRF, 보행 주기 특징 추출 |
| `cop_analyzer.py` | 압력중심(COP) 궤적 특징 |
| `gait_ml_v2.py` / `gait_hmm_v2.py` / `gait_dtw_v2.py` / `gait_sym_scorer.py` | 앙상블을 구성하는 개별 스코어러 |
| `node_calibration.py` / `cross_sensor_calib.py` / `imu_terrain_calib.py` | 센서별 보정(현재 LF/RF 노드 보정 파일은 없어서 1.0 기본값 사용 — 정상 동작) |
| `ble_receiver.py` / `frame_sync.py` | 원래 BLE 수신용 타입 정의만 재사용(`ImuData` 등) — 이 서버는 BLE로 직접 안 붙음 |
| `gait_ml_model_v2.pkl`, `hmm_2state.pkl`, `calib/*` | 학습된 모델·보정 데이터. **없으면 위 모듈들이 로드에 실패합니다** |

## 확인한 것

- 이 폴더만 갖고 `python algo_server.py` 실행 → 모델 전부 정상 로드, 자체
  테스트 5회 호출 모두 정상 반환 확인
- 압력 눌림/떼기를 반복하는 가상 보행 시뮬레이션으로 실제 보행 주기 완성 →
  `abnormal` 판정까지 끝까지 확인(중립 기본값이 아닌 실제 계산값 나옴)
- Windows 콘솔 이모지(🔴/🟢) 인코딩 문제로 결과가 통째로 날아가던 버그
  (`gait_ensemble_v2.py`의 `_print_result()`)는 이미 고쳐진 상태로 포함됨
