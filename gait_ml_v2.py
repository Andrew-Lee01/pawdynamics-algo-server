# -*- coding: utf-8 -*-
"""
gait_ml_v2.py — ML 보행 분류 모듈 v2 [2발 모드: LF/RF]
========================================
1학기 gait_ml.py 확장판.

변경사항:
    - 특징 25개 → 47개 확장 (LF/RF 2발 기준)
    - COP 궤적 특징 포함 (LF/RF + 합산)
    - IMU 특징 포함
    - 지형 조건 포함
    - 알고리즘 구조는 1학기와 동일 (RandomForest + SVM)
    - LH/RH 전용 특징(대각선/전후 SI, 체중분배율)은 발이 2개뿐이라 제거

재학습 방법:
    python gait_ml_v2.py --train
    → gait_ml_model_v2.pkl 생성

예측 방법:
    from gait_ml_v2 import MLClassifierV2
    ml = MLClassifierV2()
    score = ml.predict(feature_dict)
"""

import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

log = logging.getLogger("ML_V2")

# ── 특징 컬럼 정의 ─────────────────────────────
# 1학기 기존 25개
FEAT_COLS_V1 = [
    # LF 특징 (11개)
    'LF_Cycle_Duration(s)', 'LF_Stance_Duration(s)', 'LF_Swing_Duration(s)',
    'LF_Stance_Ratio(%)', 'LF_GRF_Peak(kgf)', 'LF_Loading_Rate(kgf/s)',
    'LF_Contact_Area(cm2)', 'LF_COP_Row_Range(mm)', 'LF_COP_Col_Range(mm)',
    'LF_Foot_Angle_est(deg)', 'LF_Stride_Length_est(mm)',
    # RF 특징 (11개)
    'RF_Cycle_Duration(s)', 'RF_Stance_Duration(s)', 'RF_Swing_Duration(s)',
    'RF_Stance_Ratio(%)', 'RF_GRF_Peak(kgf)', 'RF_Loading_Rate(kgf/s)',
    'RF_Contact_Area(cm2)', 'RF_COP_Row_Range(mm)', 'RF_COP_Col_Range(mm)',
    'RF_Foot_Angle_est(deg)', 'RF_Stride_Length_est(mm)',
    # 대칭성 (3개)
    'SI_GRF_Peak(%)', 'SI_Stance_Duration(%)', 'Force_Diff_peak(kgf)',
]

# 2학기 추가 특징 22개 [2발 모드: LF/RF] — LH/RH·대각선·전후 특징은 제거
FEAT_COLS_V2_ADD = [
    # IMU 특징 (3개)
    'IMU_Pitch(deg)', 'IMU_Roll(deg)', 'IMU_Accel_RMS',
    # COP 궤적 특징 — 발당 8개 × 2발 = 16개
    # LF COP
    'LF_COP_Length(mm)', 'LF_COP_Deviation(mm)',
    'LF_COP_Vel_Mean(mm/s)', 'LF_COP_Vel_Std(mm/s)',
    'LF_COP_Row_Mean(mm)', 'LF_COP_Col_Mean(mm)',
    # RF COP
    'RF_COP_Length(mm)', 'RF_COP_Deviation(mm)',
    'RF_COP_Vel_Mean(mm/s)', 'RF_COP_Vel_Std(mm/s)',
    'RF_COP_Row_Mean(mm)', 'RF_COP_Col_Mean(mm)',
    # LF+RF 합산 COP (3개)
    'Total_COP_Length(mm)', 'Total_COP_Deviation(mm)', 'Total_CoP_Vel_Std(mm/s)',
]

# 전체 특징 컬럼 (60개)
FEAT_COLS_ALL = FEAT_COLS_V1 + FEAT_COLS_V2_ADD

MODEL_PATH_V2 = Path('gait_ml_model_v2.pkl')


# ── ML 분류기 v2 ───────────────────────────────
class MLClassifierV2:
    """
    RandomForest + SVM VotingClassifier v2. [2발 모드: LF/RF]
    1학기 구조 그대로 유지, 특징만 47개로 확장.

    보드 도착 전:
        1학기 모델(25개 특징)로 호환 모드 실행
    실측 데이터 수집 후:
        train() 으로 47개 특징 재학습
    """

    def __init__(self, model_path: Path = MODEL_PATH_V2,
                 fallback_path: Path = Path('gait_ml_model.pkl')):
        self.model      = None
        self.feat_cols  = FEAT_COLS_ALL
        self._v1_mode   = False   # 1학기 호환 모드

        # v2 모델 먼저 시도
        if model_path.exists():
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            log.info(f"ML v2 모델 로드 ({len(FEAT_COLS_ALL)}개 특징)")

        # 없으면 1학기 모델로 폴백
        elif fallback_path.exists():
            with open(fallback_path, 'rb') as f:
                self.model = pickle.load(f)
            self.feat_cols = FEAT_COLS_V1
            self._v1_mode  = True
            log.warning(f"ML v1 모델 로드 (호환 모드, {len(FEAT_COLS_V1)}개 특징)")
        else:
            log.error("ML 모델 파일 없음 — train() 으로 학습 필요")

    def _to_array(self, feature_dict: Dict) -> np.ndarray:
        """feature_dict → numpy 배열"""
        return np.array([[feature_dict.get(c, 0.0)
                          for c in self.feat_cols]])

    def predict(self, feature_dict: Dict) -> float:
        """
        비정상 확률 반환 (0~1).

        Args:
            feature_dict: gait_realtime_v2.py 출력 딕셔너리

        Returns:
            비정상 확률 (0=정상, 1=비정상)
        """
        if self.model is None:
            log.warning("모델 없음 → 0.5 반환")
            return 0.5

        X = self._to_array(feature_dict)
        prob = self.model.predict_proba(X)[0]
        # [정상확률, 비정상확률] → 비정상 확률만
        return float(prob[1]) if len(prob) > 1 else float(prob[0])

    def train(self, csv_path: str = 'gait_collected_v2.csv'):
        """
        실측 데이터로 재학습.

        Args:
            csv_path: data_collector_v2.py 로 수집한 CSV
        """
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier, VotingClassifier
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        log.info(f"ML v2 재학습 시작: {csv_path}")
        df = pd.read_csv(csv_path)

        # 특징 존재 확인 — 없는 컬럼은 0으로 채움
        for col in FEAT_COLS_ALL:
            if col not in df.columns:
                df[col] = 0.0
                log.warning(f"컬럼 없음 → 0 채움: {col}")

        X = df[FEAT_COLS_ALL].values
        y = df['label'].values

        log.info(f"학습 데이터: {len(df)}행 "
                 f"(정상={sum(y==0)} 비정상={sum(y==1)})")

        # 1학기와 동일한 모델 구조
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=10,
            random_state=42, n_jobs=-1
        )
        svm = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='rbf', C=1.0,
                        probability=True, random_state=42))
        ])
        voting = VotingClassifier(
            estimators=[('rf', rf), ('svm', svm)],
            voting='soft'
        )
        voting.fit(X, y)

        with open(MODEL_PATH_V2, 'wb') as f:
            pickle.dump(voting, f)

        self.model     = voting
        self.feat_cols = FEAT_COLS_ALL
        self._v1_mode  = False

        # 학습 정확도
        acc = voting.score(X, y)
        log.info(f"ML v2 재학습 완료! 정확도={acc:.3f} → {MODEL_PATH_V2}")
        return acc


# ── 단독 실행 ──────────────────────────────────
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true',
                        help='실측 데이터로 재학습')
    parser.add_argument('--csv', default='gait_collected_v2.csv')
    args = parser.parse_args()

    ml = MLClassifierV2()

    if args.train:
        ml.train(args.csv)
    else:
        # 더미 데이터로 예측 테스트
        dummy = {col: np.random.uniform(0, 10)
                 for col in FEAT_COLS_ALL}
        score = ml.predict(dummy)
        mode  = "v1 호환" if ml._v1_mode else "v2"
        print(f"[{mode}] 비정상 점수: {score:.4f}")
