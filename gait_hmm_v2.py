# -*- coding: utf-8 -*-
"""
gait_hmm_v2.py — HMM 보행 리듬 분석 v2
==========================================
1학기 hmm_score() 함수 수정판.

주요 변경사항:
    1. 3상태 → 2상태 (로봇 강아지 특성 반영)
       - 기존: Loading / Peak Stance / Swing
       - 수정: Landing / Swing
       - 이유: 로봇 강아지는 순간적으로 쾅 찍히는 방식
               Loading과 Peak 구분이 의미 없음

    2. LOG_MAX 오류 수정
       - 기존: HMM_LOG_MAX = -40  (오류)
       - 수정: HMM_LOG_MAX = -93.11 (실측 정상 최댓값)
       - 효과: 정상 데이터 점수 0.729 고정 문제 해결

    3. 4발(LF/RF/LH/RH) 대각선 동기화 추가
       - LF↔RH, RF↔LH 대각선 발 동시 접지 패턴 분석
"""

import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional

log = logging.getLogger("HMM_V2")

# ── HMM 기준값 (실측값 기반으로 수정) ────────────
# 1학기 오류값
HMM_LOG_MAX_V1 = -40.0
HMM_LOG_MIN_V1 = -300.0

# 2학기 수정값 (2상태 HMM 실측 기반)
HMM_LOG_MAX = 81.19    # 실측 정상 데이터 최댓값 (2026-09-24 재학습, 202개 정상 데이터 포함 562행)
HMM_LOG_MIN = -190.0   # 실측 비정상 예상 최솟값

# 수의학 기준 SI 임계값
HMM_SI_THRESHOLD = 15.0   # Voss 2007

# 2상태 HMM 모델 경로
MODEL_PATH_V2 = Path('hmm_2state.pkl')
MODEL_PATH_V1 = Path('normal_gait_hmm.pkl')   # 폴백용

# Landing 상태 인덱스 (압력 높은 쪽)
# 학습 결과: 상태0=Landing(LF=17.13), 상태1=Swing(LF=9.80)
LANDING_IDX = 0
SWING_IDX   = 1


# ── 대각선 동기화 점수 ────────────────────────────
def _diagonal_sync_score(lf_series: np.ndarray,
                          rf_series: np.ndarray,
                          lh_series: np.ndarray,
                          rh_series: np.ndarray) -> float:
    """
    LF↔RH, RF↔LH 대각선 발 동기화 점수.

    4족 보행 정상 패턴:
        LF(왼앞) ↔ RH(오른뒤) 동시 접지
        RF(오른앞) ↔ LH(왼뒤) 동시 접지

    Returns:
        비정상 점수 (0=정상, 1=비정상)
    """
    thresh = 0.05   # 입각기 판별 임계값 (%BW)

    lf_s = (lf_series[:20] > thresh).astype(float)
    rf_s = (rf_series[:20] > thresh).astype(float)
    lh_s = (lh_series[:20] > thresh).astype(float)
    rh_s = (rh_series[:20] > thresh).astype(float)

    # LF-RH 상관계수 (정상: 높아야 함)
    corr_lf_rh = float(np.corrcoef(lf_s, rh_s)[0, 1]) \
        if lf_s.std() > 0 and rh_s.std() > 0 else 0.0

    # RF-LH 상관계수 (정상: 높아야 함)
    corr_rf_lh = float(np.corrcoef(rf_s, lh_s)[0, 1]) \
        if rf_s.std() > 0 and lh_s.std() > 0 else 0.0

    avg_corr = (corr_lf_rh + corr_rf_lh) / 2.0
    avg_corr = float(np.clip(avg_corr, -1, 1))

    # 상관 낮을수록 비정상
    return float(np.clip((1.0 - avg_corr) / 2.0, 0, 1))


# ── 앞뒤 리듬 대칭성 점수 ────────────────────────
def _front_rear_rhythm_score(lf_series: np.ndarray,
                              rf_series: np.ndarray,
                              lh_series: np.ndarray,
                              rh_series: np.ndarray) -> float:
    """
    앞발 합계 vs 뒷발 합계 입각기 타이밍 대칭성.

    Returns:
        비정상 점수 (0=정상, 1=비정상)
    """
    thresh = 0.05
    front = (lf_series[:20] + rf_series[:20]) / 2
    rear  = (lh_series[:20] + rh_series[:20]) / 2

    front_stance = float((front > thresh).sum())
    rear_stance  = float((rear  > thresh).sum())

    denom = 0.5 * (front_stance + rear_stance)
    if denom < 1e-6:
        return 0.0

    si = abs(front_stance - rear_stance) / denom * 100.0
    return float(np.clip(si / HMM_SI_THRESHOLD, 0, 1))


# ── HMM 점수 계산기 v2 ───────────────────────────
class HMMScorerV2:
    """
    2상태 HMM 기반 보행 리듬 분석기.

    1학기와 비교:
        기존: 3상태 (Loading / Peak / Swing)
              LOG_MAX = -40 (오류)
        v2:   2상태 (Landing / Swing)
              LOG_MAX = -93.11 (실측값)

    점수 구성 (4발 모드):
        P_evasion    (30%): HMM log likelihood 이탈도
        P_temporal   (30%): 앞발 좌우 입각기 대칭성
        P_diagonal   (25%): 대각선 발 동기화
        P_front_rear (15%): 앞뒤 발 리듬 대칭성

    점수 구성 (2발 호환 모드):
        P_evasion    (50%): HMM log likelihood 이탈도
        P_temporal   (50%): 앞발 좌우 입각기 대칭성
    """

    def __init__(self):
        self.hmm_model  = None
        self.n_states   = 0
        self.landing_idx = LANDING_IDX
        self.swing_idx   = SWING_IDX
        self._v1_mode    = False

        # 2상태 모델 먼저 시도
        if MODEL_PATH_V2.exists():
            self.hmm_model = joblib.load(MODEL_PATH_V2)
            self.n_states  = self.hmm_model.n_components
            log.info(f"HMM v2 로드 ({self.n_states}상태, "
                     f"LOG_MAX={HMM_LOG_MAX})")

        # 없으면 1학기 3상태 모델 폴백
        elif MODEL_PATH_V1.exists():
            self.hmm_model = joblib.load(MODEL_PATH_V1)
            self.n_states  = self.hmm_model.n_components
            self._v1_mode  = True
            log.warning(f"HMM v1 폴백 ({self.n_states}상태, "
                        f"LOG_MAX={HMM_LOG_MAX_V1} 오류값)")
        else:
            log.error("HMM 모델 없음 — 재학습 필요")

    def score(self,
              lf_series: List[float],
              rf_series: List[float],
              lh_series: Optional[List[float]] = None,
              rh_series: Optional[List[float]] = None
              ) -> Tuple[float, Dict]:
        """
        HMM 비정상 점수 계산 (0=정상, 1=비정상).

        Args:
            lf_series: 왼앞발 시계열 (100포인트 %BW)
            rf_series: 오른앞발 시계열
            lh_series: 왼뒷발 시계열 (없으면 2발 모드)
            rh_series: 오른뒷발 시계열

        Returns:
            (hmm_score, detail_dict)
        """
        if self.hmm_model is None:
            log.warning("모델 없음 → 0.5 반환")
            return 0.5, {'error': 'no_model'}

        lf = np.array(lf_series, dtype=float)
        rf = np.array(rf_series, dtype=float)
        has_4paw = lh_series is not None and rh_series is not None

        if has_4paw:
            lh = np.array(lh_series, dtype=float)
            rh = np.array(rh_series, dtype=float)

        # ── P_evasion: HMM log likelihood ──────────────
        x_strike = np.column_stack([lf[:20], rf[:20]])

        # v1/v2 기준값 분기
        log_max = HMM_LOG_MAX_V1 if self._v1_mode else HMM_LOG_MAX
        log_min = HMM_LOG_MIN_V1 if self._v1_mode else HMM_LOG_MIN

        log_lik = float(np.clip(
            self.hmm_model.score(x_strike),
            log_min, log_max
        ))
        p_evasion = float(np.clip(
            1.0 - (log_lik - log_min) / (log_max - log_min),
            0, 1
        ))

        # ── P_temporal: 입각기 대칭성 ──────────────────
        h = self.hmm_model.predict(x_strike)

        if self._v1_mode:
            # 1학기: 상태0 = Loading 으로 가정
            lf_st = float(np.sum(h == 0))
            rf_st = float(np.sum(h != 0))
        else:
            # 2학기: Landing 상태 기준
            lf_st = float(np.sum(h == self.landing_idx))
            rf_st = float(np.sum(h == self.swing_idx))

        denom = 0.5 * (lf_st + rf_st)
        si    = abs(lf_st - rf_st) / denom * 100 \
                if denom > 1e-6 else 0.0
        p_temporal = float(np.clip(si / HMM_SI_THRESHOLD, 0, 1))

        # ── 4발 추가 지표 ───────────────────────────────
        if has_4paw:
            p_diagonal   = _diagonal_sync_score(lf, rf, lh, rh)
            p_front_rear = _front_rear_rhythm_score(lf, rf, lh, rh)
        else:
            p_diagonal   = (p_evasion + p_temporal) / 2.0
            p_front_rear = p_temporal

        # ── 최종 가중 합산 ──────────────────────────────
        if has_4paw:
            final = (0.30 * p_evasion  +
                     0.30 * p_temporal +
                     0.25 * p_diagonal +
                     0.15 * p_front_rear)
        else:
            final = 0.50 * p_evasion + 0.50 * p_temporal

        final = float(np.clip(final, 0, 1))

        detail = {
            'log_likelihood': round(log_lik, 3),
            'p_evasion':      round(p_evasion, 4),
            'SI_front(%)':    round(si, 2),
            'p_temporal':     round(p_temporal, 4),
            'p_diagonal':     round(p_diagonal, 4),
            'p_front_rear':   round(p_front_rear, 4),
            'hmm_score':      round(final, 4),
            'n_states':       self.n_states,
            'mode': '4paw' if has_4paw else '2paw',
            'v1_compat': self._v1_mode,
        }

        return final, detail

    def retrain(self,
                csv_path: str = 'gait_collected_v2.csv',
                n_states: int = 2):
        """
        실측 데이터로 2상태 HMM 재학습.

        Args:
            csv_path: data_collector_v2.py 수집 CSV
            n_states: 은닉 상태 수 (기본 2)
        """
        import pandas as pd, ast
        from hmmlearn import hmm as hmmlearn

        df     = pd.read_csv(csv_path)
        normal = df[df['label'] == 0]

        X_list, lengths = [], []
        for _, row in normal.iterrows():
            lf = np.array(ast.literal_eval(row['LF_time_series']))
            rf = np.array(ast.literal_eval(row['RF_time_series']))
            x  = np.column_stack([lf[:20], rf[:20]])
            X_list.append(x)
            lengths.append(20)

        X_all = np.vstack(X_list)

        model = hmmlearn.GaussianHMM(
            n_components    = n_states,
            covariance_type = 'diag',
            n_iter          = 200,
            random_state    = 42
        )
        model.fit(X_all, lengths)

        # Landing 상태 자동 판별 (압력 높은 쪽)
        means = model.means_
        self.landing_idx = int(np.argmax(means[:, 0]))
        self.swing_idx   = 1 - self.landing_idx

        joblib.dump(model, MODEL_PATH_V2)
        self.hmm_model = model
        self.n_states  = n_states
        self._v1_mode  = False

        # LOG_MAX 실측값으로 갱신
        lls = []
        for i in range(0, len(X_list)):
            x = X_list[i]
            lls.append(model.score(x))
        lls = np.array(lls)

        log.info(f"HMM 재학습 완료 ({n_states}상태)")
        log.info(f"LOG_MAX 실측: {lls.max():.2f}")
        log.info(f"Landing = 상태{self.landing_idx} "
                 f"(평균 압력 {means[self.landing_idx,0]:.2f})")
        log.info(f"Swing   = 상태{self.swing_idx} "
                 f"(평균 압력 {means[self.swing_idx,0]:.2f})")

        return lls.max()


# ── 단독 실행 테스트 ─────────────────────────────
if __name__ == "__main__":
    import pandas as pd, ast
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    scorer = HMMScorerV2()
    df = pd.read_csv('gait_collected_data.csv')

    print(f"\n모델: {scorer.n_states}상태 "
          f"({'v1 호환' if scorer._v1_mode else 'v2'})")
    print(f"LOG_MAX: {HMM_LOG_MAX if not scorer._v1_mode else HMM_LOG_MAX_V1}")

    for label_name, label_val in [('정상', 0), ('비정상', 1)]:
        sub    = df[df['label'] == label_val]
        scores = []
        for _, row in sub.iterrows():
            lf = np.array(ast.literal_eval(row['LF_time_series']))
            rf = np.array(ast.literal_eval(row['RF_time_series']))
            s, _ = scorer.score(lf.tolist(), rf.tolist())
            scores.append(s)
        arr = np.array(scores)
        print(f"{label_name}: {arr.mean():.3f}±{arr.std():.3f} "
              f"(n={len(arr)})")
