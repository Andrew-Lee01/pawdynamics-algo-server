# -*- coding: utf-8 -*-
"""
gait_dtw_v2.py — DTW 파형 유사도 분석 v2
==========================================
1학기 GaitDTW 클래스 확장판.

변경사항:
    - 4발(LF/RF/LH/RH) 각각 기준 파형 비교
    - COP 궤적 유사도 추가
    - 대각선 발 파형 교차 비교 추가
    - 1학기 구조(RMSE 기반 DTW) 그대로 유지

1학기 방식:
    LF 시계열 1개만 정상 템플릿과 비교

2학기 방식:
    LF/RF/LH/RH 4발 각각 비교 후 가중 평균
    + COP 궤적 유사도 추가
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

log = logging.getLogger("DTW_V2")

TEMPLATE_PATH = Path('calib/dtw_templates_v2.json')


# ── DTW 거리 계산 (1학기와 동일 방식) ──────────
def _rmse_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """
    RMSE 기반 거리 계산.
    1학기 GaitDTW 와 동일한 방식 유지.

    두 시계열 길이가 다르면 보간 후 비교.
    """
    if len(s1) != len(s2):
        x1  = np.linspace(0, 1, len(s1))
        x2  = np.linspace(0, 1, len(s2))
        s2  = np.interp(x1, x2, s2)
    return float(np.sqrt(np.mean((s1 - s2) ** 2)))


# ── COP 궤적 DTW ────────────────────────────────
def _cop_trajectory_distance(
        cop_traj: List[Tuple[float, float]],
        template_traj: List[Tuple[float, float]]) -> float:
    """
    COP 궤적 간 거리 계산.

    cop_traj: [(row_mm, col_mm), ...] 현재 궤적
    template_traj: 정상 기준 궤적

    두 궤적을 같은 길이로 맞춘 후 점간 평균 거리 계산.
    """
    if not cop_traj or not template_traj:
        return 0.0

    n = min(len(cop_traj), len(template_traj))
    if n < 2:
        return 0.0

    # 길이 맞추기
    curr = np.array(cop_traj[:n])
    templ = np.array(template_traj[:n])

    dists = np.sqrt(np.sum((curr - templ) ** 2, axis=1))
    return float(dists.mean())


# ── DTW 분류기 v2 ───────────────────────────────
class GaitDTWV2:
    """
    4발 + COP 궤적 기반 DTW 유사도 분석기.

    보정(calibrate) 방법:
        정상 데이터로 발별 기준 파형 생성
        → dtw_templates_v2.json 저장

    점수 구성:
        LF/RF/LH/RH 각 DTW 점수 (70%)
        COP 궤적 유사도 (30%)
    """

    def __init__(self):
        self.calibrated    = False
        self.templates     = {}     # 발별 기준 파형
        self.thresholds    = {}     # 발별 임계값
        self.cop_templates = {}     # 발별 COP 기준 궤적
        self.cop_thresholds= {}     # COP 임계값

        # 저장된 템플릿 로드
        if TEMPLATE_PATH.exists():
            self._load_templates()

    def _load_templates(self):
        with open(TEMPLATE_PATH, 'r') as f:
            data = json.load(f)
        self.templates      = {k: np.array(v)
                               for k, v in data.get('templates', {}).items()}
        self.thresholds     = data.get('thresholds', {})
        self.cop_templates  = data.get('cop_templates', {})
        self.cop_thresholds = data.get('cop_thresholds', {})
        self.calibrated     = bool(self.templates)
        log.info(f"DTW 템플릿 로드: {list(self.templates.keys())}")

    def _save_templates(self):
        TEMPLATE_PATH.parent.mkdir(exist_ok=True)
        data = {
            'templates':      {k: v.tolist()
                               for k, v in self.templates.items()},
            'thresholds':     self.thresholds,
            'cop_templates':  self.cop_templates,
            'cop_thresholds': self.cop_thresholds,
        }
        with open(TEMPLATE_PATH, 'w') as f:
            json.dump(data, f, indent=2)
        log.info(f"DTW 템플릿 저장: {TEMPLATE_PATH}")

    # ── 보정 ────────────────────────────────────
    def calibrate(self, normal_cycles: Dict[str, List],
                  cop_cycles: Optional[Dict[str, List]] = None):
        """
        정상 데이터로 기준 파형 생성.

        Args:
            normal_cycles: {'LF': [[100포인트], ...], 'RF': ..., ...}
            cop_cycles:    {'LF': [[(r,c), ...], ...], ...}
        """
        pos_names = ['LF', 'RF', 'LH', 'RH']

        for pos in pos_names:
            cycles = normal_cycles.get(pos, [])
            if not cycles:
                log.warning(f"DTW: {pos} 정상 데이터 없음")
                continue

            series = [np.array(c) for c in cycles]

            # 기준 파형 = 정상 데이터 평균
            template = np.mean(series, axis=0)
            self.templates[pos] = template

            # 임계값 = 평균 거리 + 2×표준편차
            dists = [_rmse_distance(s, template) for s in series]
            self.thresholds[pos] = float(
                np.mean(dists) + 2 * np.std(dists)
            )
            log.info(f"DTW {pos}: 임계값={self.thresholds[pos]:.4f} "
                     f"(n={len(series)})")

        # COP 궤적 기준 생성
        if cop_cycles:
            for pos in pos_names:
                trajs = cop_cycles.get(pos, [])
                if not trajs:
                    continue
                # 가장 짧은 궤적 길이로 맞춤
                min_len = min(len(t) for t in trajs)
                arr = [t[:min_len] for t in trajs]
                self.cop_templates[pos] = arr[0]   # 첫 번째를 기준으로
                dists = [_cop_trajectory_distance(t, arr[0]) for t in arr]
                self.cop_thresholds[pos] = float(
                    np.mean(dists) + 2 * np.std(dists) + 1e-6
                )
                log.info(f"DTW COP {pos}: 임계값={self.cop_thresholds[pos]:.2f}")

        self.calibrated = bool(self.templates)
        self._save_templates()

    def calibrate_from_csv(self, csv_path: str = 'gait_collected_v2.csv'):
        """CSV 파일에서 정상 데이터 로드 후 보정"""
        import pandas as pd, ast

        df = pd.read_csv(csv_path)
        normal = df[df['label'] == 0]

        if len(normal) < 2:
            log.warning("정상 데이터 부족 — DTW 보정 실패")
            return

        normal_cycles = {}
        for pos in ['LF', 'RF', 'LH', 'RH']:
            col = f'{pos}_time_series'
            if col in normal.columns:
                normal_cycles[pos] = [
                    ast.literal_eval(r) for r in normal[col]
                ]

        self.calibrate(normal_cycles)
        log.info(f"DTW v2 보정 완료 (정상 {len(normal)}주기)")

    # ── 단일 발 DTW 점수 ─────────────────────────
    def _paw_score(self, pos: str, series: np.ndarray) -> float:
        """단일 발 DTW 비정상 점수"""
        if pos not in self.templates:
            return 0.5   # 템플릿 없으면 중간값
        template  = self.templates[pos]
        threshold = self.thresholds.get(pos, 1.0)
        dist = _rmse_distance(series, template)
        return float(np.clip(dist / (threshold + 1e-9), 0, 1))

    # ── COP DTW 점수 ─────────────────────────────
    def _cop_score(self, pos: str,
                   cop_traj: List[Tuple[float, float]]) -> float:
        """COP 궤적 DTW 비정상 점수"""
        if pos not in self.cop_templates or not cop_traj:
            return 0.5
        template  = self.cop_templates[pos]
        threshold = self.cop_thresholds.get(pos, 1.0)
        dist = _cop_trajectory_distance(cop_traj, template)
        return float(np.clip(dist / (threshold + 1e-9), 0, 1))

    # ── 메인 점수 계산 ────────────────────────────
    def score(self,
              lf_series: List[float], rf_series: List[float],
              lh_series: Optional[List[float]] = None,
              rh_series: Optional[List[float]] = None,
              cop_trajs: Optional[Dict] = None) -> Tuple[float, Dict]:
        """
        전체 DTW 비정상 점수 계산.

        Args:
            lf_series: 왼앞발 시계열 (100포인트)
            rf_series: 오른앞발 시계열
            lh_series: 왼뒷발 시계열 (선택)
            rh_series: 오른뒷발 시계열 (선택)
            cop_trajs: {'LF': [(r,c),...], ...} COP 궤적 (선택)

        Returns:
            (dtw_score, detail_dict)
        """
        if not self.calibrated:
            log.warning("DTW 미보정 — 0.5 반환")
            return 0.5, {'dtw_score': 0.5, 'mode': 'uncalibrated'}

        lf = np.array(lf_series, dtype=float)
        rf = np.array(rf_series, dtype=float)

        has_4paw = lh_series is not None and rh_series is not None
        has_cop  = cop_trajs is not None

        # ── 발별 DTW 점수 ──────────────────────────
        paw_scores = {
            'LF': self._paw_score('LF', lf),
            'RF': self._paw_score('RF', rf),
        }
        if has_4paw:
            lh = np.array(lh_series, dtype=float)
            rh = np.array(rh_series, dtype=float)
            paw_scores['LH'] = self._paw_score('LH', lh)
            paw_scores['RH'] = self._paw_score('RH', rh)

        paw_mean = float(np.mean(list(paw_scores.values())))

        # ── COP 궤적 DTW 점수 ──────────────────────
        cop_scores = {}
        if has_cop:
            for pos in ['LF', 'RF', 'LH', 'RH']:
                traj = cop_trajs.get(pos, [])
                if traj:
                    cop_scores[pos] = self._cop_score(pos, traj)
            cop_mean = float(np.mean(list(cop_scores.values()))) \
                       if cop_scores else paw_mean
        else:
            cop_mean = paw_mean

        # ── 최종 가중 합산 ──────────────────────────
        # 발 파형 70% + COP 궤적 30%
        if has_cop and cop_scores:
            final = 0.70 * paw_mean + 0.30 * cop_mean
        else:
            final = paw_mean

        final = float(np.clip(final, 0, 1))

        detail = {
            'dtw_score':   round(final, 4),
            'paw_scores':  {k: round(v, 4) for k, v in paw_scores.items()},
            'paw_mean':    round(paw_mean, 4),
            'cop_scores':  {k: round(v, 4) for k, v in cop_scores.items()},
            'cop_mean':    round(cop_mean, 4),
            'mode':        '4paw+cop' if (has_4paw and has_cop)
                           else '4paw' if has_4paw
                           else '2paw',
        }
        return final, detail


# ── 단독 실행 테스트 ───────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    dtw = GaitDTWV2()

    # 1학기 CSV 로 보정 테스트
    import os
    if os.path.exists('gait_collected_data.csv'):
        import pandas as pd, ast
        df = pd.read_csv('gait_collected_data.csv')
        normal = df[df['label'] == 0]
        normal_cycles = {
            'LF': [ast.literal_eval(r) for r in normal['LF_time_series']],
            'RF': [ast.literal_eval(r) for r in normal['RF_time_series']],
        }
        dtw.calibrate(normal_cycles)
        print("✅ 1학기 CSV 로 보정 완료")

    # 더미 데이터로 점수 테스트
    np.random.seed(42)
    lf = list(np.random.uniform(0, 30, 100))
    rf = list(np.random.uniform(0, 30, 100))
    lh = list(np.random.uniform(0, 30, 100))
    rh = list(np.random.uniform(0, 30, 100))
    cop = {'LF': [(i*0.5, i*0.3) for i in range(20)]}

    score_2p, d2 = dtw.score(lf, rf)
    score_4p, d4 = dtw.score(lf, rf, lh, rh)
    score_cop, dc = dtw.score(lf, rf, lh, rh, cop)

    print(f"\n[2발] DTW 점수: {score_2p:.4f}")
    print(f"[4발] DTW 점수: {score_4p:.4f}")
    print(f"[COP] DTW 점수: {score_cop:.4f}")
    for k, v in dc.items(): print(f"  {k}: {v}")
