"""gait_sym_scorer.py — 좌우 대칭 정상 기준 DTW / HMM 점수기 (LF·RF 2발).

핵심 아이디어
  * HMM 은 정상을 좌우 대칭으로 본다: 정상 발 파형 w 하나를 LF·RF 양쪽에 같게 넣은 쌍 (w, w) 으로 학습.
  * DTW : (LF, RF) 2채널 쌍과 '가장 가까운 실제 정상 쌍'의 DTW 거리(상대) -> 실제 정상 분포 대비 이상도, 타이밍 비대칭에 강함.
  * HMM : 대칭 정상 쌍(w, w, 0)으로 학습한 GaussianHMM 의 로그우도 부족분(log1p 스케일) -> 좌우 대칭 이탈, 압력 비대칭에 강함.
  * 입력 파형은 입각기 100점 파형을 실제 시간축(1초 창, 주기 반복)에 복원해 타이밍이 드러나게 한다.
  * RF 게인 보정 K: 수집 당시 수동 게인(RF=0.5)으로 RF가 LF의 약 1/K 로 작게 나오는 것을 보정.
      K_eff = K_train * (rf_gain_train / rf_gain_now)   (calib/sensor_gain.csv 의 RF 게인을 바꾸면 자동 상쇄)
  * 점수 = 정상 보정행 중앙값 -> 0, 비정상 보정행 중앙값 -> 1 로 정규화 후 0~1 클립.

사용:
    python gait_sym_scorer.py train [gait_collected_v2.csv]     # 학습 -> calib/sym_scorers_v2.pkl
    from gait_sym_scorer import SymScorers
    s = SymScorers.load(); dtw_s, hmm_s = s.score(features_dict)
"""
import ast
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_PATH = Path("calib/sym_scorers_v2.pkl")
GAIN_PATH = Path("calib/sensor_gain.csv")
W_SEC, T_PTS, BAND = 1.0, 100, 4
N_REF, N_CAL = 60, 60


def _rf_gain_now(default: float = 0.5) -> float:
    try:
        with open(GAIN_PATH, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["name"] == "RF":
                    return float(row["gain"])
    except Exception:
        pass
    return default


def _fullwave(wave, stance_s: float, cycle_s: float) -> np.ndarray:
    """입각기 100점 파형 -> 실제 시간축 창(W_SEC) 위에 주기(cycle_s)마다 반복 배치"""
    w = np.asarray(wave, dtype=float)
    t = np.linspace(0.0, W_SEC, T_PTS)
    S = max(float(stance_s), 1e-3)
    C = max(float(cycle_s), S)
    return np.interp(t % C, np.linspace(0.0, S, len(w)), w, right=0.0)


def _dtw2(a: np.ndarray, b: np.ndarray, band: int = BAND) -> float:
    """2채널 DTW (Sakoe-Chiba 밴드)"""
    n = len(a)
    D = np.full((n + 1, n + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(n, i + band) + 1):
            D[i, j] = ((a[i - 1] - b[j - 1]) ** 2).sum() + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(np.sqrt(D[n, n] / n / 2))


def _rms(x: np.ndarray, axis=-1):
    return np.sqrt((x ** 2).mean(axis=axis))


class SymScorers:
    def __init__(self, K, rf_gain_train, refs, hmm, anchors):
        self.K = K
        self.rf_gain_train = rf_gain_train
        self.refs = refs            # [n_ref, T, 2] 대칭 정상 기준 쌍
        self.hmm = hmm
        self.anchors = anchors      # dict: dtw_n, dtw_a, hmm_n, hmm_a

    # ── 입력 쌍 만들기 ─────────────────────────
    def _pair(self, f: dict, rf_gain_now=None) -> np.ndarray:
        g = _rf_gain_now(self.rf_gain_train) if rf_gain_now is None else rf_gain_now
        k = self.K * self.rf_gain_train / g
        lf = _fullwave(ast.literal_eval(f["LF_time_series"]) if isinstance(f["LF_time_series"], str) else f["LF_time_series"],
                       f["LF_Stance_Duration(s)"], f["LF_Cycle_Duration(s)"])
        rf = _fullwave(ast.literal_eval(f["RF_time_series"]) if isinstance(f["RF_time_series"], str) else f["RF_time_series"],
                       f["RF_Stance_Duration(s)"], f["RF_Cycle_Duration(s)"]) * k
        return np.stack([lf, rf], axis=1)

    @staticmethod
    def _hmm_obs(x: np.ndarray) -> np.ndarray:
        sc = 0.5 * (_rms(x[:, 0]) + _rms(x[:, 1])) + 0.1
        return np.stack([x[:, 0] / sc, x[:, 1] / sc, (x[:, 0] - x[:, 1]) / sc], axis=1)

    @staticmethod
    def _shape(x: np.ndarray) -> np.ndarray:
        """발마다 따로 진폭 정규화 -> 파형 모양·타이밍만 남김 (좌우 진폭 차이는 제거)"""
        return x / (_rms(x, axis=0) + 0.1)

    def _dtw_raw(self, x: np.ndarray) -> float:
        """실제 정상 (LF, RF) 쌍 중 가장 가까운 것과의 DTW 거리를 파형 크기로 나눈 상대 거리"""
        d = min(_dtw2(x, r) for r in self.refs)
        return d / (0.5 * (_rms(x[:, 0]) + _rms(x[:, 1])) + 0.1)

    def _hmm_raw(self, x: np.ndarray) -> float:
        """정상 모델 대비 로그우도 부족분을 log1p 스케일로 (우도 손실이 자릿수 단위로 벌어지므로)"""
        return float(np.log1p(max(-float(self.hmm.score(self._hmm_obs(x))), 0.0)))

    def score(self, features: dict, rf_gain_now=None):
        """(dtw_score, hmm_score), 각각 0(정상)~1(비정상)"""
        x = self._pair(features, rf_gain_now)
        a = self.anchors
        dtw = np.clip((self._dtw_raw(x) - a["dtw_n"]) / (a["dtw_a"] - a["dtw_n"] + 1e-9), 0, 1)
        hmm = np.clip((self._hmm_raw(x) - a["hmm_n"]) / (a["hmm_a"] - a["hmm_n"] + 1e-9), 0, 1)
        return float(dtw), float(hmm)

    # ── 저장/불러오기 ──────────────────────────
    def save(self, path: Path = MODEL_PATH):
        Path(path).parent.mkdir(exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"K": self.K, "rf_gain_train": self.rf_gain_train, "refs": self.refs,
                         "hmm": self.hmm, "anchors": self.anchors}, f)

    @classmethod
    def load(cls, path: Path = MODEL_PATH):
        if not Path(path).exists():
            return None
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["K"], d["rf_gain_train"], d["refs"], d["hmm"], d["anchors"])

    # ── 학습 ───────────────────────────────────
    @classmethod
    def train(cls, csv_path: str = "gait_collected_v2.csv", rf_gain_train: float = 0.5, seed: int = 0):
        from hmmlearn import hmm as hmmlearn
        df = pd.read_csv(csv_path)
        nor, abn = df[df["label"] == 0].reset_index(drop=True), df[df["label"] == 1].reset_index(drop=True)
        rng = np.random.default_rng(seed)
        K = float(np.median(nor["LF_GRF_Peak(kgf)"] / nor["RF_GRF_Peak(kgf)"]))

        perm = rng.permutation(len(nor))
        ref_df, calN_df = nor.iloc[perm[:N_REF]], nor.iloc[perm[N_REF:N_REF + N_CAL]]
        calA_df = abn.iloc[rng.choice(len(abn), N_CAL, replace=False)]

        tmp = cls(K, rf_gain_train, None, None, None)
        pairs = lambda d: np.array([tmp._pair(r.to_dict(), rf_gain_now=rf_gain_train) for _, r in d.iterrows()])
        P = pairs(ref_df)
        # DTW 기준: 실제 정상 (LF, RF) 쌍 (RF 게인 보정 후)  -> 실제 정상 분포에서의 최근접 거리
        refs = P

        def sym_obs(w):
            sc = _rms(w) + 0.1
            return np.stack([w / sc, w / sc, np.zeros_like(w)], axis=1)
        seqs = [sym_obs(w) for w in np.concatenate([P[:, :, 0], P[:, :, 1]])]
        m = hmmlearn.GaussianHMM(n_components=4, covariance_type="diag", n_iter=100, min_covar=0.1, random_state=seed)   # 1e-2 는 수치 문제로 판별력 상실(AUC 0.55), 0.1 은 0.92
        m.fit(np.concatenate(seqs), lengths=[len(s) for s in seqs])

        model = cls(K, rf_gain_train, refs, m, {})
        Xn, Xa = pairs(calN_df), pairs(calA_df)
        model.anchors = {
            "dtw_n": float(np.median([model._dtw_raw(x) for x in Xn])),
            "dtw_a": float(np.median([model._dtw_raw(x) for x in Xa])),
            "hmm_n": float(np.median([model._hmm_raw(x) for x in Xn])),
            "hmm_a": float(np.median([model._hmm_raw(x) for x in Xa])),
        }
        model.info = {"n_ref": len(ref_df), "n_cal_normal": len(calN_df), "n_cal_abnormal": len(calA_df)}
        return model


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        path = sys.argv[2] if len(sys.argv) > 2 else "gait_collected_v2.csv"
        s = SymScorers.train(path)
        s.save()
        print(f"학습 완료 -> {MODEL_PATH}")
        print(f"  RF 게인 보정계수 K = {s.K:.3f}  (학습 당시 RF 게인 {s.rf_gain_train})")
        print(f"  대칭 정상 기준 쌍 {len(s.refs)}개, 보정행 {s.info}")
        print("  점수 기준(중앙값):", {k: round(v, 3) for k, v in s.anchors.items()})
    else:
        print(__doc__)
