# -*- coding: utf-8 -*-
"""
gait_ensemble_v2.py — ML + HMM + DTW 앙상블 v2 [2발 모드: LF/RF]
==================================================
1학기 gait_ensemble.py 확장판.

변경사항:
    - LF/RF 2발 입력 (LH/RH 는 이번 실험에서 제외)
    - COP 궤적 특징 추가
    - IMU 지형 보정 적용
    - HMM LOG_MAX 실측값 기반 수정
    - WebSocket 서버 통합
    - 앱으로 결과 실시간 전송

1학기 구조 유지:
    ML(0.4) + HMM(0.3) + DTW(0.3) 가중 앙상블
    최종 점수 > 0.5 → 비정상
    연속 2회 → 최종 비정상 확정

사용법:
    python gait_ensemble_v2.py          # 실시간 측정
    python gait_ensemble_v2.py --dummy  # 더미 모드 (보드 없이 테스트)
"""

import asyncio
import json
import logging
import time
import numpy as np
import websockets
from typing import Dict, Optional

from frame_sync import SyncedFrame, FrameSynchronizer
from ble_receiver import PawReceiver, DummyPawReceiver
from gait_realtime_v2 import GaitPreprocessorV2
from cop_analyzer import CopAnalyzer
from gait_ml_v2 import MLClassifierV2
from gait_hmm_v2 import HMMScorerV2
from gait_dtw_v2 import GaitDTWV2
from gait_sym_scorer import SymScorers

log = logging.getLogger("ENSEMBLE_V2")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# ══════════════════════════════════════════════
# 설정값
# ══════════════════════════════════════════════
ML_WEIGHT  = 0.4
HMM_WEIGHT = 0.3
DTW_WEIGHT = 0.3

CONSECUTIVE_THRESHOLD = 2    # 연속 N회 비정상 → 최종 확정
ROBOT_WEIGHT_KG       = 15.0

# 앙상블 점수 → 대칭성 % 변환 기준
NORMAL_MIN = 80   # 80% 이상 → 정상
WARN_MIN   = 60   # 60~80%  → 경고

# WebSocket 서버 설정
WS_HOST = "0.0.0.0"
WS_PORT = 8765

# 더미 모드 (보드 없이 테스트)
USE_DUMMY = False


# ══════════════════════════════════════════════
# 상태 메시지 생성
# ══════════════════════════════════════════════
def make_status_message(symmetry: int, paw_scores: Dict,
                         prev_symmetry: Optional[int] = None) -> str:
    """보호자용 한 줄 상태 메시지"""
    # 악화된 발 찾기
    bad_paws = [f"{k}발" for k, v in paw_scores.items() if v < WARN_MIN]
    paw_map  = {'LF': '왼쪽 앞', 'RF': '오른쪽 앞'}   # [2발 모드]
    bad_names = [paw_map.get(p.replace('발',''), p) for p in bad_paws]

    # 변화 계산
    change = ''
    if prev_symmetry is not None:
        diff = symmetry - prev_symmetry
        if diff > 3:
            change = f" (어제보다 {diff}점 올랐어요 📈)"
        elif diff < -3:
            change = f" (어제보다 {abs(diff)}점 내려갔어요 📉)"

    if symmetry >= NORMAL_MIN:
        return f"지금 잘 걷고 있어요 👍{change}"
    elif symmetry >= WARN_MIN:
        if bad_names:
            return f"{bad_names[0]}발을 살짝 아끼고 있어요{change}"
        return f"보행이 약간 불균형해요. 지켜봐 주세요 🟡{change}"
    else:
        if bad_names:
            return f"{bad_names[0]}발을 불편해하고 있어요. 확인해보세요 🔴"
        return "걸음이 불안정해요. 수의사 상담을 권장해요 🏥"


# ══════════════════════════════════════════════
# 앙상블 판별기 v2
# ══════════════════════════════════════════════
class EnsembleV2:
    """
    LF/RF 2발 + IMU + COP 기반 앙상블 보행 판별기. [2발 모드]

    1학기 구조 그대로:
        ML(0.4) + HMM(0.3) + DTW(0.3)
        최종 점수 > 0.5 → 비정상
        연속 2회 → 최종 비정상 확정

    2학기 확장:
        COP 궤적 점수 추가
        IMU 지형 보정 적용
        WebSocket 실시간 전송
    """

    def __init__(self):
        log.info("앙상블 v2 초기화 중...")

        # 알고리즘 초기화
        self.ml    = MLClassifierV2()
        self.hmm   = HMMScorerV2()
        self.dtw   = GaitDTWV2()
        # 좌우 대칭 정상 기준 DTW/HMM (calib/sym_scorers_v2.pkl 이 있으면 기존 DTW/HMM 대신 사용)
        self.sym   = SymScorers.load()
        if self.sym:
            log.info("대칭 정상 기준 DTW/HMM 사용 (calib/sym_scorers_v2.pkl)")

        # 전처리기
        self.preprocessor = GaitPreprocessorV2(ROBOT_WEIGHT_KG)
        self.cop_analyzer  = CopAnalyzer()

        # 상태 관리
        self._consecutive  = 0
        self._forced_ab    = False
        self._cycle_count  = 0
        self._results      = []
        self._prev_symmetry: Optional[int] = None

        # COP 버퍼
        self._cop_buffer   = []

        # WebSocket 클라이언트
        self._ws_clients   = set()

        # 1학기 CSV 로 DTW 보정
        self._init_dtw()

        log.info("앙상블 v2 초기화 완료!")

    def _init_dtw(self):
        """1학기 CSV 또는 v2 CSV 로 DTW 보정"""
        import os
        if os.path.exists('gait_collected_v2.csv'):
            self.dtw.calibrate_from_csv('gait_collected_v2.csv')
        elif os.path.exists('gait_collected_data.csv'):
            # 1학기 CSV 로 폴백
            import pandas as pd, ast
            df = pd.read_csv('gait_collected_data.csv')
            normal = df[df['label'] == 0]
            cycles = {
                'LF': [ast.literal_eval(r) for r in normal['LF_time_series']],
                'RF': [ast.literal_eval(r) for r in normal['RF_time_series']],
            }
            self.dtw.calibrate(cycles)
            log.info("DTW: 1학기 CSV 로 보정 (2발 모드)")
        else:
            log.warning("DTW: CSV 없음 — 미보정 상태")

    # ── 앙상블 점수 계산 ──────────────────────────
    def predict(self, features: Dict,
                cop_trajs: Optional[Dict] = None) -> Dict:
        """
        특징 딕셔너리로 앙상블 점수 계산.

        Args:
            features:  gait_realtime_v2.py 출력
            cop_trajs: cop_analyzer.py 출력 COP 궤적

        Returns:
            결과 딕셔너리
        """
        # 시계열 추출 [2발 모드] — LH/RH 는 없으므로 HMM/DTW 는 자동으로 2발 모드로 채점됨
        lf_ts = features.get('LF_time_series', [0]*100)
        rf_ts = features.get('RF_time_series', [0]*100)

        # ── ML 점수 ──────────────────────────────
        ml_s = self.ml.predict(features)

        if self.sym:
            # ── DTW/HMM 점수: 좌우 대칭 정상 기준 (실제 시간축 파형 + RF 게인 보정) ──
            dtw_s, hmm_s = self.sym.score(features)
            hmm_detail, dtw_detail = {}, {}
        else:
            # ── HMM 점수 ─────────────────────────────
            hmm_s, hmm_detail = self.hmm.score(lf_ts, rf_ts)

            # ── DTW 점수 ─────────────────────────────
            dtw_s, dtw_detail = self.dtw.score(lf_ts, rf_ts, cop_trajs=cop_trajs)

        # ── 가중 앙상블 ──────────────────────────
        final = (ML_WEIGHT * ml_s +
                 HMM_WEIGHT * hmm_s +
                 DTW_WEIGHT * dtw_s)
        final   = float(np.clip(final, 0, 1))
        verdict = 'ABNORMAL' if final > 0.5 else 'NORMAL'

        # 연속 비정상 카운터
        if verdict == 'ABNORMAL':
            self._consecutive += 1
        else:
            self._consecutive = 0

        if self._consecutive >= CONSECUTIVE_THRESHOLD:
            verdict          = 'ABNORMAL'
            self._forced_ab  = True

        self._cycle_count += 1

        # 대칭성 % 변환 (앙상블 점수 뒤집기)
        symmetry = int(round((1 - final) * 100))

        # 발별 대칭성 점수 [2발 모드]
        paw_scores = {
            'LF': int(round((1 - ml_s) * 100)),
            'RF': int(round((1 - ml_s) * 100)),
        }

        # 상태 메시지
        message = make_status_message(
            symmetry, paw_scores, self._prev_symmetry
        )
        self._prev_symmetry = symmetry

        result = {
            # 핵심 결과
            'verdict':     verdict,
            'score':       round(final, 4),
            'symmetry':    symmetry,
            'message':     message,
            # 발별 상태
            'paw':         paw_scores,
            # 알고리즘 상세
            'ml_score':    round(ml_s, 4),
            'hmm_score':   round(hmm_s, 4),
            'dtw_score':   round(dtw_s, 4),
            # 특징 주요값
            'si_grf':      round(features.get('SI_GRF_Peak(%)', 0), 2),
            'imu_pitch':   round(features.get('IMU_Pitch(deg)', 0), 2),
            # 연속 비정상
            'consecutive': self._consecutive,
            'forced_ab':   self._forced_ab,
            'cycle':       self._cycle_count,
            # 배터리 (BLE 수신 시 채워짐) [2발 모드]
            'battery':     {'LF': 100, 'RF': 100},
            # 타임스탬프
            'timestamp':   time.time(),
        }

        self._results.append(result)
        self._print_result(result, hmm_detail, dtw_detail)
        return result

    def _print_result(self, result: Dict,
                      hmm_detail: Dict, dtw_detail: Dict):
        """콘솔 출력.

        [2026-09-28 수정] Windows 콘솔(cp949)에서 이모지 출력이
        UnicodeEncodeError 를 내면, 그냥 로그 한 줄이 안 찍히는 게 아니라
        이 함수를 호출한 predict() 전체가 예외로 죽어서 이미 계산된 결과값까지
        통째로 사라지는 문제가 있었다(서버로 감쌌을 때 API 요청이 500으로
        실패함). 콘솔 출력 실패가 실제 판정 결과 반환을 막으면 안 되므로
        통째로 try/except 로 감싸 무시한다.
        """
        try:
            icon = '🟢' if result['verdict'] == 'NORMAL' else '🔴'
            print(f"\n  [{result['cycle']:>3}주기] {icon} {result['verdict']} "
                  f"앙상블={result['score']:.3f} "
                  f"대칭성={result['symmetry']}%")
            print(f"    ML={result['ml_score']:.3f} "
                  f"HMM={result['hmm_score']:.3f} "
                  f"DTW={result['dtw_score']:.3f}")
            print(f"    SI={result['si_grf']:.1f}%")
            print(f"    💬 {result['message']}")
            if result['forced_ab']:
                print(f"    ⚠️  연속 {result['consecutive']}회 비정상 → 최종 확정!")
        except Exception as e:
            log.debug(f"_print_result 콘솔 출력 실패(무시): {e}")

    # ── SyncedFrame 처리 ─────────────────────────
    def on_synced(self, synced: SyncedFrame):
        """frame_sync.py 콜백"""
        self._cop_buffer.append(synced)

        # 전처리
        features = self.preprocessor.process_frame(synced)
        if features is None:
            return

        # COP 분석
        cop_features = self.cop_analyzer.analyze(self._cop_buffer)
        self._cop_buffer.clear()
        features.update(cop_features)

        # COP 궤적 추출 (DTW용)
        cop_trajs = None   # 실측 데이터 수집 후 연결

        # 앙상블 판별
        result = self.predict(features, cop_trajs)

        # WebSocket 전송
        asyncio.create_task(self._broadcast(result))

    # ── WebSocket 서버 ────────────────────────────
    async def ws_handler(self, websocket):
        """앱 연결 처리"""
        self._ws_clients.add(websocket)
        log.info(f"앱 연결됨: {websocket.remote_address}")
        try:
            async for _ in websocket:
                pass   # 앱에서 오는 메시지는 무시
        finally:
            self._ws_clients.discard(websocket)
            log.info(f"앱 연결 끊김: {websocket.remote_address}")

    async def _broadcast(self, result: Dict):
        """모든 연결된 앱에 결과 전송"""
        if not self._ws_clients:
            return
        msg = json.dumps(result, ensure_ascii=False)
        disconnected = set()
        for ws in self._ws_clients:
            try:
                await ws.send(msg)
            except Exception:
                disconnected.add(ws)
        self._ws_clients -= disconnected

    # ── 최종 요약 ────────────────────────────────
    def summary(self):
        """전체 세션 요약 출력"""
        if not self._results:
            return
        total  = len(self._results)
        ab_cnt = sum(1 for r in self._results if r['verdict'] == 'ABNORMAL')
        no_cnt = total - ab_cnt
        avg_sym = np.mean([r['symmetry'] for r in self._results])

        print(f"\n{'='*60}")
        print(f"  세션 요약: {total}주기")
        print(f"  정상: {no_cnt}회 / 비정상: {ab_cnt}회")
        print(f"  평균 대칭성: {avg_sym:.1f}%")

        if self._forced_ab:
            verdict = '🔴 비정상 (연속 비정상 감지)'
        elif ab_cnt > no_cnt:
            verdict = '🔴 비정상'
        else:
            verdict = '🟢 정상'
        print(f"  최종 판정: {verdict}")
        print(f"{'='*60}")


# ══════════════════════════════════════════════
# 메인 실행
# ══════════════════════════════════════════════
async def main(use_dummy: bool = False):
    ensemble = EnsembleV2()
    sync     = FrameSynchronizer(on_synced=ensemble.on_synced)

    # WebSocket 서버 시작
    ws_server = await websockets.serve(
        ensemble.ws_handler, WS_HOST, WS_PORT
    )
    log.info(f"WebSocket 서버 시작: ws://{WS_HOST}:{WS_PORT}")
    log.info(f"앱에서 ws://[PC_IP]:{WS_PORT} 로 연결하세요")

    # BLE 수신기
    if use_dummy:
        log.info("더미 모드 (보드 없이 테스트)")
        receiver = DummyPawReceiver(on_frame=sync.on_frame)
        recv_task = asyncio.create_task(receiver.run(hz=50))
    else:
        receiver = PawReceiver(on_frame=sync.on_frame)
        recv_task = asyncio.create_task(receiver.run())

    log.info("=== 측정 시작 (Ctrl+C 로 종료) ===")
    try:
        await asyncio.Event().wait()   # 영구 대기
    except asyncio.CancelledError:
        pass
    finally:
        recv_task.cancel()
        ws_server.close()
        ensemble.summary()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dummy', action='store_true',
                        help='더미 모드 (보드 없이 테스트)')
    args = parser.parse_args()

    asyncio.run(main(use_dummy=args.dummy or USE_DUMMY))
