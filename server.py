# -*- coding: utf-8 -*-
"""above_stride_app(Flutter)이 그대로 호출할 수 있게, algo_server.process_and_score()를
POST /api/analyze 로 감싼 FastAPI 서버.

앱(lib/services/api_service.dart)이 실제로 부르는 엔드포인트만 최소한으로 구현:
  - GET  /api/sessions            : 서버 살아있는지 확인(ping)용 + 세션 목록
  - POST /api/sessions            : 새 측정 세션 시작 (id 발급)
  - POST /api/sessions/{id}/steps : 걸음 기록 (실패해도 앱이 무시하므로 단순 저장만)
  - POST /api/analyze             : 핵심 — process_and_score() 호출해서 결과 반환

지금 실제 하드웨어는 앞다리(F.L/F.R)뿐이라 front_left/front_right만 쓰고,
rear_*는 있어도 무시한다(하드웨어 없음). 앱의 AnalyzeResult.fromJson은 "rear"
필드가 null이 아니어야 해서(레거시), front와 동일한 값을 그대로 채워 보낸다.
"""
import itertools
import logging
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from algo_server import process_and_score

log = logging.getLogger("SERVER")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="PawDynamics algo server")

Matrix = List[List[float]]


class AnalyzeRequest(BaseModel):
    front_left: Optional[Matrix] = None
    front_right: Optional[Matrix] = None
    rear_left: Optional[Matrix] = None
    rear_right: Optional[Matrix] = None


def _zeros() -> Matrix:
    return [[0.0] * 10 for _ in range(16)]


def _pair_json(score: float, verdict: str, detail: dict) -> dict:
    return {
        "ensemble_score": score,
        "verdict": verdict,
        "ml_score": detail.get("ml_score", 0.0),
        "dtw_score": detail.get("dtw_score", 0.0),
    }


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    lf = req.front_left or _zeros()
    rf = req.front_right or _zeros()

    result = process_and_score(lf, rf, pitch_deg=0.0)
    verdict = result["status"].upper()  # "normal" -> "NORMAL" 등 (앱이 'ABNORMAL' 대문자 비교함)
    pair = _pair_json(result["score"], verdict, result["detail"])
    # 폰에서 누를 때마다 이 로그가 바로 찍혀야 "진짜 이 서버가 계산하고 있다"는 증거가 된다.
    log.info(f">>> /api/analyze 요청 수신 — score={result['score']:.1f} status={verdict} "
             f"asymmetry={result['asymmetry']:.3f}")

    return {
        "front": pair,
        "rear": pair,  # 뒷다리 하드웨어 없음 — 앱이 null을 못 받으므로 front와 동일값으로 채움
        "overall_score": result["score"],
        "overall_symmetry": round(result["score"]),
        "verdict": verdict,
    }


# ── 세션/걸음 기록 — 메모리에만 저장 (재시작하면 날아감, 지금은 이거로 충분) ──
_sessions: dict[int, dict] = {}
_session_id_seq = itertools.count(1)


class StepIn(BaseModel):
    ensemble_score: float
    left_matrix: Optional[Matrix] = None
    right_matrix: Optional[Matrix] = None


@app.post("/api/sessions")
def create_session():
    sid = next(_session_id_seq)
    _sessions[sid] = {
        "id": sid,
        "dog_name": "코코",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "step_count": 0,
        "latest_symmetry": None,
    }
    return {"id": sid}


@app.post("/api/sessions/{session_id}/steps")
def add_step(session_id: int, step: StepIn):
    s = _sessions.get(session_id)
    if s is not None:
        s["step_count"] += 1
        s["latest_symmetry"] = round(step.ensemble_score)
    return {"ok": True}


@app.get("/api/sessions")
def list_sessions():
    return list(_sessions.values())[::-1]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
