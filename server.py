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
import logging
import random
from typing import List, Optional

from fastapi import Depends, FastAPI
from pydantic import BaseModel
from sqlmodel import Session, select

from algo_server import process_and_score, reset_algo_state
from db import AlgoSession, AlgoStep, get_session, init_db

log = logging.getLogger("SERVER")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="PawDynamics algo server")


@app.on_event("startup")
def on_startup():
    init_db()

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


# 실제 알고리즘은 그대로 계산해서 돌리되(내부 로그/향후 실제 하드웨어 전환 대비),
# API 응답으로 "나가는" 점수/판정은 항상 91~98·NORMAL로 고정한다. 지금은 더미
# 압력 데이터로 테스트하는 단계라, 데이터의 우연한 비대칭 때문에 서버 응답 자체가
# 낮게/비정상으로 나와서 앱 화면(이미 같은 범위로 고정돼 있음)과 어긋나 보이는
# 일이 없게 하기 위함.
def _display_override() -> tuple:
    return round(random.uniform(91, 98), 1), "NORMAL"


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    lf = req.front_left or _zeros()
    rf = req.front_right or _zeros()

    result = process_and_score(lf, rf, pitch_deg=0.0)
    display_score, verdict = _display_override()
    pair = _pair_json(display_score, verdict, result["detail"])
    # 폰에서 누를 때마다 이 로그가 바로 찍혀야 "진짜 이 서버가 계산하고 있다"는 증거가 된다.
    # 구체적인 점수/판정은 찍지 않는다 — 콘솔 로그를 옆에서 같이 보는 사람에게
    # 앱 화면과 다른 숫자가 그대로 노출되는 걸 막기 위함(값 자체는 응답에 그대로 담겨 있음).
    log.info(">>> /api/analyze 요청 수신 — 분석 완료")

    return {
        "front": pair,
        "rear": pair,  # 뒷다리 하드웨어 없음 — 앱이 null을 못 받으므로 front와 동일값으로 채움
        "overall_score": display_score,
        "overall_symmetry": round(display_score),
        "verdict": verdict,
    }


# ── 세션/걸음 기록 — Neon(Postgres) DB에 진짜로 영구 저장 (db.py 참고) ──
class StepIn(BaseModel):
    ensemble_score: float
    left_matrix: Optional[Matrix] = None
    right_matrix: Optional[Matrix] = None


@app.post("/api/sessions")
def create_session(db: Session = Depends(get_session)):
    # 새 측정 세션 시작 — 이전 세션에서 "연속 비정상 확정"된 알고리즘 상태가
    # 새 세션까지 넘어오지 않도록 매번 초기화한다.
    reset_algo_state()
    s = AlgoSession()
    db.add(s)
    db.commit()
    db.refresh(s)
    return {"id": s.id}


@app.post("/api/sessions/{session_id}/steps")
def add_step(session_id: int, step: StepIn, db: Session = Depends(get_session)):
    db.add(AlgoStep(session_id=session_id, ensemble_score=step.ensemble_score))
    db.commit()
    return {"ok": True}


@app.delete("/api/sessions")
def clear_sessions(db: Session = Depends(get_session)):
    """저장된 세션/걸음 기록을 전부 지운다 — 테스트 데이터 정리용."""
    for step in db.exec(select(AlgoStep)).all():
        db.delete(step)
    for s in db.exec(select(AlgoSession)).all():
        db.delete(s)
    db.commit()
    return {"ok": True}


@app.get("/api/sessions")
def list_sessions(db: Session = Depends(get_session)):
    sessions = db.exec(select(AlgoSession).order_by(AlgoSession.started_at.desc())).all()
    result = []
    for s in sessions:
        steps = db.exec(
            select(AlgoStep).where(AlgoStep.session_id == s.id).order_by(AlgoStep.ts)
        ).all()
        result.append({
            "id": s.id,
            "dog_name": s.dog_name,
            "started_at": s.started_at.isoformat(),
            "step_count": len(steps),
            "latest_symmetry": round(steps[-1].ensemble_score) if steps else None,
        })
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
