# -*- coding: utf-8 -*-
"""세션/걸음 기록을 진짜로 영구 저장하기 위한 DB 설정.

기존 with-stride-app이 이미 쓰고 있는 Neon(Postgres) DB를 그대로 재사용한다 —
DATABASE_URL 환경변수만 이 서버(Render)에도 똑같이 넣어주면 됨. 단, 테이블 이름은
with-stride-app 쪽 테이블(gaitsession/gaitstep, 컬럼 구조가 다름)과 안 겹치게
algo_session/algo_step으로 따로 둬서 서로 안전하게 공존한다.

DATABASE_URL이 없으면(로컬 개발) 파일 기반 SQLite로 자동 대체된다.
"""
import os
from datetime import datetime, timezone
from typing import Optional

from sqlmodel import SQLModel, Field, create_engine, Session

_raw_url = os.environ.get("DATABASE_URL")
if _raw_url:
    # Render/Neon이 주는 postgres:// 스킴을 SQLAlchemy가 이해하는 postgresql://로 보정
    DATABASE_URL = _raw_url.replace("postgres://", "postgresql://", 1)
    connect_args = {}
else:
    DATABASE_URL = "sqlite:///./algo_sessions.db"
    connect_args = {"check_same_thread": False}

# pool_pre_ping: 쿼리 실행 전에 연결이 살아있는지 가볍게 확인(SELECT 1)하고,
# Neon이 오래 쉬던 연결을 끊어버렸으면 자동으로 새 연결로 교체한다 — 이게 없으면
# 한참 쉬었다가 들어온 첫 요청이 "죽은 연결"을 그대로 쓰다가 500으로 실패하고,
# 그다음 재시도에서만 성공하는 패턴이 반복됐다.
#
# pool_size/max_overflow를 작게 제한하는 이유: 기본값(5+10=최대 15개)까지 늘어나면
# Neon 무료 플랜의 동시 연결 제한을 넘어서서 일부 요청이 응답 없이 멈추는
# 문제가 있었다 — 작게 제한해서 Neon 쪽 한도 안에서만 쓰게 한다.
# pool_timeout: 풀이 꽉 찼을 때 무한정 기다리지 않고 5초 안에 실패시켜서,
# 클라이언트가 수십 초씩 응답 없이 멈추는 대신 빠르게 에러를 받게 한다.
engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
    pool_size=3,
    max_overflow=2,
    pool_timeout=5,
    pool_recycle=280,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AlgoSession(SQLModel, table=True):
    __tablename__ = "algo_session"

    id: Optional[int] = Field(default=None, primary_key=True)
    dog_name: str = Field(default="코코")
    started_at: datetime = Field(default_factory=_utcnow)


class AlgoStep(SQLModel, table=True):
    __tablename__ = "algo_step"

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="algo_session.id", index=True)
    ensemble_score: float
    ts: datetime = Field(default_factory=_utcnow)


def init_db() -> None:
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
