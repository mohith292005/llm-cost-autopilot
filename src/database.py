from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import AsyncGenerator

from sqlalchemy import (
    Column, DateTime, Float, Integer, String, Text, select, func
)
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./autopilot.db")

engine = create_async_engine(DATABASE_URL, echo=False)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


class RequestLog(Base):
    __tablename__ = "request_logs"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    timestamp        = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    prompt_hash      = Column(String(64), nullable=False, index=True)
    prompt_preview   = Column(String(200), nullable=True)
    complexity_tier  = Column(Integer, nullable=False)
    tier_confidence  = Column(Float, nullable=True)
    routed_model     = Column(String(100), nullable=False)
    input_tokens     = Column(Integer, nullable=False)
    output_tokens    = Column(Integer, nullable=False)
    actual_cost      = Column(Float, nullable=False)
    baseline_cost    = Column(Float, nullable=False)
    latency_ms       = Column(Float, nullable=False)
    eval_score       = Column(Integer, nullable=True)
    escalated        = Column(Integer, default=0)
    response_preview = Column(String(500), nullable=True)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("  ✓ Database initialized (autopilot.db)")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


async def insert_log(
    db: AsyncSession,
    prompt: str,
    complexity_tier: int,
    tier_confidence: float,
    routed_model: str,
    input_tokens: int,
    output_tokens: int,
    actual_cost: float,
    baseline_cost: float,
    latency_ms: float,
    response_preview: str = "",
) -> RequestLog:
    log = RequestLog(
        prompt_hash=hash_prompt(prompt),
        prompt_preview=prompt[:200],
        complexity_tier=complexity_tier,
        tier_confidence=tier_confidence,
        routed_model=routed_model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        actual_cost=actual_cost,
        baseline_cost=baseline_cost,
        latency_ms=latency_ms,
        response_preview=response_preview[:500],
    )
    db.add(log)
    await db.commit()
    await db.refresh(log)
    return log


async def update_eval_score(
    db: AsyncSession,
    log_id: int,
    score: int,
    escalated: bool = False,
) -> None:
    from sqlalchemy import update
    stmt = (
        update(RequestLog)
        .where(RequestLog.id == log_id)
        .values(eval_score=score, escalated=int(escalated))
    )
    await db.execute(stmt)
    await db.commit()


async def get_stats(db: AsyncSession) -> dict:
    total_result = await db.execute(select(func.count(RequestLog.id)))
    total_requests = total_result.scalar() or 0

    if total_requests == 0:
        return {
            "total_requests": 0,
            "total_actual_cost": 0.0,
            "total_baseline_cost": 0.0,
            "total_savings": 0.0,
            "savings_pct": 0.0,
            "avg_latency_ms": 0.0,
            "model_distribution": {},
            "tier_distribution": {},
            "avg_eval_score": None,
            "escalation_count": 0,
        }

    agg = await db.execute(
        select(
            func.sum(RequestLog.actual_cost).label("actual"),
            func.sum(RequestLog.baseline_cost).label("baseline"),
            func.avg(RequestLog.latency_ms).label("avg_latency"),
            func.avg(RequestLog.eval_score).label("avg_score"),
            func.sum(RequestLog.escalated).label("escalations"),
        )
    )
    row = agg.one()

    total_actual   = float(row.actual   or 0)
    total_baseline = float(row.baseline or 0)
    total_savings  = total_baseline - total_actual
    savings_pct    = (total_savings / total_baseline * 100) if total_baseline > 0 else 0.0

    model_rows = await db.execute(
        select(RequestLog.routed_model, func.count(RequestLog.id))
        .group_by(RequestLog.routed_model)
    )
    model_dist = {r[0]: r[1] for r in model_rows}

    tier_rows = await db.execute(
        select(RequestLog.complexity_tier, func.count(RequestLog.id))
        .group_by(RequestLog.complexity_tier)
    )
    tier_dist = {str(r[0]): r[1] for r in tier_rows}

    return {
        "total_requests": total_requests,
        "total_actual_cost": round(total_actual, 6),
        "total_baseline_cost": round(total_baseline, 6),
        "total_savings": round(total_savings, 6),
        "savings_pct": round(savings_pct, 2),
        "avg_latency_ms": round(float(row.avg_latency or 0), 2),
        "avg_eval_score": round(float(row.avg_score), 2) if row.avg_score else None,
        "escalation_count": int(row.escalations or 0),
        "model_distribution": model_dist,
        "tier_distribution": tier_dist,
    }