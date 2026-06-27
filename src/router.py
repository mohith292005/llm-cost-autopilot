from __future__ import annotations

import time
import yaml
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from .classifier import get_classifier
from .database import (
    AsyncSessionLocal,
    get_db,
    get_stats,
    init_db,
    insert_log,
)
from .evaluator import evaluate_response
from .registry import BASELINE_MODEL_ID, MODEL_REGISTRY, send_request

CONFIG_PATH = Path("config/router_config.yaml")

_routing_config: dict = {}


def load_config() -> dict:
    global _routing_config
    with open(CONFIG_PATH) as f:
        _routing_config = yaml.safe_load(f)
    print(f"  ✓ Routing config loaded: {CONFIG_PATH}")
    return _routing_config


def get_config() -> dict:
    if not _routing_config:
        load_config()
    return _routing_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🚀 LLM Cost Autopilot starting up...")
    await init_db()
    load_config()
    print("  Loading complexity classifier...")
    get_classifier()
    print("  ✓ All systems ready. Listening for requests.\n")
    yield
    print("\n  Shutting down gracefully...")


app = FastAPI(
    title="LLM Cost Autopilot",
    description="Intelligent LLM routing layer.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "auto"
    messages: list[ChatMessage]
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, ge=1, le=8192)
    stream: bool = False


class ConfigUpdate(BaseModel):
    tier: int = Field(..., ge=1, le=3)
    model_id: str


def _extract_prompt(messages: list[ChatMessage]) -> tuple[str, Optional[str]]:
    system_prompt = None
    user_prompt = ""
    for msg in messages:
        if msg.role == "system":
            system_prompt = msg.content
        elif msg.role == "user":
            user_prompt = msg.content
    return user_prompt, system_prompt


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    t_start = time.perf_counter()

    user_prompt, system_prompt = _extract_prompt(request.messages)
    if not user_prompt.strip():
        raise HTTPException(status_code=400, detail="No user message found in messages array.")

    classifier = get_classifier()
    tier, confidence = classifier.predict(user_prompt)

    config = get_config()
    tier_config = config["tiers"].get(tier, config["tiers"][2])
    routed_model_id = tier_config["model_id"]

    model_config = MODEL_REGISTRY.get(routed_model_id)
    if not model_config:
        raise HTTPException(
            status_code=500,
            detail=f"Model '{routed_model_id}' not found in registry."
        )

    try:
        llm_response = await send_request(
            prompt=user_prompt,
            model_config=model_config,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            system_prompt=system_prompt,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream LLM error: {str(e)}")

    total_latency_ms = (time.perf_counter() - t_start) * 1000

    baseline_model = MODEL_REGISTRY[BASELINE_MODEL_ID]
    baseline_cost = baseline_model.calculate_cost(
        llm_response.input_tokens, llm_response.output_tokens
    )

    log_entry = await insert_log(
        db=db,
        prompt=user_prompt,
        complexity_tier=tier,
        tier_confidence=confidence,
        routed_model=routed_model_id,
        input_tokens=llm_response.input_tokens,
        output_tokens=llm_response.output_tokens,
        actual_cost=llm_response.cost_usd,
        baseline_cost=baseline_cost,
        latency_ms=round(total_latency_ms, 2),
        response_preview=llm_response.content[:500],
    )

    background_tasks.add_task(
        evaluate_response,
        user_prompt=user_prompt,
        assistant_response=llm_response.content,
        log_id=log_entry.id,
    )

    savings = baseline_cost - llm_response.cost_usd

    response_body = {
        "id": f"autopilot-{log_entry.id}",
        "object": "chat.completion",
        "model": routed_model_id,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": llm_response.content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": llm_response.input_tokens,
            "completion_tokens": llm_response.output_tokens,
            "total_tokens": llm_response.input_tokens + llm_response.output_tokens,
        },
        "autopilot_metadata": {
            "complexity_tier": tier,
            "tier_label": tier_config["label"],
            "tier_confidence": round(confidence, 3),
            "routed_model": routed_model_id,
            "actual_cost_usd": llm_response.cost_usd,
            "baseline_cost_usd": baseline_cost,
            "savings_usd": round(savings, 8),
            "latency_ms": round(total_latency_ms, 2),
            "log_id": log_entry.id,
        },
    }

    headers = {
        "X-Autopilot-Tier":      str(tier),
        "X-Autopilot-Model":     routed_model_id,
        "X-Autopilot-Cost-USD":  str(llm_response.cost_usd),
        "X-Autopilot-Saved-USD": str(round(savings, 8)),
        "X-Autopilot-Latency":   str(round(total_latency_ms, 2)),
    }

    return JSONResponse(content=response_body, headers=headers)


@app.get("/stats")
async def get_live_stats(db: AsyncSession = Depends(get_db)):
    stats = await get_stats(db)
    return stats


@app.put("/config")
async def update_config(update: ConfigUpdate):
    if update.model_id not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{update.model_id}' not in registry. "
                   f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    config = get_config()
    old_model = config["tiers"][update.tier]["model_id"]
    config["tiers"][update.tier]["model_id"] = update.model_id

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return {
        "success": True,
        "tier": update.tier,
        "old_model": old_model,
        "new_model": update.model_id,
        "message": f"Tier {update.tier} now routes to {update.model_id}",
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}