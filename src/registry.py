from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import httpx
from dotenv import load_dotenv
import os

load_dotenv()
try:
    import streamlit as st
    for key, value in st.secrets.items():
        os.environ.setdefault(key, value)
except Exception:
    pass

@dataclass
class ModelConfig:
    provider: str
    model_id: str
    cost_per_1m_input: float
    cost_per_1m_output: float
    quality_tier: int
    display_name: str = ""

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        input_cost  = (input_tokens  / 1_000_000) * self.cost_per_1m_input
        output_cost = (output_tokens / 1_000_000) * self.cost_per_1m_output
        return round(input_cost + output_cost, 8)


@dataclass
class LLMResponse:
    content: str
    model_id: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    raw_response: dict = field(default_factory=dict)


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "anthropic/claude-3-5-sonnet": ModelConfig(
        provider="anthropic",
        model_id="anthropic/claude-3-5-sonnet",
        cost_per_1m_input=3.00,
        cost_per_1m_output=15.00,
        quality_tier=3,
        display_name="Claude 3.5 Sonnet",
    ),
    "openai/gpt-4o": ModelConfig(
        provider="openai",
        model_id="openai/gpt-4o",
        cost_per_1m_input=5.00,
        cost_per_1m_output=15.00,
        quality_tier=3,
        display_name="GPT-4o",
    ),
    "openai/gpt-4o-mini": ModelConfig(
        provider="openai",
        model_id="openai/gpt-4o-mini",
        cost_per_1m_input=0.15,
        cost_per_1m_output=0.60,
        quality_tier=2,
        display_name="GPT-4o Mini",
    ),
    "anthropic/claude-3-haiku": ModelConfig(
        provider="anthropic",
        model_id="anthropic/claude-3-haiku",
        cost_per_1m_input=0.25,
        cost_per_1m_output=1.25,
        quality_tier=1,
        display_name="Claude 3 Haiku",
    ),
    "google/gemma-2-9b-it": ModelConfig(
        provider="google",
        model_id="google/gemma-2-9b-it",
        cost_per_1m_input=0.06,
        cost_per_1m_output=0.06,
        quality_tier=1,
        display_name="Gemma 2 9B",
    ),
    "meta-llama/llama-3.3-70b-instruct:free": ModelConfig(
        provider="meta-llama",
        model_id="meta-llama/llama-3.3-70b-instruct:free",
        cost_per_1m_input=0.0,
        cost_per_1m_output=0.0,
        quality_tier=2,
        display_name="Llama 3.3 70B (Free)",
    ),
    "meta-llama/llama-3.2-3b-instruct:free": ModelConfig(
        provider="meta-llama",
        model_id="meta-llama/llama-3.2-3b-instruct:free",
        cost_per_1m_input=0.0,
        cost_per_1m_output=0.0,
        quality_tier=1,
        display_name="Llama 3.2 3B (Free)",
    ),
    "anthropic/claude-3.5-haiku": ModelConfig(
        provider="anthropic",
        model_id="anthropic/claude-3.5-haiku",
        cost_per_1m_input=0.80,
        cost_per_1m_output=4.00,
        quality_tier=3,
        display_name="Claude 3.5 Haiku",
    ),
    "ollama/llama3.2:3b": ModelConfig(
        provider="ollama",
        model_id="ollama/llama3.2:3b",
        cost_per_1m_input=0.0,
        cost_per_1m_output=0.0,
        quality_tier=1,
        display_name="Llama 3.2 3B (Local)",
    ),
    "ollama/mistral": ModelConfig(
        provider="ollama",
        model_id="ollama/mistral",
        cost_per_1m_input=0.0,
        cost_per_1m_output=0.0,
        quality_tier=2,
        display_name="Mistral (Local)",
    ),
    "ollama/llama3.1": ModelConfig(
        provider="ollama",
        model_id="ollama/llama3.1",
        cost_per_1m_input=0.0,
        cost_per_1m_output=0.0,
        quality_tier=3,
        display_name="Llama 3.1 (Local)",
    ),
}

BASELINE_MODEL_ID = "openai/gpt-4o"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


async def send_request(
    prompt: str,
    model_config: ModelConfig,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    system_prompt: Optional[str] = None,
) -> LLMResponse:

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    start_time = time.perf_counter()

    # ── Ollama (local) ───────────────────────────────────────────────────────
    if model_config.provider == "ollama":
        ollama_model = model_config.model_id.replace("ollama/", "")
        payload = {
            "model": ollama_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            response = await client.post(
                f"{ollama_host}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        latency_ms = (time.perf_counter() - start_time) * 1000
        content = data["message"]["content"]
        input_tokens  = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)

    # ── OpenRouter (cloud) ───────────────────────────────────────────────────
    else:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env file.")

        payload = {
            "model": model_config.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/llm-cost-autopilot",
            "X-Title": "LLM Cost Autopilot",
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                OPENROUTER_API_URL,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

        latency_ms = (time.perf_counter() - start_time) * 1000
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        input_tokens  = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

    cost_usd = model_config.calculate_cost(input_tokens, output_tokens)

    return LLMResponse(
        content=content,
        model_id=model_config.model_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        latency_ms=round(latency_ms, 2),
        raw_response=data,
    )