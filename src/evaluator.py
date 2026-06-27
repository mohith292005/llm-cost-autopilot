from __future__ import annotations

import json
import re
from typing import Optional

from .registry import MODEL_REGISTRY, send_request
from .database import update_eval_score, AsyncSessionLocal

JUDGE_MODEL_ID = "openai/gpt-4o-mini"
ESCALATION_THRESHOLD = 3

JUDGE_PROMPT_TEMPLATE = """You are a quality evaluator for AI assistant responses.

ORIGINAL USER REQUEST:
{user_prompt}

AI ASSISTANT RESPONSE:
{assistant_response}

Evaluate the response on a scale of 1 to 5:
- 5: Perfect. Fully addresses the request, accurate, clear, no issues.
- 4: Good. Minor gaps or slight verbosity, but mostly correct and useful.
- 3: Acceptable. Core question answered but missing details or has minor errors.
- 2: Poor. Partially addresses the request, significant gaps or confusion.
- 1: Failure. Wrong, irrelevant, or refuses to answer when it shouldn't.

You MUST respond with ONLY valid JSON in this exact format:
{{
  "score": <integer 1-5>,
  "reason": "<one sentence explanation>"
}}

Do not include any text before or after the JSON."""


async def evaluate_response(
    user_prompt: str,
    assistant_response: str,
    log_id: int,
) -> Optional[int]:
    judge_model = MODEL_REGISTRY.get(JUDGE_MODEL_ID)
    if not judge_model:
        print(f"  Judge model {JUDGE_MODEL_ID} not in registry. Skipping evaluation.")
        return None

    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        user_prompt=user_prompt[:1000],
        assistant_response=assistant_response[:2000],
    )

    try:
        response = await send_request(
            prompt=judge_prompt,
            model_config=judge_model,
            temperature=0.0,
            max_tokens=150,
        )

        score, reason = _parse_judge_response(response.content)

        if score is None:
            print(f"  Could not parse judge response: {response.content[:100]}")
            return None

        escalated = score < ESCALATION_THRESHOLD

        async with AsyncSessionLocal() as db:
            await update_eval_score(db, log_id, score, escalated)

        if escalated:
            print(
                f"\n  ESCALATION ALERT — Log ID {log_id}\n"
                f"     Score: {score}/5 | Reason: {reason}\n"
                f"     Prompt preview: {user_prompt[:80]}...\n"
            )
        else:
            print(f"  Eval complete — Log {log_id}: {score}/5 ({reason[:60]})")

        return score

    except Exception as e:
        print(f"  Evaluator error for log {log_id}: {e}")
        return None


def _parse_judge_response(content: str) -> tuple[Optional[int], str]:
    content = re.sub(r"```(?:json)?\s*", "", content).strip()
    content = content.replace("```", "").strip()

    try:
        data = json.loads(content)
        score = int(data.get("score", 0))
        reason = str(data.get("reason", ""))

        if not (1 <= score <= 5):
            return None, ""

        return score, reason

    except (json.JSONDecodeError, ValueError, KeyError):
        match = re.search(r'"score"\s*:\s*([1-5])', content)
        if match:
            return int(match.group(1)), "Parsed from partial JSON"
        return None, ""