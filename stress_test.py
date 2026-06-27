import asyncio
import random
import time
from dataclasses import dataclass

import httpx

GATEWAY_URL = "http://localhost:8000/v1/chat/completions"
CONCURRENCY = 20

TEST_PROMPTS = [
    "What is the capital of Japan?",
    "Convert 72 degrees Fahrenheit to Celsius.",
    "What does HTTP stand for?",
    "Translate 'good morning' to French.",
    "Is 'level' a palindrome?",
    "What is 25% of 80?",
    "List 5 primary colors.",
    "What is the square root of 144?",
    "Fix the spelling: 'accomodation availible tommorow'",
    "What does SQL stand for?",
    "How many hours are in a week?",
    "What year was Python created?",
    "Convert this to uppercase: hello world",
    "What HTTP method is used to create a resource?",
    "How many bytes are in a kilobyte?",
    "What is the boolean result of: True AND False?",
    "What does CPU stand for?",
    "What does DRY stand for in programming?",
    "What is 2 to the power of 10?",
    "What is the difference between == and === in JavaScript?",
    "Write a professional out-of-office email for a 2-week vacation.",
    "Summarize the key benefits of microservices architecture in 4 bullet points.",
    "Explain the difference between TCP and UDP protocols simply.",
    "Write a LinkedIn post celebrating our team hitting 10,000 users.",
    "Compare relational databases vs NoSQL databases for a startup use case.",
    "Explain what a JWT token is and when to use it.",
    "Write a short job description for a senior Python developer role.",
    "Create 5 potential names for a B2B analytics SaaS startup.",
    "Explain Git rebase vs Git merge with a brief example.",
    "Summarize what agile software development means to a non-technical CEO.",
    "Explain the concept of idempotency in APIs with a practical example.",
    "Create a brief checklist for a code review process.",
    "Explain what Docker containers are to a developer who has never used them.",
    "Write 5 user stories for a task management mobile app.",
    "Explain database indexing and why it speeds up queries.",
    "Design a scalable URL shortener system supporting 100M URLs.",
    "Write a Python class implementing a thread-safe LRU cache.",
    "Design a multi-tenant database architecture for a SaaS application.",
    "Write a complete FastAPI endpoint that accepts a file upload and processes it async.",
    "Implement a backoff retry decorator in Python with jitter.",
    "Design the event sourcing architecture for an e-commerce order system.",
    "Write a Pytest test suite for a payment processing module.",
    "Design a real-time notification system for 1 million concurrent users.",
    "Write a Python async producer-consumer pipeline with backpressure.",
    "Implement a distributed lock mechanism using Redis.",
    "Design a comprehensive API rate limiting system.",
    "Write a SQL query for top 10 users by monthly spend with MoM growth.",
    "Design the database schema for a social media follow graph.",
    "Design and implement a feature flag system with percentage rollouts.",
    "Explain how to implement request deduplication in a distributed payment system.",
]


@dataclass
class TestResult:
    prompt_preview: str
    success: bool
    status_code: int
    latency_ms: float
    routed_model: str = ""
    tier: str = ""
    cost_usd: float = 0.0
    saved_usd: float = 0.0
    error: str = ""


async def send_one(client: httpx.AsyncClient, prompt: str, semaphore: asyncio.Semaphore) -> TestResult:
    async with semaphore:
        t_start = time.perf_counter()
        try:
            response = await client.post(
                GATEWAY_URL,
                json={"messages": [{"role": "user", "content": prompt}], "max_tokens": 512},
                timeout=90.0,
            )
            latency_ms = (time.perf_counter() - t_start) * 1000

            if response.status_code == 200:
                data = response.json()
                meta = data.get("autopilot_metadata", {})
                return TestResult(
                    prompt_preview=prompt[:50] + "...",
                    success=True,
                    status_code=200,
                    latency_ms=round(latency_ms, 2),
                    routed_model=meta.get("routed_model", "unknown"),
                    tier=str(meta.get("complexity_tier", "?")),
                    cost_usd=meta.get("actual_cost_usd", 0.0),
                    saved_usd=meta.get("savings_usd", 0.0),
                )
            else:
                return TestResult(
                    prompt_preview=prompt[:50] + "...",
                    success=False,
                    status_code=response.status_code,
                    latency_ms=round(latency_ms, 2),
                    error=response.text[:100],
                )
        except Exception as e:
            latency_ms = (time.perf_counter() - t_start) * 1000
            return TestResult(
                prompt_preview=prompt[:50] + "...",
                success=False,
                status_code=0,
                latency_ms=round(latency_ms, 2),
                error=str(e)[:100],
            )


async def run_stress_test(num_requests: int = 200) -> None:
    prompts = random.choices(TEST_PROMPTS, k=num_requests)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    results = []
    completed = 0

    print(f"\n{'='*60}")
    print(f"  Stress Test — {num_requests} requests, {CONCURRENCY} concurrent")
    print(f"  Target: {GATEWAY_URL}")
    print(f"{'='*60}\n")

    t_start = time.perf_counter()

    async with httpx.AsyncClient() as client:
        tasks = [send_one(client, p, semaphore) for p in prompts]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1
            icon = "✓" if result.success else "✗"
            print(
                f"  [{completed:3d}/{num_requests}] {icon} "
                f"Tier {result.tier} → {result.routed_model.split('/')[-1]:20s} "
                f"| {result.latency_ms:6.0f}ms | ${result.cost_usd:.6f}"
            )

    elapsed = time.perf_counter() - t_start
    successful = [r for r in results if r.success]
    total_cost  = sum(r.cost_usd  for r in successful)
    total_saved = sum(r.saved_usd for r in successful)
    avg_latency = sum(r.latency_ms for r in successful) / max(len(successful), 1)

    print(f"\n{'='*60}")
    print(f"  COMPLETE — {elapsed:.1f}s total")
    print(f"  Success: {len(successful)}/{num_requests}")
    print(f"  Avg latency: {avg_latency:.0f}ms")
    print(f"  Total cost:  ${total_cost:.4f}")
    print(f"  Total saved: ${total_saved:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(run_stress_test(200))