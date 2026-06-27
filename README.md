# LLM Cost Autopilot

> Intelligent LLM routing layer — automatically sends every prompt to the cheapest capable model, saving up to 95% on AI costs.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![OpenRouter](https://img.shields.io/badge/OpenRouter-LLM_API-7C3AED?style=flat)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLMs-000000?style=flat)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)

---

## Overview

LLM Cost Autopilot is an intelligent routing layer that sits in front of multiple LLMs. Instead of blindly sending every request to an expensive model like GPT-4o, it analyzes each prompt locally using a Scikit-learn classifier, assigns it a complexity tier, and routes it to the cheapest model capable of handling it. Quality is verified asynchronously using an LLM-as-a-Judge pattern — the user never waits for evaluation.

**The result: up to 95% cost reduction with no drop in answer quality.**

---

## Features

- **Local ML Classifier** — TF-IDF + Logistic Regression runs in ~1ms on CPU, completely free, no API call needed to decide routing
- **3-Tier Routing** — Simple → cheap model, Moderate → mid model, Complex → premium model
- **OpenAI-Compatible API** — Drop-in replacement for OpenAI SDK, just change the base URL
- **Async Quality Verification** — LLM-as-a-Judge scores every response 1-5 in the background, never blocking the user
- **Escalation Alerts** — Automatically flags low-quality responses for retraining
- **Cost Dashboard** — Live Streamlit dashboard showing savings, model distribution, and latency
- **Hot-Reload Config** — Change model assignments at runtime via `PUT /config` without restarting
- **Ollama Support** — Run fully locally with zero API costs using Ollama models
- **Docker Ready** — Full stack containerized with one `docker compose up` command

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Gateway | FastAPI + Uvicorn |
| Complexity Classifier | Scikit-learn (TF-IDF + Logistic Regression) |
| LLM Providers | OpenRouter API + Ollama (local) |
| Async Quality Judge | LLM-as-a-Judge (gpt-4o-mini) |
| Database | SQLite + SQLAlchemy AsyncIO |
| Dashboard | Streamlit + Plotly |
| Containerization | Docker + Docker Compose |

---

## Project Structure

```
llm-cost-autopilot/
├── config/
│   └── router_config.yaml   ← Tier-to-model mappings (edit without touching code)
├── src/
│   ├── __init__.py
│   ├── registry.py          ← Model configs, pricing, unified send_request()
│   ├── classifier.py        ← ML complexity classifier (train + predict)
│   ├── database.py          ← SQLite AsyncIO schema + query helpers
│   ├── evaluator.py         ← Async LLM-as-a-Judge quality verifier
│   └── router.py            ← FastAPI gateway endpoints
├── dashboard.py             ← Streamlit cost savings dashboard
├── main.py                  ← Uvicorn entry point
├── stress_test.py           ← 200-request async load tester
├── Dockerfile               ← Multi-stage production image
├── docker-compose.yml       ← Full stack orchestration
├── requirements.txt
└── .env                     ← API keys (never committed)
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- OpenRouter API key → [openrouter.ai](https://openrouter.ai)
- (Optional) Ollama → [ollama.com](https://ollama.com) for free local models

### 1. Clone the repository

```bash
git clone https://github.com/mohith292005/llm-cost-autopilot.git
cd llm-cost-autopilot
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Edit the `.env` file in the project root:

```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
DATABASE_URL=sqlite+aiosqlite:///./autopilot.db
LOG_LEVEL=INFO
```

### 5. Train the classifier

```bash
python src/classifier.py
```

Expected output:
```
Classifier trained! Accuracy: 91.7%
✓ Model saved to classifier.joblib (33.8 KB)
```

### 6. Start the API server

```bash
python main.py
```

Server runs at `http://localhost:8000`
Interactive docs at `http://localhost:8000/docs`

### 7. Start the dashboard

In a second terminal:

```bash
streamlit run dashboard.py
```

Dashboard runs at `http://localhost:8501`

---

## Using Ollama (Free Local Models)

Install Ollama from [ollama.com](https://ollama.com), then pull models:

```bash
ollama pull llama3.2:3b   # Tier 1 — fast & lightweight
ollama pull mistral        # Tier 2 — balanced
ollama pull llama3.1       # Tier 3 — powerful
```

Update `config/router_config.yaml`:

```yaml
tiers:
  1:
    model_id: "ollama/llama3.2:3b"
  2:
    model_id: "ollama/mistral"
  3:
    model_id: "ollama/llama3.1"
```

No API key needed. No rate limits. Completely free.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | Main gateway (OpenAI-compatible) |
| `GET` | `/stats` | Live aggregated metrics JSON |
| `PUT` | `/config` | Hot-reload tier-to-model mappings |
| `GET` | `/health` | Liveness check |

### Example Request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"messages": [{"role": "user", "content": "What is the capital of France?"}]}'
```

### Example Response

```json
{
  "model": "openai/gpt-4o-mini",
  "choices": [{"message": {"content": "The capital of France is Paris."}}],
  "autopilot_metadata": {
    "complexity_tier": 1,
    "tier_label": "Simple",
    "tier_confidence": 0.515,
    "routed_model": "openai/gpt-4o-mini",
    "actual_cost_usd": 0.0000063,
    "baseline_cost_usd": 0.000175,
    "savings_usd": 0.0001687,
    "latency_ms": 2005.27
  }
}
```

---

## How It Works

```
User Prompt
      ↓
Local ML Classifier (~1ms, free, offline)
      ↓ assigns tier 1 / 2 / 3
router_config.yaml → picks cheapest capable model
      ↓
OpenRouter API or Ollama (local)
      ↓
Response → User immediately
      ↓ (background, non-blocking)
LLM-as-a-Judge → quality score 1-5
      ↓ if score < 3 → escalation alert
SQLite Database → Streamlit Dashboard
```

---

## Docker Deployment

```bash
docker compose up --build
```

This starts:
- `autopilot-api` → FastAPI on port 8000
- `autopilot-dashboard` → Streamlit on port 8501
- `autopilot-ollama` → Ollama on port 11434

Pull models into the Docker Ollama container:

```bash
docker exec autopilot-ollama ollama pull llama3.2:3b
docker exec autopilot-ollama ollama pull mistral
docker exec autopilot-ollama ollama pull llama3.1
```

---

## Stress Test

Populate the dashboard with 200 concurrent real requests:

```bash
python stress_test.py
```

Sample output:
```
COMPLETE — 117.9s total
Success: 194/200
Avg latency: 10081ms
Total cost:  $0.0339
Total saved: $0.8172
Savings rate: 95.7%
```

---

## Cost Savings Example

| Scenario | Model | Cost per 1M tokens | Cost for 10K requests |
|---|---|---|---|
| No routing (baseline) | GPT-4o | $5.00 input / $15.00 output | ~$8.50 |
| With Autopilot | Mixed | Tier-based | ~$0.42 |
| **Savings** | | | **~$8.08 (95%)** |

---

## Roadmap

- [ ] Streaming response support
- [ ] Multi-turn conversation routing
- [ ] Auto-retraining from escalated prompts
- [ ] Prometheus metrics export
- [ ] OpenTelemetry tracing
- [ ] Web UI for classifier training

---

## Author

**Your Name**
GitHub: [@Mohith](https://github.com/mohith292005)
