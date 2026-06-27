import asyncio
from datetime import datetime
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import select

import sys
sys.path.insert(0, ".")

from src.database import AsyncSessionLocal, RequestLog, get_stats, init_db
from dotenv import load_dotenv
load_dotenv()
try:
    for key, value in st.secrets.items():
        os.environ.setdefault(key, value)
except Exception:
    pass
def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _fetch_stats() -> dict:
    async with AsyncSessionLocal() as db:
        return await get_stats(db)


async def _fetch_recent_logs(limit: int = 500) -> list[dict]:
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(RequestLog).order_by(RequestLog.timestamp.desc()).limit(limit)
        )
        rows = result.scalars().all()
        return [
            {
                "id":             r.id,
                "timestamp":      r.timestamp,
                "prompt_preview": r.prompt_preview or "",
                "tier":           r.complexity_tier,
                "model":          r.routed_model,
                "input_tokens":   r.input_tokens,
                "output_tokens":  r.output_tokens,
                "actual_cost":    r.actual_cost,
                "baseline_cost":  r.baseline_cost,
                "savings":        r.baseline_cost - r.actual_cost,
                "latency_ms":     r.latency_ms,
                "eval_score":     r.eval_score,
                "escalated":      bool(r.escalated),
            }
            for r in rows
        ]


st.set_page_config(
    page_title="LLM Cost Autopilot",
    page_icon="🚀",
    layout="wide",
)

st.title("🚀 LLM Cost Autopilot")
st.caption("Routing every prompt to the cheapest capable model.")

col_refresh, _ = st.columns([1, 5])
with col_refresh:
    if st.button("🔄 Refresh"):
        st.rerun()

st.divider()

run_async(init_db())

with st.spinner("Loading metrics..."):
    stats = run_async(_fetch_stats())
    logs  = run_async(_fetch_recent_logs())

df = pd.DataFrame(logs) if logs else pd.DataFrame()

st.subheader("💰 The Money Shot")
m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.metric("Total Savings", f"${stats['total_savings']:.4f}", "vs GPT-4o baseline")
with m2:
    st.metric("Cost Reduction", f"{stats['savings_pct']:.1f}%")
with m3:
    st.metric("Total Requests", f"{stats['total_requests']:,}")
with m4:
    st.metric("Actual Spend", f"${stats['total_actual_cost']:.4f}")
with m5:
    avg = stats.get("avg_eval_score")
    st.metric("Avg Quality Score", f"{avg:.1f}/5" if avg else "—",
              f"{stats['escalation_count']} escalations")

st.divider()
# ─────────────────────────────────────────────────────────────────────────────
# Live Chat Tester (works both locally and on Streamlit Cloud)
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("🧪 Live Chat Tester")

with st.form("chat_form"):
    user_input = st.text_area(
        "Type your prompt here:",
        placeholder="e.g. What is the capital of France?",
        height=100,
    )
    submitted = st.form_submit_button("Send ⚡")

if submitted and user_input.strip():
    with st.spinner("Routing request..."):
        try:
            import sys
            sys.path.insert(0, ".")
            from src.classifier import get_classifier
            from src.registry import MODEL_REGISTRY, send_request

            # Classify locally
            clf = get_classifier()
            tier, confidence = clf.predict(user_input)

            # Load config
            import yaml
            with open("config/router_config.yaml") as f:
                config = yaml.safe_load(f)

            tier_config = config["tiers"].get(tier, config["tiers"][2])
            routed_model_id = tier_config["model_id"]
            model_config = MODEL_REGISTRY.get(routed_model_id)

            if not model_config:
                st.error(f"Model {routed_model_id} not in registry.")
            else:
                # Call LLM directly
                import asyncio
                response = run_async(send_request(
                    prompt=user_input,
                    model_config=model_config,
                ))

                # Calculate savings
                baseline = MODEL_REGISTRY["openai/gpt-4o"]
                baseline_cost = baseline.calculate_cost(
                    response.input_tokens, response.output_tokens
                )
                savings = baseline_cost - response.cost_usd

                tier_colors = {1: "🟢", 2: "🟡", 3: "🔴"}
                st.success("Response received!")

                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("Tier", f"{tier_colors.get(tier, '?')} {tier_config['label']}")
                with col_b:
                    st.metric("Model", routed_model_id.split("/")[-1])
                with col_c:
                    st.metric("Actual Cost", f"${response.cost_usd:.6f}")
                with col_d:
                    st.metric("Saved vs GPT-4o", f"${savings:.6f}")

                st.markdown("**Response:**")
                st.markdown(response.content)

        except Exception as e:
            st.error(f"Error: {e}")
st.divider()

if not df.empty:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🥧 Model Distribution")
        model_counts = df["model"].value_counts().reset_index()
        model_counts.columns = ["Model", "Requests"]
        model_counts["Model"] = model_counts["Model"].str.split("/").str[-1]
        fig = px.pie(model_counts, values="Requests", names="Model", hole=0.4)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Tier Distribution")
        tier_counts = df["tier"].value_counts().sort_index().reset_index()
        tier_counts.columns = ["Tier", "Count"]
        tier_counts["Label"] = tier_counts["Tier"].map(
            {1: "Simple 🟢", 2: "Moderate 🟡", 3: "Complex 🔴"}
        )
        fig2 = px.bar(tier_counts, x="Label", y="Count",
                      color="Label", text="Count",
                      color_discrete_map={"Simple 🟢": "#00c853",
                                          "Moderate 🟡": "#ffd600",
                                          "Complex 🔴": "#dd2c00"})
        fig2.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📈 Cumulative Savings Over Time")
    df_sorted = df.sort_values("timestamp").copy()
    df_sorted["cumulative_savings"] = df_sorted["savings"].cumsum()
    df_sorted["request_number"] = range(1, len(df_sorted) + 1)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df_sorted["request_number"],
        y=df_sorted["cumulative_savings"],
        name="Cumulative Savings ($)",
        line=dict(color="#00c853", width=2),
    ))
    fig3.add_trace(go.Scatter(
        x=df_sorted["request_number"],
        y=df_sorted["latency_ms"],
        name="Latency (ms)",
        line=dict(color="#2196F3", width=1.5, dash="dot"),
        yaxis="y2",
    ))
    fig3.update_layout(
        height=400,
        xaxis_title="Request #",
        yaxis=dict(title="Cumulative Savings (USD)"),
        yaxis2=dict(title="Latency (ms)", overlaying="y", side="right"),
        hovermode="x unified",
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("📋 Recent Requests")
    display = df.copy()
    display["tier"] = display["tier"].map({1: "🟢 Simple", 2: "🟡 Moderate", 3: "🔴 Complex"})
    display["actual_$"] = display["actual_cost"].apply(lambda x: f"${x:.6f}")
    display["saved_$"]  = display["savings"].apply(lambda x: f"${x:.6f}")
    display["eval"] = display["eval_score"].apply(lambda x: f"{'⭐'*int(x)}" if x is not None and not pd.isna(x) else "⏳")    
    st.dataframe(
        display[["id", "timestamp", "tier", "model", "actual_$",
                 "saved_$", "latency_ms", "eval", "prompt_preview"]].head(50),
        hide_index=True,
        use_container_width=True,
    )
else:
    st.info("No requests yet. Start the API server and send some requests!")

with st.sidebar:
    st.header("⚙️ Config")
    try:
        import yaml
        with open("config/router_config.yaml") as f:
            config = yaml.safe_load(f)
        for tier_num, tier_data in config["tiers"].items():
            st.markdown(f"**Tier {tier_num}:** `{tier_data['model_id']}`")
    except Exception as e:
        st.error(f"Could not load config: {e}")
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    if submitted and user_input.strip():
        if submitted and user_input.strip():
    with st.spinner("Routing request..."):
        try:
            from src.classifier import get_classifier
            from src.registry import MODEL_REGISTRY, send_request
            import yaml

            clf = get_classifier()
            tier, confidence = clf.predict(user_input)

            with open("config/router_config.yaml") as f:
                config = yaml.safe_load(f)

            tier_config = config["tiers"].get(tier, config["tiers"][2])
            routed_model_id = tier_config["model_id"]
            model_config = MODEL_REGISTRY.get(routed_model_id)

            if not model_config:
                st.error(f"Model {routed_model_id} not in registry.")
            else:
                response = run_async(send_request(
                    prompt=user_input,
                    model_config=model_config,
                ))

                baseline = MODEL_REGISTRY["openai/gpt-4o"]
                baseline_cost = baseline.calculate_cost(
                    response.input_tokens, response.output_tokens
                )
                savings = baseline_cost - response.cost_usd

                tier_colors = {1: "🟢", 2: "🟡", 3: "🔴"}
                st.success("Response received!")

                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("Tier", f"{tier_colors.get(tier, '?')} {tier_config['label']}")
                with col_b:
                    st.metric("Model", routed_model_id.split("/")[-1])
                with col_c:
                    st.metric("Actual Cost", f"${response.cost_usd:.6f}")
                with col_d:
                    st.metric("Saved vs GPT-4o", f"${savings:.6f}")

                st.markdown("**Response:**")
                st.markdown(response.content)

        except Exception as e:
            st.error(f"Error: {e}")