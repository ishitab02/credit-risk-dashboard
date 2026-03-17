import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine

from src.pipeline import (
    get_pipeline_history, get_pipeline_summary, run_pipeline,
    get_sla_status, get_duration_trend, PIPELINE_DAG,
)

st.set_page_config(page_title="Pipeline Monitor", page_icon="ETL", layout="wide")


@st.cache_resource
def get_engine():
    return create_engine(
        "sqlite:///database/credit_risk.db",
        connect_args={"check_same_thread": False},
    )


engine = get_engine()

st.title("ETL Pipeline Monitor")
st.markdown("Pipeline health, data quality results, and run history.")
st.markdown("---")

col_btn, col_status = st.columns([1, 3])
with col_btn:
    if st.button("Re-run Pipeline", type="primary"):
        with st.spinner("Running ETL pipeline..."):
            results = run_pipeline(force=True)
            st.cache_data.clear()
            st.session_state["last_pipeline_results"] = results
            st.rerun()

if "last_pipeline_results" in st.session_state:
    with col_status:
        results = st.session_state["last_pipeline_results"]
        passed = sum(1 for r in results if r["status"] == "success")
        st.success(f"Pipeline complete: {passed}/{len(results)} tables loaded successfully.")
    del st.session_state["last_pipeline_results"]

summary = get_pipeline_summary(engine)
st.subheader("Pipeline Health")

if summary["total_runs"] == 0:
    st.info("No pipeline runs recorded yet. Click 'Re-run Pipeline' to start.")
    st.stop()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Runs", summary["total_runs"])
k2.metric("Successes", summary["successes"])
k3.metric("Failures", summary["failures"])
k4.metric("Skipped (No Changes)", summary["skips"])

st.markdown("---")
st.subheader("Latest Run Per Table")

latest = summary["latest_runs"]
if not latest.empty:
    for _, row in latest.iterrows():
        table = row["table_name"]
        status = row["status"]
        icon = {"success": ":green[PASS]", "failed": ":red[FAIL]", "skipped": ":orange[SKIP]"}.get(status, status)

        with st.expander(f"**{table}** — {icon}", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows Loaded", f"{int(row['rows_loaded']):,}" if row["rows_loaded"] else "—")
            c2.metric("DQ Passed", int(row["dq_checks_passed"]) if row["dq_checks_passed"] else 0)
            c3.metric("DQ Failed", int(row["dq_checks_failed"]) if row["dq_checks_failed"] else 0)
            c4.metric("Duration", f"{row['duration_secs']:.1f}s" if row["duration_secs"] else "—")

            last_run = row["last_run"]
            st.caption(f"Last run: {last_run}")

st.markdown("---")
st.subheader("Pipeline DAG")

dag = PIPELINE_DAG
node_map = {n["id"]: n for n in dag["nodes"]}

layer_x = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
layer_counts = {}
for node in dag["nodes"]:
    layer = node["layer"]
    layer_counts[layer] = layer_counts.get(layer, 0) + 1

layer_current = {k: 0 for k in layer_counts}
node_positions = {}
for node in dag["nodes"]:
    layer = node["layer"]
    total_in_layer = layer_counts[layer]
    idx = layer_current[layer]
    x = layer * 1.5
    y = (idx - (total_in_layer - 1) / 2) * 1.2
    node_positions[node["id"]] = (x, y)
    layer_current[layer] += 1

type_colors = {"source": "#3498db", "process": "#f39c12", "sink": "#2ecc71"}

fig_dag = go.Figure()

for src, dst in dag["edges"]:
    x0, y0 = node_positions[src]
    x1, y1 = node_positions[dst]
    fig_dag.add_trace(go.Scatter(
        x=[x0, x1, None], y=[y0, y1, None],
        mode="lines",
        line=dict(color="#555", width=1.5),
        hoverinfo="none",
        showlegend=False,
    ))

for node in dag["nodes"]:
    x, y = node_positions[node["id"]]
    color = type_colors.get(node["type"], "#888")
    fig_dag.add_trace(go.Scatter(
        x=[x], y=[y],
        mode="markers+text",
        marker=dict(size=30, color=color, line=dict(color="#fff", width=1)),
        text=[node["label"]],
        textposition="bottom center",
        textfont=dict(size=10, color="#e8eaf0"),
        hoverinfo="text",
        hovertext=node["label"],
        showlegend=False,
    ))

fig_dag.update_layout(
    template="plotly_dark",
    height=400,
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    margin=dict(l=20, r=20, t=20, b=20),
)

for label, color in [("Source (CSV)", "#3498db"), ("Process (ETL Step)", "#f39c12"), ("Sink (SQLite)", "#2ecc71")]:
    fig_dag.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=10, color=color),
        name=label,
    ))

st.plotly_chart(fig_dag, use_container_width=True)

st.markdown("---")
st.subheader("SLA Monitoring")

sla_df = get_sla_status(engine)
if not sla_df.empty:
    sla_col1, sla_col2 = st.columns([2, 1])

    with sla_col1:
        trend_df = get_duration_trend(engine)
        if not trend_df.empty:
            fig_trend = px.line(
                trend_df, x="run_timestamp", y="duration_secs",
                color="table_name",
                title="Pipeline Duration Trend",
                template="plotly_dark",
                labels={"run_timestamp": "Run Time", "duration_secs": "Duration (s)", "table_name": "Table"},
            )
            fig_trend.update_layout(height=300)
            st.plotly_chart(fig_trend, use_container_width=True)

    with sla_col2:
        st.markdown("**SLA Status**")
        for _, row in sla_df.iterrows():
            status = row["sla_status"]
            icon = {"OK": ":green[OK]", "WARNING": ":orange[WARNING]", "BREACH": ":red[BREACH]"}.get(status, status)
            st.markdown(f"**{row['table_name']}** — {icon}")
            st.caption(f"Latest: {row['latest_duration']}s | Avg: {row['avg_duration']}s | Ratio: {row['sla_ratio']}x")

        st.markdown("---")
        st.caption("SLA thresholds: OK < 1.5x avg | WARNING 1.5-2x | BREACH > 2x")
else:
    st.info("Not enough run history for SLA analysis. Run the pipeline a few times to build a baseline.")

st.markdown("---")
st.subheader("Data Quality Check Details")

history = get_pipeline_history(engine, limit=10)
if not history.empty:
    for table_name in ["applications", "bureau", "previous_application"]:
        table_runs = history[history["table_name"] == table_name]
        if table_runs.empty:
            continue

        dq_detail_sql = f"""
        SELECT dq_details FROM pipeline_runs
        WHERE table_name = '{table_name}' AND dq_details IS NOT NULL
        ORDER BY run_timestamp DESC LIMIT 1
        """
        dq_row = pd.read_sql(dq_detail_sql, engine)
        if dq_row.empty:
            continue

        raw = dq_row["dq_details"].iloc[0]
        if not raw:
            continue

        details = json.loads(raw)
        st.markdown(f"**{table_name}**")

        rows = []
        for check in details:
            status_icon = "PASS" if check["passed"] else "FAIL"
            rows.append({
                "Check": check["check_name"],
                "Status": status_icon,
                "Details": check["details"],
            })
        check_df = pd.DataFrame(rows)

        def highlight_status(val):
            if val == "PASS":
                return "color: #2ecc71"
            return "color: #e74c3c"

        styled = check_df.style.map(highlight_status, subset=["Status"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("Run History")

if not history.empty:
    display_df = history[[
        "run_timestamp", "table_name", "status", "rows_loaded",
        "dq_checks_passed", "dq_checks_failed", "duration_secs", "error_message"
    ]].copy()
    display_df.columns = [
        "Timestamp", "Table", "Status", "Rows",
        "DQ Passed", "DQ Failed", "Duration (s)", "Error"
    ]
    display_df["Error"] = display_df["Error"].fillna("")

    def color_status(val):
        colors = {"success": "color: #2ecc71", "failed": "color: #e74c3c", "skipped": "color: #f39c12"}
        return colors.get(val, "")

    styled_hist = display_df.style.map(color_status, subset=["Status"])
    st.dataframe(styled_hist, use_container_width=True, hide_index=True)
else:
    st.info("No run history available.")

st.markdown("---")
st.caption("ETL Pipeline — schema validation, data quality checks, checksum-based incremental detection")