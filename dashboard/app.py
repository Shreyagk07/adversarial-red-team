"""Streamlit dashboard for the Adversarial Multi-Agent Red-Team System.

A polished UI over the FastAPI backend:
  * Targets & Launch — register a target and kick off an evaluation.
  * Run Results      — overall score, per-category chart, failing transcripts,
                       and mitigations for a selected run.
  * Compare Runs     — before/after regression comparison.

Run with:
    streamlit run dashboard/app.py

It expects the backend running (default http://127.0.0.1:8000); set a different
URL in the sidebar. All backend access goes through dashboard/api_client.py.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import streamlit as st

from dashboard.api_client import ApiClient, ApiError

DEFAULT_BACKEND = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")

# Severity -> emoji, for quick visual scanning of failures.
SEVERITY_ICON = {
    "critical": "🟥",
    "high": "🟧",
    "medium": "🟨",
    "low": "🟦",
    "none": "🟩",
}
PRIORITY_ICON = {"high": "🔴", "medium": "🟠", "low": "🟡"}


# --- page + client setup -----------------------------------------------------
st.set_page_config(
    page_title="AI Red-Team Dashboard",
    page_icon="🛡️",
    layout="wide",
)


def get_client() -> ApiClient:
    """Build an API client from the sidebar-configured backend URL."""
    return ApiClient(st.session_state.get("backend_url", DEFAULT_BACKEND))


def sidebar() -> str:
    """Render the sidebar (backend config + health) and return the chosen view."""
    st.sidebar.title("🛡️ AI Red-Team")
    st.sidebar.caption("Automated robustness evaluation for AI agents")

    st.session_state.setdefault("backend_url", DEFAULT_BACKEND)
    st.session_state["backend_url"] = st.sidebar.text_input(
        "Backend URL", value=st.session_state["backend_url"]
    )

    # Live health indicator.
    try:
        health = get_client().health()
        key = "✅" if health.get("llm_key_configured") else "⚠️ no key"
        st.sidebar.success(f"Backend OK · {health.get('llm_provider')} {key}")
    except ApiError as exc:
        st.sidebar.error(f"Backend unreachable\n\n{exc}")

    st.sidebar.divider()
    return st.sidebar.radio(
        "View",
        ["Targets & Launch", "Run Results", "Compare Runs"],
        label_visibility="collapsed",
    )


# --- helpers -----------------------------------------------------------------
def _score_color(score: float | None) -> str:
    if score is None:
        return "gray"
    if score >= 75:
        return "green"
    if score >= 50:
        return "orange"
    return "red"


def _run_label(run: dict[str, Any]) -> str:
    score = run.get("overall_score")
    score_str = f"{score:.0f}" if score is not None else "—"
    return f"{run['id'][:8]} · {run['status']} · score {score_str} · {run['created_at'][:19]}"


# --- view: Targets & Launch --------------------------------------------------
def view_targets(client: ApiClient) -> None:
    st.header("Targets & Launch")

    col_register, col_launch = st.columns(2, gap="large")

    with col_register:
        st.subheader("Register a target")
        with st.form("register_target"):
            name = st.text_input("Name", placeholder="My Q&A agent")
            description = st.text_area(
                "Description",
                placeholder="A general-purpose Q&A/reasoning assistant.",
                height=80,
            )
            system_prompt = st.text_area(
                "System prompt (optional — leave blank for the default persona)",
                height=120,
            )
            provider = st.selectbox("Provider", ["groq", "gemini"])
            submitted = st.form_submit_button("Register", type="primary")
        if submitted:
            if not name.strip():
                st.warning("Name is required.")
            else:
                try:
                    target = client.create_target(
                        {
                            "name": name,
                            "description": description,
                            "system_prompt": system_prompt,
                            "provider": provider,
                        }
                    )
                    st.success(f"Registered target {target['id']}")
                except ApiError as exc:
                    st.error(str(exc))

    with col_launch:
        st.subheader("Launch an evaluation")
        try:
            targets = client.list_targets()
        except ApiError as exc:
            st.error(str(exc))
            targets = []

        if not targets:
            st.info("No targets yet — register one on the left.")
            return

        labels = {f"{t['name']} ({t['id'][:8]})": t["id"] for t in targets}
        chosen = st.selectbox("Target", list(labels))
        tests = st.slider("Probes per category", 1, 10, 3)

        try:
            categories = client.list_categories()
        except ApiError:
            categories = []
        cat_labels = {c["name"]: c["id"] for c in categories}
        picked = st.multiselect(
            "Categories (none = all)", list(cat_labels), default=list(cat_labels)
        )

        if st.button("🚀 Launch evaluation", type="primary"):
            body = {
                "tests_per_category": tests,
                "category_ids": [cat_labels[p] for p in picked] or None,
            }
            try:
                run = client.launch_evaluation(labels[chosen], body)
                st.session_state["last_run_id"] = run["id"]
                st.success(
                    f"Launched run {run['id']} (status: {run['status']}). "
                    "Switch to **Run Results** to watch it."
                )
            except ApiError as exc:
                st.error(str(exc))


# --- view: Run Results -------------------------------------------------------
def view_results(client: ApiClient) -> None:
    st.header("Run Results")

    try:
        runs = client.list_runs()
    except ApiError as exc:
        st.error(str(exc))
        return

    if not runs:
        st.info("No runs yet. Launch one from **Targets & Launch**.")
        return

    labels = {_run_label(r): r["id"] for r in runs}
    # Preselect the most recently launched run when available.
    default_idx = 0
    last = st.session_state.get("last_run_id")
    if last:
        for i, r in enumerate(runs):
            if r["id"] == last:
                default_idx = i
                break

    chosen = st.selectbox("Run", list(labels), index=default_idx)
    run_id = labels[chosen]

    cols = st.columns([1, 1, 6])
    if cols[0].button("🔄 Refresh"):
        st.rerun()

    run = client.get_run(run_id)
    status = run["status"]

    if status == "running":
        st.info("⏳ Run in progress — click Refresh to check again.")
        return
    if status == "failed":
        st.error(f"Run failed: {run.get('error')}")
        return

    # Completed: load and render the full report.
    try:
        report = client.get_run_report(run_id)
    except ApiError as exc:
        st.error(str(exc))
        return

    _render_report(report)


def _render_report(report: dict[str, Any]) -> None:
    # Headline metrics.
    m1, m2, m3 = st.columns(3)
    score = report["overall_score"]
    m1.metric("Overall robustness", f"{score:.1f}/100")
    m2.metric("Pass rate", f"{report['overall_pass_rate']:.0%}")
    m3.metric(
        "Probes",
        f"{report['total_passed']}/{report['total_tests']} passed",
    )
    st.markdown(
        f":{_score_color(score)}[**{report['target_name']}** — {report['target_description']}]"
    )

    # Per-category bar chart.
    st.subheader("Per-category scores")
    summaries = report.get("category_summaries", [])
    if summaries:
        df = pd.DataFrame(
            {"Score": {s["name"]: s["score"] for s in summaries}}
        )
        st.bar_chart(df, height=320)

    # Worst failing transcripts.
    st.subheader("Worst failing transcripts")
    failures = report.get("worst_failures", [])
    if not failures:
        st.success("No failures recorded. 🎉")
    for ex in failures:
        tc = ex["test_case"]
        v = ex.get("verdict") or {}
        sev = v.get("severity", "none")
        icon = SEVERITY_ICON.get(sev, "⬜")
        with st.expander(f"{icon} [{tc['category_id']}] {tc['prompt'][:80]}"):
            st.markdown(f"**Probe:** {tc['prompt']}")
            st.markdown(f"**Target answered:** {ex['response']['text']}")
            st.markdown(f"**Expected:** {tc['expected_behavior']}")
            st.markdown(f"**Judge ({sev}):** {v.get('reason', '')}")

    # Mitigations.
    st.subheader("Recommended mitigations")
    mitigations = report.get("mitigations", [])
    if not mitigations:
        st.caption("No mitigations generated (LLM unavailable at report time).")
    for m in mitigations:
        icon = PRIORITY_ICON.get(m["priority"], "⚪")
        st.markdown(
            f"{icon} **[{m['priority'].upper()}] {m['category_id']}** — {m['issue']}  \n"
            f"&nbsp;&nbsp;↳ _Fix:_ {m['suggestion']}"
        )


# --- view: Compare Runs ------------------------------------------------------
def view_compare(client: ApiClient) -> None:
    st.header("Compare Runs (before / after)")

    try:
        runs = [r for r in client.list_runs() if r["status"] == "completed"]
    except ApiError as exc:
        st.error(str(exc))
        return

    if len(runs) < 2:
        st.info("Need at least two completed runs to compare.")
        return

    labels = {_run_label(r): r["id"] for r in runs}
    c1, c2 = st.columns(2)
    before_label = c1.selectbox("Before (baseline)", list(labels), index=min(1, len(labels) - 1))
    after_label = c2.selectbox("After (improved)", list(labels), index=0)

    if st.button("Compare", type="primary"):
        try:
            cmp = client.compare(labels[before_label], labels[after_label])
        except ApiError as exc:
            st.error(str(exc))
            return

        delta = cmp["overall_delta"]
        st.metric(
            "Overall score",
            f"{cmp['overall_after']:.1f}/100",
            delta=f"{delta:+.1f} vs before",
        )
        st.caption(
            f"Improved: {cmp['num_improved']} · Regressed: {cmp['num_regressed']} "
            f"· Unchanged: {cmp['num_unchanged']}"
        )

        deltas = cmp.get("category_deltas", [])
        if deltas:
            df = pd.DataFrame(
                {
                    "Before": {d["name"]: d["before_score"] for d in deltas},
                    "After": {d["name"]: d["after_score"] for d in deltas},
                }
            )
            st.bar_chart(df, height=340)
            st.dataframe(
                pd.DataFrame(deltas)[
                    ["name", "before_score", "after_score", "delta"]
                ].rename(
                    columns={
                        "name": "Category",
                        "before_score": "Before",
                        "after_score": "After",
                        "delta": "Δ",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )


# --- main --------------------------------------------------------------------
def main() -> None:
    view = sidebar()
    client = get_client()
    if view == "Targets & Launch":
        view_targets(client)
    elif view == "Run Results":
        view_results(client)
    else:
        view_compare(client)


main()
