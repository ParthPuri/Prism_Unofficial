"""
Text Dataset Quality Auditor — Streamlit Dashboard
Run: streamlit run dashboard.py
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Text Quality Auditor",
    page_icon="🔍",
    layout="wide",
)

WORKSPACE = Path(__file__).parent

DECISION_COLOR = {
    "CHECKS_PASSED_REVIEW_REQUIRED": "#2ecc71",
    "REVIEW_REQUIRED": "#f39c12",
    "BLOCKED": "#e74c3c",
    "INCOMPLETE": "#95a5a6",
}
STATUS_EMOJI = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️"}

EXAMPLE_CONTEXTS = {
    "BANKING77 — intent classifier": (
        "Train a 77-intent banking customer-support classifier. "
        "Evaluate generalisation on the publisher test split. "
        "Prioritise label consistency and split independence. "
        "Preserve legitimate short queries, repeated messages and natural class prevalence. "
        "Missing labels are unacceptable."
    ),
    "SMS Spam — binary classifier": (
        "Train a binary SMS spam/ham classifier for a messaging platform. "
        "Phone numbers can appear legitimately in ham messages. "
        "Flag encoding anomalies for review. Missing labels are unacceptable."
    ),
    "Custom (type your own)": "",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def decision_badge(decision: str) -> str:
    color = DECISION_COLOR.get(decision, "#95a5a6")
    label = decision.replace("_", " ")
    return (
        f'<span style="background:{color};color:white;padding:5px 16px;'
        f'border-radius:14px;font-weight:700;font-size:1.05rem">{label}</span>'
    )


def gauge_chart(value: float | None) -> go.Figure:
    v = value if value is not None else 0
    color = "#2ecc71" if v >= 95 else "#f39c12" if v >= 80 else "#e74c3c"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=v,
        number={"suffix": "/100", "font": {"size": 32, "color": color}},
        title={"text": "Quality Index", "font": {"size": 15}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "white",
            "steps": [
                {"range": [0, 80],  "color": "#fdecea"},
                {"range": [80, 95], "color": "#fef9e7"},
                {"range": [95, 100],"color": "#eafaf1"},
            ],
        },
    ))
    fig.update_layout(height=230, margin=dict(t=30, b=10, l=20, r=20))
    return fig


def metric_bar_chart(metrics: list[dict]) -> go.Figure:
    names  = [m["metric"].replace("_", " ") for m in metrics]
    scores = [round((m["score"] or 0) * 100, 2) for m in metrics]
    mins   = [round(m["pass_min"] * 100, 2) for m in metrics]
    colors = ["#2ecc71" if m["status"] == "PASS" else "#e74c3c" for m in metrics]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Score", x=scores, y=names, orientation="h",
        marker_color=colors,
        text=[f"{s:.1f}" for s in scores], textposition="auto",
    ))
    fig.add_trace(go.Scatter(
        name="Pass threshold", x=mins, y=names, mode="markers",
        marker=dict(symbol="line-ns", size=14, color="black",
                    line=dict(width=2, color="black")),
    ))
    fig.update_layout(
        height=max(300, 42 * len(metrics)),
        xaxis=dict(range=[0, 104], title="Score (0–100)"),
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=190, r=20, t=40, b=40),
        plot_bgcolor="#f8f9fa",
    )
    return fig


def weight_pie(metrics: list[dict]) -> go.Figure:
    labels = [m["metric"].replace("_", " ") for m in metrics]
    weights = [m["weight"] for m in metrics]
    colors  = ["#2ecc71" if m["status"] == "PASS" else "#e74c3c" for m in metrics]
    fig = go.Figure(go.Pie(
        labels=labels, values=weights, marker_colors=colors,
        hole=0.4, textinfo="label+percent",
        hovertemplate="%{label}<br>Weight: %{value:.3f}<extra></extra>",
    ))
    fig.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
    return fig


def run_audit(
    input_path: Path,
    label_col: str,
    split_col: str,
    allowed_labels_path: Path | None,
    out_dir: Path,
    use_case: str | None,
    provider_choice: str | None,
    hf_token: str | None,
    hf_model: str | None,
    ollama_model: str | None,
) -> dict:
    """Call textaudit CLI; returns the parsed report dict."""
    cmd = [
        sys.executable, "-m", "textaudit.cli",
        str(input_path),
        "--label-column", label_col,
        "--out", str(out_dir),
    ]
    if split_col.strip():
        cmd += ["--split-column", split_col.strip()]
    if allowed_labels_path:
        cmd += ["--allowed-labels", str(allowed_labels_path)]

    env = os.environ.copy()

    if use_case and use_case.strip():
        cmd += ["--use-case", use_case.strip()]
        if provider_choice == "Local Ollama (free)":
            cmd += [
                "--provider", "local",
                "--model", ollama_model.strip(),
                "--base-url", "http://127.0.0.1:11434/v1",
            ]
        else:
            # Hugging Face
            cmd += ["--provider", "huggingface", "--allow-remote"]
            if hf_model and hf_model.strip():
                cmd += ["--model", hf_model.strip()]
            if hf_token:
                env["HF_TOKEN"] = hf_token
    else:
        # Demo policy — no model needed
        cmd += ["--policy", str(WORKSPACE / "policies" / "classification_demo.json")]

    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=str(WORKSPACE), env=env)
    if result.returncode == 1:
        raise RuntimeError(result.stderr.strip() or "Audit failed (exit code 1)")
    report_path = out_dir / "report.json"
    if not report_path.exists():
        raise RuntimeError("Audit produced no report. stderr:\n" + result.stderr)
    return json.loads(report_path.read_text())


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔍 Text Quality Auditor")
    st.caption("Upload a dataset, describe your use case, get an audit report.")
    st.divider()

    # ── Step 1: Dataset ──────────────────────────────────────────────────────
    st.markdown("### Step 1 · Dataset")
    dataset_mode = st.radio("Source", ["Use a bundled example", "Upload my own CSV"],
                            label_visibility="collapsed")

    if dataset_mode == "Use a bundled example":
        example = st.selectbox("Example", ["BANKING77", "SMS Spam"])
        if example == "BANKING77":
            _dataset_path       = WORKSPACE / "data" / "banking77" / "dataset.csv"
            _label_col_default  = "label"
            _split_col_default  = "split"
            _labels_path        = WORKSPACE / "data" / "banking77" / "labels.json"
            _context_preset     = "BANKING77 — intent classifier"
        else:
            _dataset_path       = WORKSPACE / "data" / "sms_spam" / "dataset.csv"
            _label_col_default  = "label"
            _split_col_default  = "split"
            _labels_path        = WORKSPACE / "data" / "sms_spam" / "labels.json"
            _context_preset     = "SMS Spam — binary classifier"
        uploaded_file = None
    else:
        uploaded_file       = st.file_uploader("CSV file", type=["csv"])
        _dataset_path       = None
        _label_col_default  = "label"
        _split_col_default  = "split"
        _labels_path        = None
        _context_preset     = "Custom (type your own)"

    label_col = st.text_input("Label column", value=_label_col_default)
    split_col = st.text_input("Split column (optional)", value=_split_col_default)

    st.divider()

    # ── Step 2: Business context ─────────────────────────────────────────────
    st.markdown("### Step 2 · Business context")

    if dataset_mode == "Use a bundled example":
        preset_choice = st.selectbox(
            "Preset",
            list(EXAMPLE_CONTEXTS.keys()),
            index=list(EXAMPLE_CONTEXTS.keys()).index(_context_preset),
        )
        default_context = EXAMPLE_CONTEXTS[preset_choice]
    else:
        preset_choice = "Custom (type your own)"
        default_context = ""

    business_context = st.text_area(
        "Describe what this dataset is for",
        value=default_context,
        height=150,
        placeholder=(
            "e.g. Train a banking intent classifier. "
            "Missing labels are unacceptable. "
            "Prioritise split independence."
        ),
        help=(
            "Leave blank to use the demo policy (no model needed). "
            "Fill in to have the AI generate a policy tailored to your use case "
            "(requires HF_TOKEN below)."
        ),
    )

    use_ai = business_context.strip() != ""

    if use_ai:
        st.info("AI policy will be generated from your context.", icon="🤖")

        provider_choice = st.selectbox(
            "Inference provider",
            ["Hugging Face (paid)", "Local Ollama (free)"],
            help=(
                "Hugging Face requires a paid inference token. "
                "Ollama is free and runs locally — install from ollama.com and pull a model first."
            ),
        )

        if provider_choice == "Hugging Face (paid)":
            hf_token = st.text_input(
                "Hugging Face token (HF_TOKEN)",
                type="password",
                help="Create at huggingface.co/settings/tokens. Requires Inference Providers credits.",
            )
            hf_model = st.text_input(
                "Model (optional)",
                value="Qwen/Qwen3-235B-A22B-Instruct-2507:nscale",
                help="Default is the project's configured model. Change only if you know another model supports structured JSON output.",
            )
            ollama_model = None
        else:
            st.caption("Make sure Ollama is running: `ollama serve` and the model is pulled.")
            ollama_model = st.text_input(
                "Ollama model name",
                value="qwen3:8b",
                help="Must be already pulled via `ollama pull <model>`. Runs at http://127.0.0.1:11434",
            )
            hf_token = None
            hf_model = None
    else:
        st.caption("No context → demo policy used (works offline, no model needed).")
        hf_token = None
        hf_model = None
        ollama_model = None
        provider_choice = None

    st.divider()

    # ── Step 3: Run ──────────────────────────────────────────────────────────
    st.markdown("### Step 3 · Run")
    run_btn = st.button("▶ Run Audit", type="primary", use_container_width=True)

# ── main area ─────────────────────────────────────────────────────────────────

st.title("Text Dataset Quality Auditor")
st.markdown(
    "Dataset + business context → AI selects a quality policy → "
    "deterministic local checks → weighted score + blocking gates → report"
)

if not run_btn:
    st.info("Configure your dataset and business context in the sidebar, then click **▶ Run Audit**.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("📂 Dataset")
        st.markdown("Upload a CSV with `text`, `label`, and `split` columns — or pick a bundled benchmark.")
    with c2:
        st.subheader("📝 Business context")
        st.markdown(
            "Describe what you're building. The AI reads this and selects which checks "
            "to run and how much weight each gets. Leave blank to use the demo policy."
        )
    with c3:
        st.subheader("📊 Report")
        st.markdown(
            "Get a quality index (0–100), pass/fail gates, per-metric breakdown, "
            "and exact row IDs for every finding."
        )
    st.stop()

# ── validate inputs ───────────────────────────────────────────────────────────

if dataset_mode == "Upload my own CSV" and uploaded_file is None:
    st.error("Please upload a CSV file before running the audit.")
    st.stop()

if use_ai and provider_choice == "Hugging Face (paid)" and not hf_token:
    st.error(
        "You selected Hugging Face but didn't enter a token. "
        "Add your HF_TOKEN, switch to Local Ollama, or clear the business context to use the demo policy."
    )
    st.stop()

if use_ai and provider_choice == "Local Ollama (free)" and not ollama_model.strip():
    st.error("Enter an Ollama model name (e.g. `qwen3:8b`).")
    st.stop()

# ── run audit ─────────────────────────────────────────────────────────────────

with st.spinner("Running audit…"):
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            out_dir  = tmp_path / "report"
            out_dir.mkdir()

            if uploaded_file is not None:
                input_path = tmp_path / uploaded_file.name
                input_path.write_bytes(uploaded_file.read())
            else:
                input_path = _dataset_path

            report = run_audit(
                input_path=input_path,
                label_col=label_col,
                split_col=split_col,
                allowed_labels_path=_labels_path,
                out_dir=out_dir,
                use_case=business_context if use_ai else None,
                provider_choice=provider_choice,
                hf_token=hf_token,
                hf_model=hf_model,
                ollama_model=ollama_model,
            )
    except Exception as exc:
        st.error(f"Audit failed: {exc}")
        st.stop()

# ── unpack report ─────────────────────────────────────────────────────────────

decision = report["decision"]
qi       = report["quality_index"]
metrics  = report["metrics"]
blockers = report.get("blockers", [])
profile  = report.get("profile", {})
planner  = report.get("planner", {})

# ── top row ───────────────────────────────────────────────────────────────────

st.markdown("---")
gauge_col, decision_col, profile_col = st.columns([2, 2, 3])

with gauge_col:
    st.plotly_chart(gauge_chart(qi), use_container_width=True)

with decision_col:
    st.markdown("### Decision")
    st.markdown(decision_badge(decision), unsafe_allow_html=True)
    st.markdown("&nbsp;", unsafe_allow_html=True)
    cov = report.get("evaluated_weight_fraction", 1.0)
    st.metric("Coverage", f"{cov*100:.0f}%",
              help="Fraction of policy weight with evaluable evidence")
    st.metric("Elapsed", f"{report.get('elapsed_seconds', 0):.2f}s")
    src = planner.get("source", "explicit_policy_file")
    st.caption(f"Policy source: `{src}`")

with profile_col:
    st.markdown("### Dataset profile")
    splits = profile.get("recognized_split_counts", {})
    m1, m2 = st.columns(2)
    rows = profile.get("rows", "—")
    m1.metric("Total rows",    f"{rows:,}" if isinstance(rows, int) else rows)
    m2.metric("Label classes", profile.get("label_classes_observed", "—"))
    m1.metric("Train rows",    f"{splits.get('train', 0):,}")
    m2.metric("Test rows",     f"{splits.get('test', 0):,}")

# ── blocker / status banner ───────────────────────────────────────────────────

if blockers:
    st.error(f"🚫 **Blocked by:** {', '.join(blockers)}")
elif decision == "REVIEW_REQUIRED":
    st.warning("⚠️ All gates passed — some checks need human review.")
else:
    st.success("✅ All configured checks passed. Human review still required before production.")

# ── context card (AI policy only) ─────────────────────────────────────────────

if "context" in report and report["context"]:
    ctx = report["context"]
    with st.expander("🤖 AI-interpreted business context", expanded=True):
        cc1, cc2 = st.columns(2)
        cc1.markdown(f"**Purpose:** `{ctx.get('purpose','—')}`")
        cc2.markdown(f"**Privacy:** `{ctx.get('privacy','—')}`")
        st.markdown(ctx.get("description", ""))
        assumptions = ctx.get("assumptions", [])
        if assumptions:
            st.markdown("**Inferred assumptions:**")
            for a in assumptions:
                st.markdown(f"- {a}")

st.markdown("---")

# ── charts ────────────────────────────────────────────────────────────────────

bar_col, pie_col = st.columns([3, 2])

with bar_col:
    st.subheader("Per-metric scores vs. thresholds")
    st.plotly_chart(metric_bar_chart(metrics), use_container_width=True)

with pie_col:
    st.subheader("Weight distribution")
    st.caption("Green = PASS · Red = FAIL")
    st.plotly_chart(weight_pie(metrics), use_container_width=True)

st.markdown("---")

# ── metric table ──────────────────────────────────────────────────────────────

st.subheader("Metric details")
table_rows = []
for m in metrics:
    score_str = f"{m['score']*100:.2f}" if m["score"] is not None else "—"
    table_rows.append({
        "Status":         STATUS_EMOJI.get(m["status"], m["status"]),
        "Metric":         m["metric"],
        "Score":          score_str,
        "Pass min":       f"{m['pass_min']*100:.0f}",
        "Weight":         f"{m['weight']:.3f}",
        "Gate":           "🔒" if m.get("gate") else "—",
        "Rows evaluated": m.get("evaluated_rows", "—"),
        "Bad rows":       m.get("bad_rows", 0),
    })
st.dataframe(table_rows, use_container_width=True, hide_index=True)

# ── findings ──────────────────────────────────────────────────────────────────

failing = [m for m in metrics if m["status"] != "PASS"]
if failing:
    st.markdown("---")
    st.subheader("🔎 Findings requiring attention")
    for m in failing:
        score_label = (
            f"score {m['score']*100:.2f}/100" if m["score"] is not None else "not evaluated"
        )
        with st.expander(
            f"{STATUS_EMOJI.get(m['status'], '?')} **{m['metric']}** — {score_label}"
        ):
            st.write(m.get("details") or m.get("reason") or "No additional details.")
            ids = m.get("issue_row_ids", [])
            if ids:
                st.write(
                    f"**Affected row IDs ({len(ids)}):** "
                    f"{ids[:50]}{'…' if len(ids) > 50 else ''}"
                )
            st.caption(m.get("rationale", ""))

# ── policy rationale ──────────────────────────────────────────────────────────

policy_rationale = report.get("policy", {}).get("rationale")
if policy_rationale:
    st.markdown("---")
    with st.expander("📋 Policy rationale"):
        st.write(policy_rationale)

# ── reproducibility ───────────────────────────────────────────────────────────

st.markdown("---")
with st.expander("🔑 Reproducibility"):
    st.code(
        f"Input SHA-256:  {report.get('input_sha256','—')}\n"
        f"Policy SHA-256: {report.get('policy_sha256','—')}\n"
        f"Engine:         {report.get('engine_version','—')}\n"
        f"Planner source: {planner.get('source','—')}",
        language="text",
    )

with st.expander("📄 Raw report JSON"):
    st.json(report)
