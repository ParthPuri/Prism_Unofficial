"""
Text Dataset Quality Auditor — Streamlit Dashboard
Run: streamlit run dashboard.py
"""

import numpy as np
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from textaudit.loader import get_supported_extensions, inspect_columns


@st.cache_data(ttl=300, show_spinner=False)
def get_available_groq_models(api_key: str) -> list[str]:
    if not api_key or not api_key.strip():
        return []
    try:
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key.strip()}", "User-Agent": "textaudit/0.3.0"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            models = [m["id"] for m in data.get("data", []) if m.get("active", True)]
            chat_models = [m for m in models if not any(x in m.lower() for x in ("whisper", "guard", "safeguard", "orpheus"))]
            return sorted(chat_models) if chat_models else sorted(models)
    except Exception:
        return []

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

DECISION_EXPLANATIONS = {
    "CHECKS_PASSED_REVIEW_REQUIRED": {
        "title": "Checks Passed (Review Recommended)",
        "summary": "All mandatory gates and quality thresholds were met. The dataset is structurally sound and ready for model training, pending standard qualitative review.",
        "action": "Proceed to downstream training or deployment pipeline.",
        "severity": "success"
    },
    "REVIEW_REQUIRED": {
        "title": "Review Required (Non-Blocking Quality Warnings)",
        "summary": "Zero fatal blockers were encountered, but soft quality metrics (such as class imbalance, repetitive phrasing, or text length bounds) scored below the acceptance target.",
        "action": "Review the flagged metrics below. Training can still run, but accuracy or generalization on tail examples may be impacted.",
        "severity": "warning"
    },
    "BLOCKED": {
        "title": "Dataset Blocked (Fatal Pipeline Defect)",
        "summary": "One or more hard quality gates failed (e.g. train/test data leakage, conflicting intent labels, unredacted PII, or corrupted text).",
        "action": "Do not train on this data! Fix the specific row IDs listed in the findings below first.",
        "severity": "error"
    },
    "INCOMPLETE": {
        "title": "Audit Incomplete",
        "summary": "Evaluation could not complete full policy coverage due to missing required columns (e.g. missing label column or split column).",
        "action": "Provide the missing column configurations in the sidebar and rerun the audit.",
        "severity": "info"
    }
}

METRIC_INFO = {
    "split_disjointness": {
        "plain_name": "Train/Test Data Leakage",
        "what_it_means": "Checks whether customer queries from the training split accidentally appear inside the validation or test splits.",
        "why_it_matters": "If test queries exist in training data, the model will 'cheat' by memorizing answers, producing falsely inflated test accuracy that collapses in production.",
        "how_to_fix": "Remove or deduplicate the leaked queries from either the training or evaluation split."
    },
    "label_consistency": {
        "plain_name": "Contradictory Intent Labels",
        "what_it_means": "Checks if the exact same customer query was tagged with two or more conflicting intent classes.",
        "why_it_matters": "Conflicting labels give contradictory learning signals to the neural network, degrading model convergence and confidence.",
        "how_to_fix": "Inspect the affected row IDs, determine the single correct intent label, and harmonize annotations."
    },
    "privacy_pattern_clearance": {
        "plain_name": "Unredacted PII (Emails & Phones)",
        "what_it_means": "Scans customer messages for personal identifiable information candidate patterns (email addresses, telephone numbers).",
        "why_it_matters": "Training language models on raw customer PII risks memorization, data extraction attacks, and severe GDPR/privacy violations.",
        "how_to_fix": "Run an automated regex or NER anonymization script to replace emails and phones with <EMAIL> and <PHONE> tokens."
    },
    "encoding_cleanliness": {
        "plain_name": "Corrupted UTF-8 & Encoding Artifacts",
        "what_it_means": "Detects Unicode replacement characters (e.g.  / U+FFFD), broken mojibake, and control character anomalies.",
        "why_it_matters": "Broken characters distort tokenization, generating fragmented subwords and corrupting vector embeddings.",
        "how_to_fix": "Re-export data using standard UTF-8 encoding or clean inputs with `ftfy` (fixes text for you)."
    },
    "text_completeness": {
        "plain_name": "Blank or Missing Texts",
        "what_it_means": "Verifies that every row contains actual readable text after stripping empty whitespace.",
        "why_it_matters": "Null or whitespace-only inputs cause loss computation crashes and waste model capacity on blank inputs.",
        "how_to_fix": "Drop blank rows or impute missing text entries before model ingestion."
    },
    "label_validity": {
        "plain_name": "Unrecognized / Hallucinated Labels",
        "what_it_means": "Validates that every row's label matches the approved project taxonomy or label vocabulary.",
        "why_it_matters": "Typos in labels create phantom classes, shifting softmax probabilities and breaking evaluation metrics.",
        "how_to_fix": "Correct spelling errors or remap outlier labels to approved classes."
    },
    "length_validity": {
        "plain_name": "Character Length Bounds",
        "what_it_means": "Checks whether query lengths stay within reasonable bounds (e.g. between 1 and 10,000 characters).",
        "why_it_matters": "Single-character queries convey no semantic meaning, while 2,000+ character spam dumps trigger GPU out-of-memory errors.",
        "how_to_fix": "Filter out spam dumps and single-character accidental clicks."
    },
    "normalized_uniqueness": {
        "plain_name": "Duplicate Query Redundancy",
        "what_it_means": "Measures the ratio of unique customer questions versus boilerplate repetitive phrases.",
        "why_it_matters": "Heavy repetition biases the model to over-focus on frequent questions at the expense of rare intents.",
        "how_to_fix": "Apply smart deduplication or sample balancing if diversity is required."
    },
    "class_support": {
        "plain_name": "Minimum Per-Class Sample Support",
        "what_it_means": "Measures whether tail classes meet the minimum target sample size (e.g. 20 examples per class).",
        "why_it_matters": "Classes with very few training examples suffer from high error rates and poor few-shot generalization.",
        "how_to_fix": "Collect additional training data or use data augmentation for under-represented classes."
    },
    "label_completeness": {
        "plain_name": "Missing Labels",
        "what_it_means": "Ensures every training row has an associated target label assigned.",
        "why_it_matters": "Supervised training algorithms cannot compute gradient updates on unlabeled examples.",
        "how_to_fix": "Label the missing rows or exclude them from the supervised training split."
    },
    "split_completeness": {
        "plain_name": "Split Assignment",
        "what_it_means": "Ensures every row is explicitly assigned to a recognized split (train, validation, or test).",
        "why_it_matters": "Unassigned rows are discarded or cause accidental data leakage during random partitioning.",
        "how_to_fix": "Assign missing rows explicitly to train, validation, or test."
    },
    "text_type": {
        "plain_name": "Valid String Data Type",
        "what_it_means": "Verifies that all values in the text column are string objects rather than raw numbers or nulls.",
        "why_it_matters": "Non-string data types cause fatal crashes in text tokenizers and NLP pipelines.",
        "how_to_fix": "Cast all text column entries to strings during ingestion."
    }
}

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
    "Benchmark 0: Clean Baseline": (
        "Train a 10-intent banking support classifier. "
        "All data rows are clean; expect 100% Quality Index and zero blockers."
    ),
    "Benchmark 1: Split Leakage": (
        "Train an intent classifier with independent train and test evaluation splits. "
        "Prioritise split independence. Training queries leaking into test split are unacceptable."
    ),
    "Benchmark 2: Label Contradictions": (
        "Train an intent classifier. "
        "Prioritise label consistency. Conflicting labels on identical queries are unacceptable."
    ),
    "Benchmark 3: Encoding Errors": (
        "Train an intent classifier. "
        "Flag encoding anomalies, Unicode replacement characters, and control codes for review."
    ),
    "Benchmark 4: PII Leakage": (
        "Train an intent classifier from customer chats. "
        "Require complete redaction of customer contact details and personal emails/phone numbers."
    ),
    "Benchmark 5: Empty Texts": (
        "Train an intent classifier. "
        "Text completeness is mandatory. Empty or whitespace-only queries are unacceptable."
    ),
    "Benchmark 6: Invalid Labels": (
        "Train an intent classifier with approved vocabulary. "
        "Labels outside the allowed vocabulary classes are unacceptable."
    ),
    "Benchmark 7: Length Anomalies": (
        "Train an intent classifier. "
        "Flag extreme length anomalies (single character queries or 2000+ char text) for review."
    ),
    "Benchmark 8: High Redundancy": (
        "Train an intent classifier. Many customer questions use common repetitive phrasing; "
        "measure uniqueness rate without blocking the pipeline."
    ),
    "Benchmark 9: Class Imbalance": (
        "Train an intent classifier. Tail intents have limited training examples; "
        "highlight class support for review without blocking."
    ),
    "Benchmark 10: Realistic Noise": (
        "Production banking support classifier. Data contains natural repetitions, "
        "tail classes and minor formatting quirks; evaluate overall fitness for human review."
    ),
    "Custom (type your own)": "",
}

DATASET_PRESETS = {
    "BANKING77 (Full real-world dataset)": {
        "path": WORKSPACE / "data" / "banking77" / "dataset.csv",
        "label": "label",
        "split": "split",
        "labels_path": WORKSPACE / "data" / "banking77" / "labels.json",
        "context_preset": "BANKING77 — intent classifier",
        "policy": WORKSPACE / "policies" / "classification_demo.json",
        "require_redacted": False,
        "desc": "Real-world customer support intent dataset (13,083 rows across 77 intents: 10,003 train, 3,080 test).",
    },
    "SMS Spam (Full binary classifier)": {
        "path": WORKSPACE / "data" / "sms_spam" / "dataset.csv",
        "label": "label",
        "split": "split",
        "labels_path": WORKSPACE / "data" / "sms_spam" / "labels.json",
        "context_preset": "SMS Spam — binary classifier",
        "policy": WORKSPACE / "policies" / "classification_demo.json",
        "require_redacted": False,
        "desc": "SMS Spam vs Ham binary classification dataset (5,574 messages).",
    },
    "Benchmark 0: Clean Baseline (100% Quality Index)": {
        "path": WORKSPACE / "data" / "benchmarks" / "benchmark_clean_baseline.csv",
        "label": "label",
        "split": "split",
        "labels_path": WORKSPACE / "data" / "benchmarks" / "labels.json",
        "context_preset": "Benchmark 0: Clean Baseline",
        "policy": WORKSPACE / "policies" / "benchmark_policy.json",
        "require_redacted": False,
        "desc": "Control baseline (1,000 clean rows, 10 balanced classes). Expected score: 100/100, All PASS.",
    },
    "Benchmark 1: Split Leakage (Train in test split)": {
        "path": WORKSPACE / "data" / "benchmarks" / "benchmark_split_leakage.csv",
        "label": "label",
        "split": "split",
        "labels_path": WORKSPACE / "data" / "benchmarks" / "labels.json",
        "context_preset": "Benchmark 1: Split Leakage",
        "policy": WORKSPACE / "policies" / "benchmark_policy.json",
        "require_redacted": False,
        "desc": "25 training queries copied into test split. Expected outcome: BLOCKED by split_disjointness gate.",
    },
    "Benchmark 2: Label Contradictions (Conflicting intents)": {
        "path": WORKSPACE / "data" / "benchmarks" / "benchmark_label_conflict.csv",
        "label": "label",
        "split": "split",
        "labels_path": WORKSPACE / "data" / "benchmarks" / "labels.json",
        "context_preset": "Benchmark 2: Label Contradictions",
        "policy": WORKSPACE / "policies" / "benchmark_policy.json",
        "require_redacted": False,
        "desc": "20 identical queries assigned contradictory intent labels. Expected outcome: BLOCKED by label_consistency gate.",
    },
    "Benchmark 3: Encoding Errors (Corrupt UTF-8 & controls)": {
        "path": WORKSPACE / "data" / "benchmarks" / "benchmark_encoding_errors.csv",
        "label": "label",
        "split": "split",
        "labels_path": WORKSPACE / "data" / "benchmarks" / "labels.json",
        "context_preset": "Benchmark 3: Encoding Errors",
        "policy": WORKSPACE / "policies" / "benchmark_policy.json",
        "require_redacted": False,
        "desc": "20 queries with Unicode replacement chars and control codes. Expected outcome: BLOCKED by encoding_cleanliness.",
    },
    "Benchmark 4: PII Leakage (Unredacted email & phone)": {
        "path": WORKSPACE / "data" / "benchmarks" / "benchmark_pii_leakage.csv",
        "label": "label",
        "split": "split",
        "labels_path": WORKSPACE / "data" / "benchmarks" / "labels.json",
        "context_preset": "Benchmark 4: PII Leakage",
        "policy": WORKSPACE / "policies" / "benchmark_pii_policy.json",
        "require_redacted": True,
        "desc": "20 queries containing customer emails and phone numbers. Expected outcome: BLOCKED by privacy gate.",
    },
    "Benchmark 5: Empty Texts (Blank / null queries)": {
        "path": WORKSPACE / "data" / "benchmarks" / "benchmark_empty_texts.csv",
        "label": "label",
        "split": "split",
        "labels_path": WORKSPACE / "data" / "benchmarks" / "labels.json",
        "context_preset": "Benchmark 5: Empty Texts",
        "policy": WORKSPACE / "policies" / "benchmark_policy.json",
        "require_redacted": False,
        "desc": "15 blank, whitespace-only, or missing queries. Expected outcome: BLOCKED by text_completeness gate.",
    },
    "Benchmark 6: Invalid Labels (Hallucinated classes)": {
        "path": WORKSPACE / "data" / "benchmarks" / "benchmark_invalid_labels.csv",
        "label": "label",
        "split": "split",
        "labels_path": WORKSPACE / "data" / "benchmarks" / "labels.json",
        "context_preset": "Benchmark 6: Invalid Labels",
        "policy": WORKSPACE / "policies" / "benchmark_policy.json",
        "require_redacted": False,
        "desc": "20 queries assigned unrecognized labels outside the 10 approved banking classes. Expected outcome: BLOCKED by label_validity gate.",
    },
    "Benchmark 7: Length Anomalies (Extreme query lengths)": {
        "path": WORKSPACE / "data" / "benchmarks" / "benchmark_length_anomalies.csv",
        "label": "label",
        "split": "split",
        "labels_path": WORKSPACE / "data" / "benchmarks" / "labels.json",
        "context_preset": "Benchmark 7: Length Anomalies",
        "policy": WORKSPACE / "policies" / "benchmark_policy.json",
        "require_redacted": False,
        "desc": "15 single-character queries and 2,500-character spam texts violating length bounds. Expected outcome: BLOCKED by length_validity.",
    },
    "Benchmark 8: High Redundancy (Graded ~95 QI, Unblocked)": {
        "path": WORKSPACE / "data" / "benchmarks" / "benchmark_high_redundancy.csv",
        "label": "label",
        "split": "split",
        "labels_path": WORKSPACE / "data" / "benchmarks" / "labels.json",
        "context_preset": "Benchmark 8: High Redundancy",
        "policy": WORKSPACE / "policies" / "benchmark_policy.json",
        "require_redacted": False,
        "desc": "Graded Benchmark: 55% repeated boilerplate queries. Evaluates continuous scoring (~95/100, zero blockers).",
    },
    "Benchmark 9: Class Imbalance (Graded ~97 QI, Unblocked)": {
        "path": WORKSPACE / "data" / "benchmarks" / "benchmark_class_imbalance.csv",
        "label": "label",
        "split": "split",
        "labels_path": WORKSPACE / "data" / "benchmarks" / "labels.json",
        "context_preset": "Benchmark 9: Class Imbalance",
        "policy": WORKSPACE / "policies" / "benchmark_policy.json",
        "require_redacted": False,
        "desc": "Graded Benchmark: Tail classes have only 8 rows. Evaluates class support penalty (~97/100, zero blockers).",
    },
    "Benchmark 10: Realistic Noise (Graded ~94 QI, Unblocked)": {
        "path": WORKSPACE / "data" / "benchmarks" / "benchmark_realistic_noise.csv",
        "label": "label",
        "split": "split",
        "labels_path": WORKSPACE / "data" / "benchmarks" / "labels.json",
        "context_preset": "Benchmark 10: Realistic Noise",
        "policy": WORKSPACE / "policies" / "benchmark_policy.json",
        "require_redacted": False,
        "desc": "Graded Benchmark: Production-like mix of repetitions, tail classes, and quirks (~94/100, zero blockers).",
    },
}

# ── helpers ───────────────────────────────────────────────────────────────────

def decision_badge(decision: str) -> str:
    color = DECISION_COLOR.get(decision, "#95a5a6")
    label = decision.replace("_", " ")
    return (
        f'<span style="background:{color};color:white;padding:5px 16px;'
        f'border-radius:14px;font-weight:700;font-size:1.05rem">{label}</span>'
    )


def gauge_chart(value: float | None, is_blocked: bool = False, raw_value: float | None = None) -> go.Figure:
    v = value if value is not None else 0
    if is_blocked:
        color = "#e74c3c"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=v,
            number={"suffix": "/100", "font": {"size": 32, "color": color}},
            title={"text": "Quality Index — 🚫 BLOCKED", "font": {"size": 15, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "white",
                "steps": [
                    {"range": [0, 80], "color": "#fdecea"},
                    {"range": [80, 100], "color": "#f8f9fa"},
                ],
            },
        ))
        fig.update_layout(height=230, margin=dict(t=30, b=10, l=20, r=20))
        return fig

    color = "#2ecc71" if v >= 95 else "#f39c12" if v >= 80 else "#e74c3c"
    title_text = "Quality Index — ⚠️ REVIEW" if color == "#f39c12" else "Quality Index — ✅ PASSED" if color == "#2ecc71" else "Quality Index"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=v,
        number={"suffix": "/100", "font": {"size": 32, "color": color}},
        title={"text": title_text, "font": {"size": 15, "color": color}},
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
    groq_api_key: str | None = None,
    groq_model: str | None = None,
    policy_path: Path | None = None,
    require_redacted: bool = False,
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
    if require_redacted:
        cmd += ["--require-redacted"]

    env = os.environ.copy()

    if use_case and use_case.strip():
        cmd += ["--use-case", use_case.strip()]
        if provider_choice == "Groq (recommended & fast)":
            cmd += ["--provider", "groq", "--allow-remote"]
            if groq_model and groq_model.strip():
                cmd += ["--model", groq_model.strip()]
            if groq_api_key and groq_api_key.strip():
                env["GROQ_API_KEY"] = groq_api_key.strip()
        elif provider_choice == "Local Ollama (free)":
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
        # Reference policy — no model needed
        p = policy_path or (WORKSPACE / "policies" / "classification_demo.json")
        cmd += ["--policy", str(p)]

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
    dataset_mode = st.radio("Source", ["Use a bundled example", "Upload custom dataset"],
                            label_visibility="collapsed")

    if dataset_mode == "Use a bundled example":
        example = st.selectbox("Choose dataset or benchmark", list(DATASET_PRESETS.keys()))
        spec = DATASET_PRESETS[example]
        _dataset_path       = spec["path"]
        _label_col_default  = spec["label"]
        _split_col_default  = spec["split"]
        _labels_path        = spec["labels_path"]
        _context_preset     = spec["context_preset"]
        _policy_path        = spec["policy"]
        _require_redacted   = spec["require_redacted"]
        uploaded_file       = None
        st.caption(f"💡 {spec['desc']}")
    else:
        uploaded_file = st.file_uploader(
            "Dataset file",
            type=["csv", "tsv", "tab", "json", "jsonl", "ndjson", "parquet", "arrow", "feather", "txt"],
            help="Supports CSV, TSV, JSON, JSONL, Parquet, Arrow, and plain TXT"
        )
        _dataset_path       = None
        _label_col_default  = "label"
        _split_col_default  = "split"
        _labels_path        = None
        _context_preset     = "Custom (type your own)"
        _policy_path        = None
        _require_redacted   = False
        if uploaded_file is not None:
            st.caption(f"📄 Loaded file: `{uploaded_file.name}`")
            try:
                with tempfile.NamedTemporaryFile(suffix=Path(uploaded_file.name).suffix, delete=False) as tf:
                    tf.write(uploaded_file.getvalue())
                    tf_path = Path(tf.name)
                detected_cols = inspect_columns(tf_path)
                tf_path.unlink()
                if detected_cols:
                    st.caption(f"🔍 Columns: `{', '.join(detected_cols[:10])}`" + ("…" if len(detected_cols) > 10 else ""))
                    for cand in ["label", "intent", "target", "category", "sentiment", "class"]:
                        if cand in detected_cols:
                            _label_col_default = cand
                            break
                    for cand in ["split", "partition", "subset", "set"]:
                        if cand in detected_cols:
                            _split_col_default = cand
                            break
            except Exception:
                pass

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
            ["Groq (recommended & fast)", "Local Ollama (free)", "Hugging Face (paid)"],
            help=(
                "Groq is ultra-fast with generous free tier (console.groq.com). "
                "Ollama is free and runs locally offline. "
                "Hugging Face requires paid inference credits."
            ),
        )

        if provider_choice == "Groq (recommended & fast)":
            groq_api_key = st.text_input(
                "Groq API key (GROQ_API_KEY)",
                value=os.getenv("GROQ_API_KEY", ""),
                type="password",
                help="Create for free at console.groq.com/keys.",
            )
            live_models = get_available_groq_models(groq_api_key)
            if live_models:
                prioritized = [m for m in ("openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b") if m in live_models]
                rest = [m for m in live_models if m not in prioritized]
                options = prioritized + rest + ["Custom (type your own)"]
            else:
                options = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b", "Custom (type your own)"]

            groq_model_choice = st.selectbox(
                "Model",
                options,
                index=0,
                help="Models available on your Groq account. openai/gpt-oss-20b is recommended.",
            )
            st.caption("⚡ **openai/gpt-oss-20b** is the lightest, fastest instruction model on Groq (no thinking token overhead).")
            if groq_model_choice == "Custom (type your own)":
                groq_model = st.text_input("Enter Groq Model ID", value="openai/gpt-oss-20b")
            else:
                groq_model = groq_model_choice
            hf_token = None
            hf_model = None
            ollama_model = None
        elif provider_choice == "Hugging Face (paid)":
            hf_token = st.text_input(
                "Hugging Face token (HF_TOKEN)",
                type="password",
                help="Create at huggingface.co/settings/tokens. Requires Inference Providers credits.",
            )
            hf_model = st.text_input(
                "Model (optional)",
                value="Qwen/Qwen3-235B-A22B-Instruct-2507",
                help="Default is the project's configured model. Change only if you know another model supports structured JSON output.",
            )
            ollama_model = None
            groq_api_key = None
            groq_model = None
        else:
            st.caption("Make sure Ollama is running: `ollama serve` and the model is pulled.")
            ollama_model = st.text_input(
                "Ollama model name",
                value="qwen3:8b",
                help="Must be already pulled via `ollama pull <model>`. Runs at http://127.0.0.1:11434",
            )
            hf_token = None
            hf_model = None
            groq_api_key = None
            groq_model = None
    else:
        st.caption("No context → demo policy used (works offline, no model needed).")
        hf_token = None
        hf_model = None
        ollama_model = None
        groq_api_key = None
        groq_model = None
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
        st.markdown("Upload a dataset (`.csv`, `.tsv`, `.json`, `.jsonl`, `.parquet`, `.arrow`, `.txt`) with `text`, `label`, and `split` columns — or pick a bundled benchmark.")
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

if dataset_mode == "Upload custom dataset" and uploaded_file is None:
    st.error("Please upload a dataset file before running the audit.")
    st.stop()

if use_ai and provider_choice == "Groq (recommended & fast)" and not (groq_api_key and groq_api_key.strip()):
    st.error(
        "You selected Groq but didn't enter an API key. "
        "Add your GROQ_API_KEY (from console.groq.com/keys), switch to Local Ollama, or clear the business context to use the demo policy."
    )
    st.stop()

if use_ai and provider_choice == "Hugging Face (paid)" and not hf_token:
    st.error(
        "You selected Hugging Face but didn't enter a token. "
        "Add your HF_TOKEN, switch to Groq or Local Ollama, or clear the business context to use the demo policy."
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
                groq_api_key=groq_api_key,
                groq_model=groq_model,
                policy_path=_policy_path,
                require_redacted=_require_redacted,
            )
    except Exception as exc:
        st.error(f"Audit failed: {exc}")
        st.stop()

# ── unpack report ─────────────────────────────────────────────────────────────

decision = report["decision"]
qi       = report["quality_index"]
blockers = report.get("blockers", [])
is_blocked = (decision == "BLOCKED" or len(blockers) > 0)
effective_qi = report.get("effective_quality_index", 0.0 if is_blocked else qi)
metrics  = report["metrics"]
profile  = report.get("profile", {})
planner  = report.get("planner", {})

# ── top row ───────────────────────────────────────────────────────────────────

st.markdown("---")
gauge_col, decision_col, profile_col = st.columns([2, 2, 3])

with gauge_col:
    try:
        st.plotly_chart(gauge_chart(effective_qi, is_blocked=is_blocked, raw_value=qi), use_container_width=True)
    except Exception:
        if is_blocked:
            st.metric("Quality Index", f"{effective_qi:.1f} / 100", delta=f"-{report.get('gate_penalty', 0):.1f} Gate Penalty", delta_color="inverse")
        else:
            v = qi if qi is not None else 0
            st.metric("Quality Index", f"{v:.1f}/100")
            st.progress(min(1.0, max(0.0, float(v) / 100.0)))
    if is_blocked and qi is not None:
        penalty = report.get("gate_penalty", 0.0)
        st.caption(f"🚫 **Penalized Gate Score:** Effective score is **{effective_qi:.1f}/100** (-{penalty:.1f} pts penalty for {len(blockers)} failed fatal gate{'s' if len(blockers) > 1 else ''}; unpenalized raw points: {qi:.1f}/100).")
    elif decision == "REVIEW_REQUIRED":
        st.caption("⚠️ **Review Required:** Passed all fatal gates, but soft quality metrics need review.")
    elif decision == "CHECKS_PASSED_REVIEW_REQUIRED":
        st.caption("✅ **Checks Passed:** All gates and quality thresholds satisfied.")

with decision_col:
    st.markdown("### Decision")
    st.markdown(decision_badge(decision), unsafe_allow_html=True)
    st.markdown("&nbsp;", unsafe_allow_html=True)
    if is_blocked:
        st.metric("Fatal Blockers", f"{len(blockers)} blocker{'s' if len(blockers) > 1 else ''}", help="Non-negotiable gates that failed.")
        st.metric("Effective Score", f"{effective_qi:.1f} / 100", delta=f"-{report.get('gate_penalty', 0):.1f} Gate Penalty", delta_color="inverse", help="Fatal gate failure penalizes the score into the failing zone.")
    else:
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

dec_info = DECISION_EXPLANATIONS.get(decision, {
    "title": decision.replace("_", " "),
    "summary": "Audit evaluation completed.",
    "action": "Inspect metric results below.",
    "severity": "info"
})

if blockers:
    st.error(
        f"### 🚫 {dec_info['title']}\n\n"
        f"**Why:** {dec_info['summary']}\n\n"
        f"**Fatal Blockers Triggered:** `{', '.join(blockers)}`\n\n"
        f"👉 **Action Required:** {dec_info['action']}"
    )
elif decision == "REVIEW_REQUIRED":
    st.warning(
        f"### ⚠️ {dec_info['title']}\n\n"
        f"**Why:** {dec_info['summary']}\n\n"
        f"👉 **Action Required:** {dec_info['action']}"
    )
else:
    st.success(
        f"### ✅ {dec_info['title']}\n\n"
        f"**Why:** {dec_info['summary']}\n\n"
        f"👉 **Next Step:** {dec_info['action']}"
    )

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
    try:
        st.plotly_chart(metric_bar_chart(metrics), use_container_width=True)
    except Exception:
        st.info("Chart view unavailable; see Metric Details table below.")

with pie_col:
    st.subheader("Weight distribution")
    st.caption("Green = PASS · Red = FAIL")
    try:
        st.plotly_chart(weight_pie(metrics), use_container_width=True)
    except Exception:
        pass

st.markdown("---")

# ── metric table ──────────────────────────────────────────────────────────────

st.subheader("Metric details")
table_rows = []
for m in metrics:
    score_str = f"{m['score']*100:.2f}" if m["score"] is not None else "—"
    info = METRIC_INFO.get(m["metric"], {})
    display_name = f"{m['metric']} ({info.get('plain_name', '')})" if info.get("plain_name") else m["metric"]
    is_gate_failed = m.get("gate") and m["status"] != "PASS"
    gate_str = "🔒 Fatal Gate (FAILED)" if is_gate_failed else "🔒 Gated (Must Pass)" if m.get("gate") else "📊 Graded"
    if is_gate_failed:
        score_str = f"{score_str}% (Disqualified)"
    
    bad_cnt = m.get("bad_rows", 0)
    total_eval = m.get("evaluated_rows", 1) or 1
    pct_bad = f" ({100 * bad_cnt / total_eval:.1f}%)" if bad_cnt > 0 else ""
    row_ids = m.get("issue_row_ids", [])
    row_preview = ", ".join(str(x) for x in row_ids[:5]) + ("…" if len(row_ids) > 5 else "") if row_ids else "—"

    table_rows.append({
        "Status":           STATUS_EMOJI.get(m["status"], m["status"]),
        "Check Name":       display_name,
        "Affected Column":  m.get("affected_column", "—"),
        "Score":            score_str,
        "Pass min":         f"{m['pass_min']*100:.0f}%",
        "Weight":           f"{m['weight']:.3f}",
        "Gate":             gate_str,
        "Rows evaluated":   m.get("evaluated_rows", "—"),
        "Defect rows":      f"{bad_cnt}{pct_bad}",
        "Sample Bad Row IDs": row_preview,
    })
st.dataframe(table_rows, use_container_width=True, hide_index=True)

# ── findings ──────────────────────────────────────────────────────────────────

failing = [m for m in metrics if m["status"] != "PASS"]
if failing:
    st.markdown("---")
    st.subheader("🔎 Detailed Findings & Remediation Steps")
    st.caption("Click on each flagged metric below to inspect affected columns, exact row numbers, root causes, and remediation guidance.")

    # Master Corrupted Rows Inspector across all failing checks
    all_defects = []
    for fm in failing:
        for d in fm.get("sample_defects", []):
            if d.get("row_id") is not None:
                all_defects.append({
                    "Row #": d["row_id"],
                    "Column": d.get("column", fm.get("affected_column", "—")),
                    "Failed Check": fm["metric"],
                    "Severity": "🔴 FATAL GATE" if fm.get("gate") else "🟡 WARNING",
                    "Defect Cause": d["cause"]
                })
    if all_defects:
        with st.expander(f"📑 Master Corrupted Rows Table ({len(all_defects)} total defect rows across all checks)", expanded=True):
            st.markdown("All corrupted rows across the dataset in one unified view. Filter or download as CSV to fix directly in your pipeline:")
            st.dataframe(all_defects, use_container_width=True, hide_index=True)
            master_csv = ["row_id,column,failed_check,severity,defect_cause"]
            for r_item in all_defects:
                safe_cause = r_item["Defect Cause"].replace('"', '""')
                master_csv.append(f"{r_item['Row #']},{r_item['Column']},{r_item['Failed Check']},{r_item['Severity']},\"{safe_cause}\"")
            st.download_button(
                label=f"📥 Download All {len(all_defects)} Corrupted Rows Across Dataset (.csv)",
                data="\n".join(master_csv),
                file_name="all_corrupted_rows.csv",
                mime="text/csv",
                key="dl_master_defects"
            )

    for m in failing:
        score_label = (
            f"score {m['score']*100:.2f}/100" if m["score"] is not None else "not evaluated"
        )
        is_gate = m.get("gate", False)
        gate_badge = "🔴 **CRITICAL BLOCKER (GATE)**" if is_gate else "🟡 **QUALITY WARNING (GRADED)**"
        info = METRIC_INFO.get(m["metric"], {})
        plain_name = info.get("plain_name", m["metric"].replace("_", " ").title())

        with st.expander(
            f"{STATUS_EMOJI.get(m['status'], '?')} {plain_name} (`{m['metric']}`) — {score_label}",
            expanded=is_gate
        ):
            # Prominent defect diagnostic banner
            defect_summary = m.get("defect_summary") or f"Failed check: {m['metric']}"
            st.error(f"### 🚨 {defect_summary}")

            c_diag1, c_diag2 = st.columns(2)
            c_diag1.markdown(f"**📌 Affected Column:** `{m.get('affected_column', '—')}`")
            bad_cnt = m.get("bad_rows", 0)
            total_eval = m.get("evaluated_rows", 1) or 1
            c_diag1.markdown(f"**🔢 Total Defect Count:** **{bad_cnt}** rows ({100 * bad_cnt / total_eval:.1f}% defect rate)")
            c_diag2.markdown(f"**⚖️ Gate Severity:** {gate_badge}")
            c_diag2.markdown(f"**🎯 Policy Threshold:** `{m['pass_min']*100:.1f}%` (Observed: `{score_label}`)")

            st.markdown(f"**🔍 Root Cause Diagnosis:** {m.get('cause', '—')}")

            # Sample row defects table
            sample_defects = m.get("sample_defects", [])
            row_ids = m.get("issue_row_ids", [])
            if sample_defects and any(d.get("row_id") is not None for d in sample_defects):
                st.markdown(f"#### 📋 Sample Defect Row Inspector ({len(sample_defects)} sample rows shown)")
                defect_data = []
                for d in sample_defects:
                    if d.get("row_id") is not None:
                        defect_data.append({
                            "Row #": d["row_id"],
                            "Column": d.get("column", m.get("affected_column", "—")),
                            "Defect Cause": d["cause"]
                        })
                if defect_data:
                    st.dataframe(defect_data, use_container_width=True, hide_index=True)
                    csv_lines = ["row_id,column,metric,defect_cause"]
                    for d in defect_data:
                        safe_cause = d["Defect Cause"].replace('"', '""')
                        csv_lines.append(f"{d['Row #']},{d['Column']},{m['metric']},\"{safe_cause}\"")
                    st.download_button(
                        label=f"📥 Download Defect Rows for {m['metric']} (.csv)",
                        data="\n".join(csv_lines),
                        file_name=f"defect_rows_{m['metric']}.csv",
                        mime="text/csv",
                        key=f"dl_csv_{m['metric']}"
                    )
            elif row_ids:
                st.markdown(f"**Sample Corrupted Row IDs ({len(row_ids)} shown):** `{', '.join(str(x) for x in row_ids)}`")

            st.markdown("---")
            if info:
                st.markdown(f"**📖 What this check tests:** {info.get('what_it_means', '')}")
                st.markdown(f"**💥 Business & Model Impact:** {info.get('why_it_matters', '')}")
                st.markdown(f"**🛠️ How to fix:** {info.get('how_to_fix', '')}")
            else:
                st.write(m.get("details") or m.get("reason") or "No additional details.")

            if m.get("rationale"):
                st.caption(f"**Context Rationale:** {m['rationale']}")
else:
    st.markdown("---")
    with st.expander("🎉 All checks passed! Learn about what was evaluated", expanded=False):
        st.markdown("Every configured metric satisfied its required thresholds:")
        for m in metrics:
            info = METRIC_INFO.get(m["metric"], {})
            p_name = info.get("plain_name", m["metric"])
            st.markdown(f"- ✅ **{p_name}** (`{m['metric']}`): {m['score']*100:.1f}% score (Min required: {m['pass_min']*100:.1f}%)")

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
