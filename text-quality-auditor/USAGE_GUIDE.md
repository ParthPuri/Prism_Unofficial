# Text Dataset Quality Auditor — Usage Guide

This guide explains how to install and use the dashboard to assess the quality of a text dataset.

---

## What this tool does

You provide a text dataset (CSV) and describe what you are building with it. The tool runs a set of quality checks on the dataset and produces a scored report telling you whether the data is ready to use, needs review, or is blocked due to a critical issue.

Examples of what it checks:
- Are all text fields present and non-empty?
- Do all labels belong to the declared vocabulary?
- Is there any text that appears in both the training and test split (data leakage)?
- Are there encoding anomalies or unexpected control characters?
- Does every class have enough training examples?

The final output is a quality index (0–100) and a clear pass/fail verdict.

---

## Requirements

- Python 3.11 or higher
- pip

Check your Python version:
```bash
python3 --version
```

---

## Installation

**Step 1 — Download or clone the project**

If you received this as a zip file, extract it to a folder on your computer.

**Step 2 — Open a terminal and navigate to the project folder**

```bash
cd path/to/text-quality-auditor
```

**Step 3 — Install dependencies**

```bash
pip install streamlit plotly
pip install -e ".[arrow]"  # installs textaudit with Parquet & Arrow support
```

This installs the dashboard and the auditor engine with all format loaders. It only needs to be done once.

---

## Starting the dashboard

```bash
streamlit run dashboard.py
```

The dashboard opens automatically in your browser at `http://localhost:8501`.

Keep the terminal open while using the dashboard — closing it stops the server.

---

## Using the dashboard

The sidebar has three steps.

---

### Step 1 · Dataset

Choose how to provide your dataset:

**Use a bundled example**
Two real benchmark datasets are included and ready to use:
- **BANKING77** — 13,083 banking customer support messages across 77 intent categories
- **SMS Spam** — 5,574 SMS messages labelled as spam or ham

Select one from the dropdown. No file upload needed.

**Upload custom dataset**
Click the upload button and select your dataset file. The auditor natively supports 7 major textual dataset formats:
- **CSV (`.csv`) & TSV (`.tsv`, `.tab`)**: Standard comma or tab-delimited tabular datasets.
- **JSON Lines (`.jsonl`, `.ndjson`)**: Streamable newline-delimited JSON objects (ideal for instruction-tuning and conversational datasets).
- **Structured JSON (`.json`)**: Array of objects `[{"text":...}]`, envelope payloads (`data`, `rows`, `records`), or Hugging Face split mappings (`{"train":[...], "test":[...]}`).
- **Parquet (`.parquet`)**: Columnar storage format default on Hugging Face Hub (fast, compressed, streamable).
- **Apache Arrow / Feather (`.arrow`, `.feather`)**: Hugging Face `datasets` in-memory/cache format.
- **Plain Text (`.txt`)**: Raw text corpora (one document per line) for language modeling or tokenization.

When you upload a file, the dashboard automatically inspects and displays the available columns. Set the column names:
- **Label column** — the column containing category labels (default: `label`)
- **Split column** — the column indicating train/test split (default: `split`, leave blank if not present)

---

### Step 2 · Business context

Describe what this dataset is for in plain English. A few sentences is enough.

Examples:
> "Train a 77-intent banking customer-support classifier. Missing labels are unacceptable. Prioritise split independence."

> "Build a spam detection model for a messaging app. Phone numbers can appear legitimately in non-spam messages."

> "Create a retrieval corpus for a document search system. Deduplication matters."

The tool reads this description and automatically selects which quality checks are relevant for your use case and how much weight each gets.

**Leave this blank** to skip AI policy generation and use the built-in demo policy instead. This works offline with no model required and is the fastest way to get a result.

---

#### Inference provider (only shown when business context is provided)

If you enter a business context, select how the AI policy should be generated:

---

**Option A — Google AI Studio / Gemini (Free Tier, Recommended)**

Generous free-tier cloud inference using Google's frontier Gemini models (`gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-2.0-flash`).

1. Go to [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) and create a free API key.
2. Select **Google AI Studio / Gemini (free tier)** in the dashboard and paste your API key (or set `GEMINI_API_KEY` or `GOOGLE_API_KEY` in your environment).
3. Default model is `gemini-2.5-flash` (fast, structured policy planning).

---

**Option B — Groq (Fast & Free Tier)**

Ultra-fast cloud inference with generous free tier quotas.

1. Go to [console.groq.com/keys](https://console.groq.com/keys) and create a free API key.
2. Select **Groq (recommended & fast)** in the dashboard and paste your API key (or set `GROQ_API_KEY` in your environment).
3. Default model is `openai/gpt-oss-20b` or `qwen/qwen3.8-27b`.

---

**Option C — Local Ollama (free, offline)**

Runs entirely on your machine. No internet connection or account required.

1. Install Ollama from [ollama.com](https://ollama.com)

2. Open a terminal and pull the model (one-time download, ~5GB):
   ```bash
   ollama pull qwen3:8b
   ```

3. Start the Ollama server (keep this terminal open):
   ```bash
   ollama serve
   ```

4. In the dashboard, select **Local Ollama (free)** and enter `qwen3:8b` as the model name.

---

**Option C — Hugging Face (requires paid credits)**

Uses the Hugging Face Inference API. Requires an account with inference credits.

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new token with **repo** and **Inference** permissions
3. Select **Hugging Face (paid)** in the dashboard and paste the token

Note: the default model (`Qwen/Qwen3-235B-A22B-Instruct-2507`) requires paid inference credits. A free HF account will get a 402 error.

---

### Step 3 · Run

Click **▶ Run Audit**.

The audit completes in under a second for the bundled datasets (plus model inference time if using AI policy generation).

---

## Reading the report

### Quality index
A number from 0 to 100. Higher is better. Displayed as a gauge chart coloured:
- 🟢 Green — 95 and above
- 🟡 Amber — 80 to 94
- 🔴 Red — below 80

### Decision badge
- **CHECKS PASSED REVIEW REQUIRED** — all checks passed, data looks good, human review still needed
- **REVIEW REQUIRED** — all gates passed but some non-critical checks flagged issues
- **BLOCKED** — at least one mandatory gate failed; data must not be used as-is

### Dataset profile
Row count, number of label classes, and train/test split sizes.

### Per-metric bar chart
Every check is shown as a horizontal bar. The black tick mark is the pass threshold. Green bars passed, red bars failed.

### Weight distribution pie chart
Shows how much each check contributes to the overall quality index. Checks the AI considered more important for your use case get a larger slice.

### Metric table
A table with every check showing:
- Score (0–100)
- Pass minimum threshold
- Weight
- Gate (🔒 means mandatory — a fail here blocks the dataset)
- Number of rows evaluated and number of bad rows

### Findings
Expandable cards for any failing checks. Each card shows what went wrong and the exact row IDs of affected rows so you can inspect them directly.

### Policy rationale
Explains why this particular set of checks and weights was chosen for your use case.

### Reproducibility
SHA-256 hashes of the input file and policy, so the same inputs always produce the same result.

### Raw JSON
The complete machine-readable report for integration with other tools or storage.

---

## Common errors

| Error | Cause | Fix |
|---|---|---|
| `Connection refused [Errno 61]` | Ollama is not running | Run `ollama serve` in a terminal and keep it open |
| `Planner HTTP 402` | HF model requires paid credits | Switch to Local Ollama or clear business context |
| `Planner HTTP 401` | Invalid or expired HF token | Generate a new token at huggingface.co/settings/tokens |
| `Audit failed (exit code 1)` | Missing column or bad CSV | Check column names match what you entered in the sidebar |
| Dashboard won't open | Port 8501 in use | Run `streamlit run dashboard.py --server.port 8502` |

---

## Quick start (no model, instant results)

1. Install: `pip install streamlit plotly && pip install -e .`
2. Run: `streamlit run dashboard.py`
3. Select **BANKING77** from the bundled examples
4. Leave business context **blank**
5. Click **▶ Run Audit**

You will see a quality index of 99.94/100 with a BLOCKED verdict — blocked because 7 test rows overlap with training after text normalisation. This is a real finding in the published dataset.

---

## Synthetic Corruption Benchmarks (Ground Truth Evaluation)

To rigorously verify the auditor's detection accuracy, 11 dedicated synthetic benchmark datasets are bundled under `data/benchmarks/`:

### Fatal Defect Benchmarks (Hard Gates → BLOCKED)
1. **`benchmark_clean_baseline.csv`**: Pristine control baseline (1,000 clean rows, 10 balanced classes) → **100/100 Quality Index, CHECKS PASSED**.
2. **`benchmark_split_leakage.csv`**: 25 training queries copied into the test split → **BLOCKED by split_disjointness**.
3. **`benchmark_label_conflict.csv`**: 20 identical queries assigned contradictory labels → **BLOCKED by label_consistency**.
4. **`benchmark_encoding_errors.csv`**: 20 queries with corrupt UTF-8 bytes and terminal codes → **BLOCKED by encoding_cleanliness**.
5. **`benchmark_pii_leakage.csv`**: 20 queries with customer email addresses and phone numbers → **BLOCKED by privacy gate**.
6. **`benchmark_empty_texts.csv`**: 15 blank, whitespace-only, or missing queries → **BLOCKED by text_completeness**.
7. **`benchmark_invalid_labels.csv`**: 20 queries with hallucinated labels outside allowed classes → **BLOCKED by label_validity**.
8. **`benchmark_length_anomalies.csv`**: 15 queries violating min/max character limits → **BLOCKED by length_validity**.

### Graded Quality Benchmarks (Continuous Scoring → REVIEW_REQUIRED, 0 Blockers)
9. **`benchmark_high_redundancy.csv`**: 55% repeated boilerplate queries → **~95.6/100 Quality Index, REVIEW_REQUIRED** (0 blockers).
10. **`benchmark_class_imbalance.csv`**: Head classes have 80 rows, 2 tail classes have only 8 rows → **~97.0/100 Quality Index, REVIEW_REQUIRED** (0 blockers).
11. **`benchmark_realistic_noise.csv`**: Production-like mix of repetitions, tail classes, and character quirks → **~94.6/100 Quality Index, REVIEW_REQUIRED** (0 blockers).

### Running the Benchmark Accuracy Evaluator
Run the automated evaluation script to verify all 11 benchmarks:
```bash
python scripts/verify_benchmarks.py
```

### Testing from the Dashboard
In **Step 1 (Dataset)**, select any benchmark from the **"Choose dataset or benchmark"** dropdown, leave business context blank (or enter a custom context), and click **▶ Run Audit** to see the score, verdict, and findings.

---

## File structure

```
text-quality-auditor/
├── dashboard.py          ← the web dashboard (start here)
├── textaudit/            ← auditor engine
├── data/
│   ├── banking77/        ← BANKING77 dataset
│   ├── sms_spam/         ← SMS Spam dataset
│   └── benchmarks/       ← 8 synthetic corruption benchmark CSVs + ground_truth.json
├── policies/
│   ├── classification_demo.json   ← demo policy (no model needed)
│   ├── benchmark_policy.json      ← benchmark reference policy
│   └── benchmark_pii_policy.json  ← benchmark policy with privacy gate
├── scripts/
│   ├── generate_benchmarks.py     ← reproducible benchmark generator
│   └── verify_benchmarks.py       ← automated precision/recall accuracy evaluator
├── reports/              ← example audit outputs
├── tests/                ← automated tests
└── USAGE_GUIDE.md        ← this file
```
