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
pip install -e .
```

This installs the dashboard and the auditor engine. It only needs to be done once.

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

**Upload my own CSV**
Click the upload button and select your CSV file.

Your CSV must have at minimum a text column. Optionally it can have a label column and a split column (`train` / `test` / `validation`). Example:

```csv
text,label,split
My card has not arrived,card_arrival,train
How do I cancel a transfer,cancel_transfer,test
```

After selecting or uploading a dataset, set the column names:
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

**Option A — Local Ollama (free, recommended)**

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

**Option B — Hugging Face (requires paid credits)**

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

## File structure

```
text-quality-auditor/
├── dashboard.py          ← the web dashboard (start here)
├── textaudit/            ← auditor engine
├── data/
│   ├── banking77/        ← BANKING77 dataset
│   └── sms_spam/         ← SMS Spam dataset
├── policies/
│   └── classification_demo.json   ← demo policy (no model needed)
├── reports/              ← example audit outputs
├── tests/                ← automated tests
└── USAGE_GUIDE.md        ← this file
```
