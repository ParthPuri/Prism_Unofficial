# Text Dataset Quality Auditor

A tool for assessing the structural quality of text datasets before they are used for machine learning. You describe what the dataset is for, the system figures out which quality checks matter for that use case, runs them locally against your data, and produces a scored report with a pass/fail verdict.

---

## What it does

Most dataset quality tools apply the same fixed set of checks to every dataset. This tool takes a different approach: the checks that run, and how much each one contributes to the final score, are determined by the business context you provide.

For example, label consistency and train/test split independence matter a lot when training a classifier, but are irrelevant for a retrieval corpus. Encoding anomalies are worth flagging in a spam dataset but may be noise in a banking support dataset. The system accounts for this distinction automatically.

The overall flow is:

```
Dataset + Business Context
        ↓
  Quality Policy
  (which checks to run, weights, thresholds)
        ↓
  Local Deterministic Checks
  (row-level scanning, no external calls)
        ↓
  Weighted Quality Index + Blocking Gates
        ↓
  JSON + Markdown Report  →  Dashboard
```

The quality index is a number from 0 to 100. Blocking gates are hard requirements — a dataset can score 99/100 and still be blocked if a mandatory gate fails. The final fitness decision always stays with the human reviewer.

---

## How the policy is determined

There are two modes:

**Demo policy (no model required)**
A handwritten policy file is provided for classification tasks. It covers the most common checks with conservative thresholds. This is what runs when you do not provide a business context. It works offline and is useful for quick checks and demonstrations.

**AI-generated policy**
When you describe your use case in plain text, a language model reads a compact profile of your dataset (row counts, split sizes, label count — never the raw text or labels) alongside the available check catalog, and selects which checks to include, what weight each gets, and what thresholds to use. The AI decision is validated before any scoring runs — invalid weights, unknown metrics, or weakened mandatory gates are rejected outright. The policy is cached so the same dataset and context always produce the same result.

The key separation: the AI decides *what to measure and how to weight it*. All actual measurement — row scanning, counting violations, computing statistics — runs locally in Python with no model involvement.

---

## Checks available

| Check | What it measures |
|---|---|
| `text_type` | All text fields are strings |
| `text_completeness` | No empty or missing text |
| `encoding_cleanliness` | No Unicode replacement characters or unexpected control characters |
| `normalized_uniqueness` | Fraction of distinct texts after NFKC normalisation and casefolding |
| `length_validity` | Text length within configured character bounds |
| `label_completeness` | No missing labels |
| `label_validity` | All labels belong to the declared vocabulary |
| `label_consistency` | Same normalised text does not appear with different labels |
| `split_completeness` | All rows have a valid split assignment |
| `split_disjointness` | No normalised text appears in both train and test |
| `group_completeness` | All rows have a group assignment (when groups are declared) |
| `group_disjointness` | No group spans multiple splits (when groups are declared) |
| `class_support` | Every class has at least the configured minimum training examples |
| `privacy_pattern_clearance` | No email or phone patterns present (when redaction is required) |

Mandatory gates — text type, text completeness, label completeness, label validity, split completeness, split disjointness, privacy clearance — cannot be turned off or weakened by the policy. A failing gate blocks the dataset regardless of the weighted score.

---

## Datasets included

Two public benchmark datasets are bundled for demonstration:

| Dataset | Records | Task |
|---|---:|---|
| BANKING77 | 13,083 (10,003 train / 3,080 test) | 77-intent banking customer support classification |
| UCI SMS Spam Collection | 5,574 | Binary spam/ham classification |

Both are downloaded from their original publishers with provenance hashes recorded in `data/*/provenance.json`. Neither dataset has been modified to improve audit scores.

Sources:
- BANKING77: https://github.com/PolyAI-LDN/task-specific-datasets (CC BY 4.0)
- SMS Spam: https://archive.ics.uci.edu/dataset/228/sms+spam+collection (CC BY 4.0)

---

## Running the dashboard

The dashboard is the recommended way to interact with the tool. It provides a visual interface for uploading datasets, entering business context, and exploring the audit results.

**Install dependencies:**

```bash
pip install streamlit plotly
pip install -e .
```

**Start the dashboard:**

```bash
streamlit run dashboard.py
```

This opens at `http://localhost:8501`.

### Dashboard walkthrough

**Step 1 — Dataset**
Choose one of the bundled examples or upload your own CSV. The CSV must have a text column and optionally label and split columns. Column names are configurable.

**Step 2 — Business context**
Describe what this dataset is for. A few sentences is enough. For example:

> "Train a 77-intent banking customer-support classifier. Evaluate generalisation on the publisher test split. Missing labels are unacceptable."

Leave this blank to run with the demo policy immediately, with no model required.

If you provide context, choose an inference provider:

- **Local Ollama (free)** — runs entirely on your machine. Install Ollama from [ollama.com](https://ollama.com), pull a model, and start the server:
  ```bash
  ollama pull qwen3:8b
  ollama serve
  ```
  Then select Local Ollama in the dashboard and enter `qwen3:8b`.

- **Hugging Face (paid)** — uses the Hugging Face Inference Providers API. Requires a token with inference credits from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). The default model is `Qwen/Qwen3-235B-A22B-Instruct-2507:nscale`.

**Step 3 — Run Audit**
Click the button. The audit completes in under a second for the bundled datasets (excluding any model inference time for AI policy generation).

### What the report shows

- **Quality index** — weighted score from 0 to 100
- **Decision** — CHECKS_PASSED_REVIEW_REQUIRED, REVIEW_REQUIRED, or BLOCKED
- **Dataset profile** — row count, class count, train/test split sizes
- **Per-metric bar chart** — every check's score plotted against its pass threshold
- **Weight distribution** — how much each check contributes to the index
- **Metric table** — score, weight, gate status, and bad row count for each check
- **Findings** — expandable cards for failing checks, including the exact row IDs
- **Policy rationale** — why this policy was chosen for this context
- **Reproducibility hashes** — SHA-256 of the input file and policy for traceability

---

## Running from the command line

Without the dashboard, run audits directly:

```bash
# Using the demo policy (no model needed)
python3 -m textaudit.cli data/banking77/dataset.csv \
  --label-column label --split-column split \
  --allowed-labels data/banking77/labels.json \
  --policy policies/classification_demo.json \
  --out reports/local/banking77

# Using a local Ollama model to generate the policy
python3 -m textaudit.cli data/banking77/dataset.csv \
  --label-column label --split-column split \
  --allowed-labels data/banking77/labels.json \
  --use-case "Train a 77-intent banking support classifier. Missing labels are unacceptable." \
  --provider local --model qwen3:8b \
  --base-url http://127.0.0.1:11434/v1 \
  --out reports/local/ai-banking77
```

Exit code `2` means the audit ran and the dataset was blocked or flagged for review. Exit code `1` means an execution error. Exit code `0` means all configured checks passed (human review still required).

Reports are written to the `--out` directory as `report.json`, `report.md`, and `policy.json`.

---

## Running tests

```bash
python3 -m unittest discover -s tests -v
```

36 tests covering gates, scoring, invalid policies, missing evidence, group isolation, privacy scope, and planner caching. 3 tests are skipped when the optional Great Expectations dependency is not installed.

---

## Input format

CSV with a header row:

```csv
text,label,split
My card has not arrived,card_arrival,train
How do I cancel a transfer,cancel_transfer,test
```

JSONL is also supported (one object per line). The text column must contain strings. Split values must be `train`, `validation`, or `test`. The label vocabulary is supplied as an external JSON array — the dataset alone cannot reveal classes that are completely absent from it.

---

## What the score means and what it does not

The quality index is a weighted sum of individual metric scores. Each metric score is a fraction of good rows over eligible rows for that check. The index reflects what was measured by the selected policy — it is not a probability that the data is good, and it does not account for semantic label correctness, deployment representativeness, or model behaviour.

A high score with a blocked gate still fails. A passing score still requires human review. No exit code certifies fitness for production use.

---

## Project structure

```
textaudit/
  core.py          — ingestion, metric registry, policy validation, scoring
  planner.py       — AI policy generation, caching, repair
  cli.py           — command-line interface and report writing
  providers.py     — inference provider configuration
  policy_schema.py — JSON schema for structured model output
  gx_backend.py    — optional Great Expectations integration

dashboard.py       — Streamlit web dashboard

data/
  banking77/       — BANKING77 dataset and provenance
  sms_spam/        — SMS Spam dataset and provenance

policies/
  classification_demo.json   — handwritten demo policy for classification tasks

reports/           — example audit outputs from bundled datasets

tests/             — unit and integration tests

scripts/
  prepare_datasets.py   — re-download datasets from publishers
  context_demo.py       — context-aware scoring demonstration
```

---

## Re-downloading the datasets

```bash
python3 scripts/prepare_datasets.py --dataset all
```

The script reuses existing source files. To fetch fresh copies, remove the corresponding `data/<name>/raw/` folder first.
