# Prism — Data Quality Assessment for AI Datasets

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-iteration%201%20active-success.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

> **Early, evidence-backed dataset quality firewall for AI workflows.**
> Assesses datasets for completeness, validity, uniqueness, consistency, accuracy, and leakage *before* model training begins.

---

## 📌 Problem & Motivation

AI model performance and stability depend critically on training data quality. Poor or corrupted data causes:
- Weak accuracy and model hallucination
- Subtle biases and evaluation data leakage
- Wasted GPU training cycles and expensive rework

**Prism** provides an early automated quality-assessment module that:
1. Translates business intent into a tailored quality policy using AI planning.
2. Runs deterministic local quality audits across structural dimensions.
3. Computes multi-attribute utility scorecards with blocking gates.
4. Generates audit reports (JSON & Markdown) and provides an interactive web dashboard for human review and governance.

---

## 🗂️ Project Structure

```
Prism_Unofficial/
├── README.md                      ← Project overview and getting started (this file)
├── .gitignore                     ← Repository gitignore
└── text-quality-auditor/          ← Text modality quality audit engine & UI
    ├── dashboard.py               ← Interactive Streamlit web dashboard
    ├── pyproject.toml             ← Package definition and dependencies
    ├── README.md                  ← Detailed text auditor engine documentation
    ├── USAGE_GUIDE.md             ← Step-by-step UI and CLI user guide
    ├── CONTEXT_SCORING.md         ← Multi-attribute utility scoring architecture
    ├── LLM_SETUP.md               ← Ollama and Hugging Face provider setup
    ├── DATA_ATTRIBUTION.md        ← Benchmark dataset provenance and licenses
    │
    ├── textaudit/                 ← Core Python package
    │   ├── core.py                ← Ingestion, 14 deterministic checks, scoring engine
    │   ├── planner.py             ← AI policy planning with schema repair & caching
    │   ├── providers.py           ← LLM provider interface (HF, Ollama, compatible)
    │   ├── policy_schema.py       ← Structured JSON schema for policies
    │   ├── gx_backend.py          ← Optional Great Expectations integration
    │   └── cli.py                 ← Command-line audit runner
    │
    ├── data/                      ← Bundled benchmark datasets
    │   ├── banking77/             ← 13,083 banking intent classification messages
    │   └── sms_spam/              ← 5,574 SMS binary classification messages
    │
    ├── policies/                  ← Reference and demo policy contracts
    │   ├── classification_demo.json
    │   └── banking_context_demo.json
    │
    ├── reports/                   ← Audit evidence and generated reports
    │   ├── banking77/
    │   ├── context/
    │   └── sms_spam/
    │
    ├── scripts/                   ← Utility scripts
    │   ├── context_demo.py        ← Same data, 3 purposes demonstration
    │   └── prepare_datasets.py    ← Publisher download and hash verification
    │
    └── tests/                     ← Automated test suite (36 unit tests)
```

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install the auditor in editable mode along with dashboard dependencies:

```bash
git clone https://github.com/ParthPuri/Prism_Unofficial.git
cd Prism_Unofficial/text-quality-auditor

# Install dashboard libraries
pip install streamlit plotly

# Install the text-quality-auditor engine
pip install -e .
```

### 2. Launch the Interactive Dashboard

```bash
streamlit run dashboard.py
```

The web dashboard opens at `http://localhost:8501`:
1. **Choose a Dataset**: Pick bundled **BANKING77** or **SMS Spam**, or upload your own CSV.
2. **Define Business Context**: Enter your use case description (or leave blank to use the offline demo policy).
3. **Run Audit**: View the real-time quality index gauge, per-metric pass/fail breakdown, weight distribution, and expandable defect findings with affected row IDs.

---

## 🧪 Core Quality Dimensions & Checks

Prism implements 14 deterministic checks across fundamental data quality dimensions:

| Dimension | Checks | What it Verifies |
|---|---|---|
| **Completeness** | `text_completeness`, `label_completeness`, `split_completeness`, `group_completeness` | Verifies non-empty text, presence of target labels, valid train/val/test splits, and group identifiers. |
| **Validity** | `text_type`, `encoding_cleanliness`, `length_validity`, `label_validity` | Flags invalid data types, replacement characters (``), unexpected ASCII controls, out-of-bound string lengths, and out-of-vocabulary labels. |
| **Uniqueness & Cardinality** | `normalized_uniqueness`, `class_support` | Computes NFKC/casefold normalized text uniqueness and verifies sufficient minimum examples per training class. |
| **Consistency & Leakage** | `label_consistency`, `split_disjointness`, `group_disjointness` | Flags contradictory labels on identical text, cross-split train/test data leakage, and group contamination across evaluation folds. |
| **Privacy & Compliance** | `privacy_pattern_clearance` | Scans for unredacted phone numbers and email addresses when anonymization is required. |

> 🔒 **Mandatory Blocking Gates**: Critical integrity checks (such as split leakage or missing mandatory labels) act as gates. A dataset failing a mandatory gate is marked **`BLOCKED`**, regardless of its numerical score.

---

## 💻 Command Line Interface (CLI)

Run an immediate audit without launching the web server:

```bash
python -m textaudit.cli data/banking77/dataset.csv \
  --label-column label --split-column split \
  --allowed-labels data/banking77/labels.json \
  --policy policies/classification_demo.json \
  --out reports/local/banking77
```

### Exit Codes:
- `0`: All checks passed (ready for human review).
- `2`: Audit completed but dataset is `BLOCKED`, `INCOMPLETE`, or `REVIEW_REQUIRED`.
- `1`: Execution or validation error.

---

## 🤖 AI Policy Planning (LLM Integration)

Instead of relying on hardcoded static rules, Prism uses AI to tailor the quality policy to the target business task:

- **Local Ollama (Free / Recommended)**:
  ```bash
  ollama pull qwen3:8b
  ollama serve
  ```
- **Hugging Face Inference Providers**:
  Set your token (`export HF_TOKEN=...`) and select `--provider huggingface`.
- **Zero Raw Data Transmission**: The AI planner only receives an aggregate structural profile (row counts, class counts, split ratios) and the business description. Raw text, labels, and customer data never leave your machine.

---

## 🧪 Running Tests

Verify the engine with the test suite:

```bash
python -m unittest discover -s tests -v
```

All 36 tests validate gate enforcement, privacy rules, weight limits, leakage detection, provider security, and caching.

---

## 🗺️ Roadmap Alignment

- [x] **Milestone 1 (Month 2)**: Core quality dimensions (completeness, validity, uniqueness), deterministic scoring engine, baseline datasets.
- [x] **Milestone 2 (Month 3)**: Consistency & leakage checks, multi-attribute contextual utility curves, interactive Streamlit upload-and-assess UI.
- [ ] **Milestone 3 & Closure (Month 4)**: Multimodal expansion (Image, Speech, Video data quality auditors), consolidated governance scorecard.
