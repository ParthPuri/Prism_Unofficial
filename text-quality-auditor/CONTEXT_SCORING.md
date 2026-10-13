# Context-aware quality scoring (v0.2)

**The target is quality for a declared purpose, not universal dataset goodness.** The report preserves the raw evidence and separately quantifies its relevance to that purpose.

## Technique: requirements-driven multi-attribute utility scoring

The AI translates a business use case into a validated contract containing:

1. Purpose: supervised training, retrieval corpus, or log monitoring.
2. Privacy requirement: retain permitted content or require pattern redaction.
3. Applicable checks and context-specific measurement parameters.
4. For each check, a weight, a raw pass threshold, a gate flag, a utility curve and a business explanation.
5. Explicit assumptions about missing business details.

Local code computes each raw fraction `q_i`. A bounded monotone utility curve turns it into a contextual score:

```text
u_i(q_i | context) = clip((q_i - bad_i) / (good_i - bad_i), 0, 1)
Q(dataset | context) = 100 × Σ weight_i × u_i(q_i | context)
```

`bad_i` is the raw quality at which this metric has zero utility for the purpose; `good_i` is fully satisfactory quality. Intermediate values are linearly interpolated. Gates and raw pass thresholds are evaluated separately, so an average cannot erase a critical failure. A missing metric remains missing and withholds the full score.

This design changes selection, raw metric parameters, tolerances, penalties, and weights. It does more than multiply the same global scores by different weights.

Example: 98% valid content scores 80% utility with bad=90%, good=100%; it scores 96% utility with bad=50%, good=100%. Those anchors describe different tolerances; neither is an empirically justified standard until validated for its business task.

## Same dataset, three purposes

The bundled `scripts/context_demo.py` audits the exact same SMS file under three handwritten demonstration contracts. No model output is fabricated.

| Purpose | Index | Why the score differs |
|---|---:|---|
| Spam classifier training | 99.69 | Labels and split independence matter; natural repeated campaigns and phone numbers are retained |
| Message retrieval corpus | 62.15 | Normalized repetition has a strong deduplication penalty; label and train/test checks are irrelevant |
| Event log monitoring | 98.74 | Repeated events are legitimate, so uniqueness is excluded; basic content integrity remains relevant |

The retrieval result is structural suitability under an illustrative deduplication contract, not measured retrieval accuracy, relevance, factuality, or RAG answer quality. All three reports contain the same input SHA-256. Their weights and anchors are demonstration values, not calibrated production estimates.

The current purpose taxonomy is intentionally explicit. An additional domain such as legal retrieval, multilingual search or fine-tuning can be represented through the business description and optional metrics/anchors only insofar as implemented checks can measure it. New semantic requirements need registered local checks; the AI cannot invent an unsupported metric and call it measured.

## Great Expectations integration

Install the tested optional dependencies:

```bash
python3 -m pip install -e '.[gx]'
```

Tested library versions: Great Expectations 1.24.0, pandas 2.2.3.

The AI's validated policy is compiled into GX Expectation Suites and Validation Definitions. `mostly` receives the raw pass threshold; severity is critical for gates and warning otherwise. The adapter uses an ephemeral in-memory GX context and disables GX analytics. It exports safe validation counts and expectation configurations into the local report.

For text, ingestion first derives measurable features such as string type, nonempty content, control-character flags, label-vocabulary validity and character count. GX checks these features using `ExpectColumnValuesToBeInSet` and `ExpectColumnValuesToBeBetween`. No raw messages enter the GX dataframe. The actual GX unexpected count determines the raw metric score for supported checks. Feature extraction itself remains local application code and must also be tested.

Cross-row metrics—normalized uniqueness, conflicting labels, split/group leakage and minimum class support—remain local SQL aggregations. A report identifies each metric's executor. Do not describe built-in GX row expectations as an exhaustive semantic or leakage checker.

The adapter materializes one numeric feature dataframe and reuses it across checks. This adds O(rows) RAM, unlike the SQLite-only execution path. It does not sample. For very large data, use chunk-aware validation with globally aggregated counts, or a validated SQL/Spark integration; do not average chunk percentages without their denominators.

## Run the context demonstration

```bash
python3 scripts/context_demo.py
```

Reports and policies are written to `reports/context/`; comparison evidence is in `reports/context/comparison.json`. The bundled copies were produced with real GX execution.

## Ask the AI to produce a new policy

For the current hosted Hugging Face default, see `LLM_SETUP.md`. The command below explicitly selects a local installed model.

```bash
python3 -m textaudit.cli data/sms_spam/dataset.csv \
  --label-column label --split-column split \
  --allowed-labels data/sms_spam/labels.json \
  --use-case "Audit these messages for training a spam classifier. Recurring campaign messages and phone numbers are legitimate. Prioritize trustworthy labels, sufficient minority-class support and independent evaluation splits. Explain all inferred assumptions." \
  --purpose supervised_training \
  --provider local --model YOUR_INSTALLED_LOCAL_MODEL \
  --backend gx --out reports/local/context-ai
```

Omit `--purpose` if the AI should infer the purpose. Retain it when the task is known and must not be reinterpreted. It is a scope constraint, not manual assignment of weights or thresholds. `--require-redacted` is a trusted privacy constraint that the model cannot waive. Otherwise, privacy can be inferred from the business context and must be reviewed.

The AI receives an aggregate profile and a per-context catalog of supported metrics and mandatory gates. It produces version-2 JSON, which is locally validated before compilation. One repair request is allowed. Saved/cached policies make subsequent scoring reproducible; raw messages and arbitrary executable code are never passed to the planner.

The actual local-model inference path has not been tested against an installed LLM here. Mock-server tests cover the version-2 output contract, repair and caching. GX execution is real and tested.

## Interpreting the report

Each contextual report shows the interpreted purpose, privacy requirement, assumptions, raw metric values, utility values, weights, raw pass thresholds, gate outcomes, and business rationale. `report.json` additionally includes utility anchors, weighted points, source/policy hashes, metric executors and compiled GX configurations.

Changing a dataset's context changes the ruler. Scores from different contexts are not directly comparable as universal goodness rankings. For progress over time, hold both the context and its approved policy fixed.

Calibration is still required: evaluate on documented business contexts, inject controlled defects, measure downstream development-set performance or business loss, calibrate utility anchors and nonnegative weights, and keep a final test set untouched. For retrieval, add a held-out query/relevance evaluation before claiming retrieval quality; for classifiers, use per-class metrics and task-specific error costs.

For larger organizations, a further extension is retrieval over approved company quality contracts and metric recipes: retrieve task-relevant requirements, have the AI cite their source/version when drafting the policy, then validate and execute it locally. This would be policy RAG. This release supplies a bounded metric catalog; it does not yet implement a company-document retrieval index.

## Primary documentation

- GX Expectation creation, `mostly`, metadata and severity: https://docs.greatexpectations.io/docs/core/define_expectations/create_an_expectation/
- GX dataframe Data Sources, Assets, Batch Definitions and Validation Definitions: https://docs.greatexpectations.io/docs/core/connect_to_data/dataframes/
- GX result fields and format controls: https://docs.greatexpectations.io/docs/core/trigger_actions_based_on_results/choose_a_result_format/
