# Text Dataset Quality Report

Decision: **BLOCKED**

Quality index: **94.32/100**
Evaluated weight: 100.0%

The index describes this policy's checks. Final fitness requires human review and downstream validation.

| Metric | Raw / 100 | Context utility / 100 | Weight | Raw pass minimum / 100 | Gate | Status |
|---|---:|---:|---:|---:|---|---|
| text_type | 100.00 | 100.00 | 0.030 | 100.00 | True | PASS |
| text_completeness | 100.00 | 100.00 | 0.120 | 99.00 | True | PASS |
| label_completeness | 100.00 | 100.00 | 0.120 | 99.00 | True | PASS |
| label_consistency | 100.00 | 100.00 | 0.150 | 100.00 | True | PASS |
| label_validity | 100.00 | 100.00 | 0.080 | 100.00 | True | PASS |
| split_completeness | 100.00 | 100.00 | 0.050 | 100.00 | True | PASS |
| split_disjointness | 99.77 | 77.27 | 0.250 | 100.00 | True | FAIL |
| encoding_cleanliness | 100.00 | 100.00 | 0.050 | 99.00 | False | PASS |
| class_support | 100.00 | 100.00 | 0.150 | 100.00 | False | PASS |

## Interpreted business context

Purpose: **supervised_training**; privacy: **retain**.

Supervised BANKING77 banking-support intent classifier: prioritize label integrity, per-class support and independent evaluation splits; preserve legitimate short customer queries.

Assumptions:

- Public SMS content is permitted for this example.
- Only implemented structural quality dimensions are assessed.

## Blocking checks

split_disjointness

## Policy rationale

Explicit demonstration of contextual utility. Anchors and weights are illustrative, not empirically calibrated or AI-generated.

## Findings and local evidence

- **text_type**: Fraction of rows with a string text value. Row IDs: []. Supervised BANKING77 banking-support intent classifier: prioritize label integrity, per-class support and independent evaluation splits; preserve legitimate short customer queries.
- **text_completeness**: Fraction of rows with nonempty string text after whitespace stripping. Row IDs: []. Supervised BANKING77 banking-support intent classifier: prioritize label integrity, per-class support and independent evaluation splits; preserve legitimate short customer queries.
- **label_completeness**: Fraction of rows with a nonempty scalar label. Row IDs: []. Supervised BANKING77 banking-support intent classifier: prioritize label integrity, per-class support and independent evaluation splits; preserve legitimate short customer queries.
- **label_consistency**: Conflicting labels require review; context-dependent ambiguity is possible. Row IDs: []. Supervised BANKING77 banking-support intent classifier: prioritize label integrity, per-class support and independent evaluation splits; preserve legitimate short customer queries.
- **label_validity**: Fraction of nonempty scalar labels in the user-supplied allowed label vocabulary. Row IDs: []. Supervised BANKING77 banking-support intent classifier: prioritize label integrity, per-class support and independent evaluation splits; preserve legitimate short customer queries.
- **split_completeness**: Fraction of rows with a recognized train/validation/test split. Row IDs: []. Supervised BANKING77 banking-support intent classifier: prioritize label integrity, per-class support and independent evaluation splits; preserve legitimate short customer queries.
- **split_disjointness**: Counts evaluation rows shared across recognized splits; normalization-based overlap, not semantic leakage. Row IDs: [10558, 10980, 10981, 11436, 11478, 12153, 13074]. Supervised BANKING77 banking-support intent classifier: prioritize label integrity, per-class support and independent evaluation splits; preserve legitimate short customer queries.
- **encoding_cleanliness**: Fraction of nonempty texts without replacement characters or unexpected controls; heuristic, not proof of correct encoding. Row IDs: []. Supervised BANKING77 banking-support intent classifier: prioritize label integrity, per-class support and independent evaluation splits; preserve legitimate short customer queries.
- **class_support**: Support adequacy against a provisional target; not semantic label accuracy or representativeness. Row IDs: []. Supervised BANKING77 banking-support intent classifier: prioritize label integrity, per-class support and independent evaluation splits; preserve legitimate short customer queries.

## Limits

- Provisional policy score, not a calibrated probability of fitness or a production certification.
- Cannot prove semantic label correctness, fairness, deployment representativeness, or future drift.
- Text/group overlap is not exhaustive leakage detection; no embeddings or temporal checks in v0.1.
- No raw texts or label names were sent to the planner. Business use case text is sent as entered.
- Full scan of eligible rows. Missing eligibility is shown explicitly; no sampling confidence claimed.
- AI-interpreted context must be reviewed. Use --purpose to constrain task scope. Utility anchors are provisional and require downstream calibration.

## Reproducibility

Input SHA-256: `7ddcf1f00b7b2bb8dddc3646116ff0a31a5c322e498fbad6ab8f8b06131567af`
Policy SHA-256: `21aba505fb195b26ba8b46f3f63e93c03aadbb1dadee9130ee96383e27d6f991`
Engine: `0.2.0`
Planner source: `explicit_policy_file`
