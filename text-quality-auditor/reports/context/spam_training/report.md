# Text Dataset Quality Report

Decision: **CHECKS_PASSED_REVIEW_REQUIRED**

Quality index: **99.69/100**
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
| split_disjointness | 100.00 | 100.00 | 0.250 | 100.00 | True | PASS |
| encoding_cleanliness | 99.37 | 93.72 | 0.050 | 99.00 | False | PASS |
| class_support | 100.00 | 100.00 | 0.150 | 100.00 | False | PASS |

## Interpreted business context

Purpose: **supervised_training**; privacy: **retain**.

Supervised SMS spam classifier. Preserve legitimate repeated campaigns and phone numbers; require independent train/evaluation splits.

Assumptions:

- Public SMS content is permitted for this example.
- Only implemented structural quality dimensions are assessed.

## Blocking checks

None

## Policy rationale

Explicit demonstration of contextual utility. Anchors and weights are illustrative, not empirically calibrated or AI-generated.

## Findings and local evidence

- **text_type**: Fraction of rows with a string text value. Row IDs: []. Supervised SMS spam classifier. Preserve legitimate repeated campaigns and phone numbers; require independent train/evaluation splits.
- **text_completeness**: Fraction of rows with nonempty string text after whitespace stripping. Row IDs: []. Supervised SMS spam classifier. Preserve legitimate repeated campaigns and phone numbers; require independent train/evaluation splits.
- **label_completeness**: Fraction of rows with a nonempty scalar label. Row IDs: []. Supervised SMS spam classifier. Preserve legitimate repeated campaigns and phone numbers; require independent train/evaluation splits.
- **label_consistency**: Conflicting labels require review; context-dependent ambiguity is possible. Row IDs: []. Supervised SMS spam classifier. Preserve legitimate repeated campaigns and phone numbers; require independent train/evaluation splits.
- **label_validity**: Fraction of nonempty scalar labels in the user-supplied allowed label vocabulary. Row IDs: []. Supervised SMS spam classifier. Preserve legitimate repeated campaigns and phone numbers; require independent train/evaluation splits.
- **split_completeness**: Fraction of rows with a recognized train/validation/test split. Row IDs: []. Supervised SMS spam classifier. Preserve legitimate repeated campaigns and phone numbers; require independent train/evaluation splits.
- **split_disjointness**: Counts evaluation rows shared across recognized splits; normalization-based overlap, not semantic leakage. Row IDs: []. Supervised SMS spam classifier. Preserve legitimate repeated campaigns and phone numbers; require independent train/evaluation splits.
- **encoding_cleanliness**: Fraction of nonempty texts without replacement characters or unexpected controls; heuristic, not proof of correct encoding. Row IDs: [19, 458, 794, 821, 833, 842, 1045, 1050, 1267, 1725]. Supervised SMS spam classifier. Preserve legitimate repeated campaigns and phone numbers; require independent train/evaluation splits.
- **class_support**: Support adequacy against a provisional target; not semantic label accuracy or representativeness. Row IDs: []. Supervised SMS spam classifier. Preserve legitimate repeated campaigns and phone numbers; require independent train/evaluation splits.

## Limits

- Provisional policy score, not a calibrated probability of fitness or a production certification.
- Cannot prove semantic label correctness, fairness, deployment representativeness, or future drift.
- Text/group overlap is not exhaustive leakage detection; no embeddings or temporal checks in v0.1.
- No raw texts or label names were sent to the planner. Business use case text is sent as entered.
- Full scan of eligible rows. Missing eligibility is shown explicitly; no sampling confidence claimed.
- AI-interpreted context must be reviewed. Use --purpose to constrain task scope. Utility anchors are provisional and require downstream calibration.

## Reproducibility

Input SHA-256: `b5680afc8b1c19cd7bc9aed7cac8d256d707820af326f7886a940fb40b3f2973`
Policy SHA-256: `748cbf0ac256aa52174d4ee03e2029bea868450ecf21f324ea8f234c6a83e672`
Engine: `0.2.0`
Planner source: `explicit_context_demo`
