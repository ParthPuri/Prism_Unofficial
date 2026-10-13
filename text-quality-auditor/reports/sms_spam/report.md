# Text Dataset Quality Report

Decision: **REVIEW_REQUIRED**

Quality index: **99.46/100**
Evaluated weight: 100.0%

The index describes this policy's checks. Final fitness requires human review and downstream validation.

| Metric | Score / 100 | Weight | Pass minimum / 100 | Gate | Status |
|---|---:|---:|---:|---|---|
| text_type | 100.00 | 0.030 | 100.00 | True | PASS |
| text_completeness | 100.00 | 0.140 | 99.00 | True | PASS |
| encoding_cleanliness | 99.37 | 0.030 | 100.00 | False | FAIL |
| normalized_uniqueness | 92.55 | 0.070 | 85.00 | False | PASS |
| length_validity | 100.00 | 0.030 | 99.00 | False | PASS |
| label_completeness | 100.00 | 0.130 | 99.00 | True | PASS |
| label_consistency | 100.00 | 0.150 | 100.00 | True | PASS |
| label_validity | 100.00 | 0.080 | 100.00 | True | PASS |
| split_completeness | 100.00 | 0.040 | 100.00 | True | PASS |
| split_disjointness | 100.00 | 0.250 | 100.00 | True | PASS |
| class_support | 100.00 | 0.050 | 100.00 | False | PASS |

## Blocking checks

None

## Policy rationale

Handwritten demonstration policy, not AI-generated or empirically calibrated. Highlights completeness, label consistency and split independence for text classification. Duplicate rates and class support require business review.

## Findings and local evidence

- **text_type**: Full scan of eligible rows. Row IDs: []. Illustrative conservative classification policy; validate against downstream performance before production.
- **text_completeness**: Full scan of eligible rows. Row IDs: []. Illustrative conservative classification policy; validate against downstream performance before production.
- **encoding_cleanliness**: Full scan of eligible rows. Row IDs: [19, 458, 794, 821, 833, 842, 1045, 1050, 1267, 1725]. Illustrative conservative classification policy; validate against downstream performance before production.
- **normalized_uniqueness**: Normalized repetition is a signal; recurring messages can be legitimate. Row IDs: []. Illustrative conservative classification policy; validate against downstream performance before production.
- **length_validity**: Full scan of eligible rows. Row IDs: []. Illustrative conservative classification policy; validate against downstream performance before production.
- **label_completeness**: Full scan of eligible rows. Row IDs: []. Illustrative conservative classification policy; validate against downstream performance before production.
- **label_consistency**: Conflicting labels require review; context-dependent ambiguity is possible. Row IDs: []. Illustrative conservative classification policy; validate against downstream performance before production.
- **label_validity**: Full scan of eligible rows. Row IDs: []. Illustrative conservative classification policy; validate against downstream performance before production.
- **split_completeness**: Full scan of eligible rows. Row IDs: []. Illustrative conservative classification policy; validate against downstream performance before production.
- **split_disjointness**: Counts evaluation rows shared across recognized splits; normalization-based overlap, not semantic leakage. Row IDs: []. Illustrative conservative classification policy; validate against downstream performance before production.
- **class_support**: Support adequacy against a provisional target; not semantic label accuracy or representativeness. Row IDs: []. Illustrative conservative classification policy; validate against downstream performance before production.

## Limits

- Provisional policy score, not a calibrated probability of fitness or a production certification.
- Cannot prove semantic label correctness, fairness, deployment representativeness, or future drift.
- Text/group overlap is not exhaustive leakage detection; no embeddings or temporal checks in v0.1.
- No raw texts or label names were sent to the planner. Business use case text is sent as entered.
- Full scan of eligible rows. Missing eligibility is shown explicitly; no sampling confidence claimed.

## Reproducibility

Input SHA-256: `b5680afc8b1c19cd7bc9aed7cac8d256d707820af326f7886a940fb40b3f2973`
Policy SHA-256: `48be18daa077fab759da534a88906f18a9218e60b606bffae0b9be1f6c605f3e`
Engine: `0.1.0`
Planner source: `explicit_policy_file`
