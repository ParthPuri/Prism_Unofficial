# Text Dataset Quality Report

Decision: **REVIEW_REQUIRED**

Quality index: **62.15/100**
Evaluated weight: 100.0%

The index describes this policy's checks. Final fitness requires human review and downstream validation.

| Metric | Raw / 100 | Context utility / 100 | Weight | Raw pass minimum / 100 | Gate | Status |
|---|---:|---:|---:|---:|---|---|
| text_type | 100.00 | 100.00 | 0.050 | 100.00 | True | PASS |
| text_completeness | 100.00 | 100.00 | 0.150 | 99.00 | True | PASS |
| normalized_uniqueness | 92.55 | 25.55 | 0.500 | 99.00 | False | FAIL |
| encoding_cleanliness | 99.37 | 93.72 | 0.100 | 99.00 | False | PASS |
| length_validity | 100.00 | 100.00 | 0.200 | 99.00 | False | PASS |

## Interpreted business context

Purpose: **retrieval_corpus**; privacy: **retain**.

Structural suitability as a message retrieval corpus. Redundant normalized entries waste retrieval/storage capacity; relevance and factuality are not assessed.

Assumptions:

- Public SMS content is permitted for this example.
- Only implemented structural quality dimensions are assessed.

## Blocking checks

None

## Policy rationale

Explicit demonstration of contextual utility. Anchors and weights are illustrative, not empirically calibrated or AI-generated.

## Findings and local evidence

- **text_type**: Fraction of rows with a string text value. Row IDs: []. Structural suitability as a message retrieval corpus. Redundant normalized entries waste retrieval/storage capacity; relevance and factuality are not assessed.
- **text_completeness**: Fraction of rows with nonempty string text after whitespace stripping. Row IDs: []. Structural suitability as a message retrieval corpus. Redundant normalized entries waste retrieval/storage capacity; relevance and factuality are not assessed.
- **normalized_uniqueness**: Normalized repetition is a signal; recurring messages can be legitimate. Row IDs: []. Structural suitability as a message retrieval corpus. Redundant normalized entries waste retrieval/storage capacity; relevance and factuality are not assessed.
- **encoding_cleanliness**: Fraction of nonempty texts without replacement characters or unexpected controls; heuristic, not proof of correct encoding. Row IDs: [19, 458, 794, 821, 833, 842, 1045, 1050, 1267, 1725]. Structural suitability as a message retrieval corpus. Redundant normalized entries waste retrieval/storage capacity; relevance and factuality are not assessed.
- **length_validity**: Fraction of nonempty texts within policy character limits; not model token limits. Row IDs: []. Structural suitability as a message retrieval corpus. Redundant normalized entries waste retrieval/storage capacity; relevance and factuality are not assessed.

## Limits

- Provisional policy score, not a calibrated probability of fitness or a production certification.
- Cannot prove semantic label correctness, fairness, deployment representativeness, or future drift.
- Text/group overlap is not exhaustive leakage detection; no embeddings or temporal checks in v0.1.
- No raw texts or label names were sent to the planner. Business use case text is sent as entered.
- Full scan of eligible rows. Missing eligibility is shown explicitly; no sampling confidence claimed.
- AI-interpreted context must be reviewed. Use --purpose to constrain task scope. Utility anchors are provisional and require downstream calibration.

## Reproducibility

Input SHA-256: `b5680afc8b1c19cd7bc9aed7cac8d256d707820af326f7886a940fb40b3f2973`
Policy SHA-256: `e0b7f3977d0061cf921c05d3be12ad19a578bee5865ae12d1ad44a29d244651c`
Engine: `0.2.0`
Planner source: `explicit_context_demo`
