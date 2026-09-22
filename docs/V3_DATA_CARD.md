# RetinaSathi V3 Initial Data Audit

Generated: 2026-09-19T19:04:28.741277+00:00

Raw images were read only. No image was edited or removed by this audit.

## Sources

| Source | Records | Groups | Grades 0/1/2/3/4 | Validation |
|---|---:|---:|---|---|
| APTOS_2019 | 3662 | 3531 | 1805/370/999/193/295 | valid: 3662 |
| DeepDRiD | 1600 | 400 | 714/186/326/282/92 | valid: 1600 |
| IDRiD | 516 | 510 | 168/25/168/93/62 | valid: 516 |

## Split policy

- APTOS 2019 labelled training images are development data; patient identifiers are unavailable.
- IDRiD official training is development data; its consumed official test is historical evidence only.
- DeepDRiD official training enters the development pool with patient grouping.
- DeepDRiD official validation remains a source-validation partition.
- Messidor is excluded from V3 selection because its V2 result has already been inspected.
- A new untouched external source or prospective Indian cohort is still required before final model promotion.

## Duplicate audit

- Exact duplicate groups: 129
- Near-duplicate candidates requiring review: 31
- Visually confirmed near-duplicate pairs: 3
- Visually rejected lookalike pairs: 29
- Confirmed duplicate clusters are quarantined in the manifest; raw images are retained.

## Training eligibility

- development_pool: 5095
- duplicate_excluded: 98
- historical_test_excluded_from_v3_selection: 102
- quarantine_confirmed_near_duplicate_cluster: 4
- quarantine_exact_duplicate_label_conflict: 68
- quarantine_label_alignment: 11
- source_validation: 400

Eligible development grade counts (0/1/2/3/4): 2461/495/1287/462/390

Eligible referable counts: 0: 2956, 1: 2139

## Blocking conditions

- Missing or invalid labelled files: 0
- EyePACS remains deferred until extraction, patient/eye grouping, licence confirmation, and duplicate audit are complete.
- Final Indian clinical validation remains unavailable.
