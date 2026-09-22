# RetinaSathi V3.4 partial DINOv2-S results

Date: 2026-09-21

## Decision

V3.4 seed 26038 is the first V3 candidate to pass both the predeclared calibration gate and the independent-source screening gate. It qualifies for adaptation-stability repeats with seeds 26039 and 26040. The locked official test remains unused and must stay closed until the repeat-seed decision is complete.

## Design

- Starting checkpoint: verified V3.2 DINOv2-S/14 seed-26038 checkpoint.
- Adaptation: final two transformer blocks, final normalization, and all three RetinaSathi heads trainable.
- Input: 392 × 392.
- Differential learning rates: 1e-5 backbone, 1e-4 heads.
- Effective batch size: 16.
- Selected epoch: 5.
- Checkpoint SHA-256: `0407cbf0402e9e791307045e8d42b38ad188bec19b5ddc58f454d14f3f677f39`.

## Calibration gate

| Metric | Requirement | Result |
|---|---:|---:|
| Sensitivity | at least 95% | 95.16% |
| Specificity | at least 85% | 89.42% |

Selected referable threshold: 0.207326.

## Independent DeepDRiD source validation

| Metric | V3.4 |
|---|---:|
| Sensitivity | 92.22% |
| Specificity | 91.36% |
| AUROC | 0.9756 |
| AUPRC | 0.9745 |
| QWK | 0.7910 |
| Macro-F1 | 0.5339 |

Grade recalls:

- Grade 0: 87.36% (152/174)
- Grade 1: 54.35% (25/46)
- Grade 2: 39.13% (36/92)
- Grade 3: 52.94% (36/68)
- Grade 4: 40.00% (8/20)

## Direct comparison

| Model | Sensitivity | Specificity | QWK | Macro-F1 | Gate |
|---|---:|---:|---:|---:|---|
| V3.1 EfficientNet-B3 | 97.22% | 77.73% | 0.8340 | 0.5341 | Failed specificity |
| V3.2 frozen DINOv2-S | 87.22% | 87.27% | 0.7440 | 0.4220 | Failed calibration/source sensitivity |
| V3.3 frozen RETFound | 58.89% | 88.64% | 0.1412 | 0.2010 | Failed |
| V3.4 partial DINOv2-S | 92.22% | 91.36% | 0.7910 | 0.5339 | Passed |

V3.4 improves V3.2 through partial domain adaptation. Compared with V3.1, it trades 5 percentage points of sensitivity for 13.6 percentage points of specificity while retaining nearly identical macro-F1. Grade 2 remains the weakest grade and must be reported honestly.

## Next actions

1. Repeat the identical V3.4 adaptation with seeds 26039 and 26040.
2. Report mean, range, and worst-seed metrics.
3. Do not alter the threshold rule or use source-validation labels for training.
4. If at least two repeats pass and the worst seed remains useful, freeze the V3.4 candidate package.
5. Before any clinical or Indian-generalization claim, evaluate on a new compatible untouched source or governed Indian cohort.
6. Use DINOv3-S as the next challenger only after V3.4 seed stability is known.
