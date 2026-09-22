# RetinaSathi V3.4 three-seed stability report

Date: 21 September 2026

## Decision

Freeze the V3.4 architecture and training recipe as the current RetinaSathi candidate. All three independently initialized runs passed the predeclared calibration gate and the DeepDRiD source gate. The referral-screening metrics are stable enough to stop changing this model before the locked test.

Do not describe the five-grade classifier as clinically solved. Grade 2 and Grade 3 vary meaningfully between seeds, and Grade 4 recall remains low. The model is strongest as a referable-DR screening candidate that requires human review.

The official locked test has not been opened or used.

## V3.4 source-validation results

| Seed | Selected epoch | Sensitivity | Specificity | AUROC | AUPRC | QWK | Macro-F1 | Gate |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 26038 | 5 | 92.22% | 91.36% | 0.9756 | 0.9745 | 0.7910 | 0.5339 | Pass |
| 26039 | 10 | 95.00% | 89.55% | 0.9779 | 0.9758 | 0.8635 | 0.6017 | Pass |
| 26040 | 11 | 94.44% | 90.91% | 0.9807 | 0.9787 | 0.8195 | 0.5641 | Pass |

## Mean, range, and worst seed

| Metric | Mean | Minimum–maximum | Spread | Worst seed value |
|---|---:|---:|---:|---:|
| Sensitivity | 93.89% | 92.22–95.00% | 2.78 points | 92.22% |
| Specificity | 90.61% | 89.55–91.36% | 1.82 points | 89.55% |
| AUROC | 0.9781 | 0.9756–0.9807 | 0.0051 | 0.9756 |
| AUPRC | 0.9763 | 0.9745–0.9787 | 0.0043 | 0.9745 |
| QWK | 0.8247 | 0.7910–0.8635 | 0.0725 | 0.7910 |
| Macro-F1 | 0.5666 | 0.5339–0.6017 | 0.0678 | 0.5339 |

Referral discrimination is very stable: AUROC and AUPRC move by about 0.005 or less, while sensitivity and specificity remain above the source gate in every run. Exact five-grade predictions are less stable.

## Grade recall across the three seeds

| Grade | Meaning | Mean recall | Minimum–maximum | Spread | Worst seed recall |
|---:|---|---:|---:|---:|---:|
| 0 | No DR | 84.67% | 82.76–87.36% | 4.60 points | 82.76% |
| 1 | Mild NPDR | 53.62% | 52.17–54.35% | 2.17 points | 52.17% |
| 2 | Moderate NPDR | 51.45% | 39.13–60.87% | 21.74 points | 39.13% |
| 3 | Severe NPDR | 57.35% | 52.94–64.71% | 11.76 points | 52.94% |
| 4 | Proliferative DR | 43.33% | 40.00–45.00% | 5.00 points | 40.00% |

Grade 2 is the main stability problem. Grade 4 is consistently weak, partly because the source-validation set contains only 20 Grade 4 images. More training alone cannot prove clinical performance for these grades; stronger and independently collected data are required.

## Comparison with V3.1–V3.3

| Model | Sensitivity | Specificity | AUROC | AUPRC | QWK | Macro-F1 | Outcome |
|---|---:|---:|---:|---:|---:|---:|---|
| V3.1 EfficientNet-B3 | 97.22% | 77.73% | 0.9663 | 0.9631 | 0.8340 | 0.5341 | Failed specificity gate |
| V3.2 frozen DINOv2-S | 87.22% | 87.27% | 0.9412 | 0.9433 | 0.7440 | 0.4220 | Failed calibration/sensitivity |
| V3.3 frozen RETFound | 58.89% | 88.64% | 0.8584 | 0.8337 | 0.1412 | 0.2010 | Failed badly |
| V3.4 partial DINOv2-S, three-seed mean | 93.89% | 90.61% | 0.9781 | 0.9763 | 0.8247 | 0.5666 | Passed all three repeats |

V3.4 gives the best balance. V3.1 detects more referable cases but produces too many false positives. V3.2 does not adapt enough to the retinal data. V3.3 underfits because its large retinal encoder was frozen with a very small trainable head. V3.4 adapts the final two DINOv2 blocks and keeps both sensitivity and specificity above the gate.

## What to do next

1. Freeze the V3.4 code, data manifest, preprocessing, thresholds, and checkpoint hashes. Do not tune them again using DeepDRiD.
2. Keep the official locked test closed until the freeze record is committed and verified. Then run it once and report every result, including failures.
3. Present V3.4 as a research screening-support model for referable DR. Keep human review and uncertainty routing in the workflow.
4. Improve five-grade performance in the next research branch using new independent Indian-camera data, especially confirmed Grades 2–4. Do not change the frozen candidate after seeing locked-test labels.
5. If deployment speed matters, distill the frozen V3.4 teacher into a smaller student only after the locked evaluation. Validate the student separately.

Machine-readable results are stored in `runs/v3_4_three_seed_stability.json`. Individual audits are stored in each V3.4 run directory as `candidate_audit.json`.
