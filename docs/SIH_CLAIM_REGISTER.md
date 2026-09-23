# SIH claim register

Use the exact safe wording below in the PPT, video and demo.

| Topic | Evidence class | Safe claim | Do not claim |
|---|---|---|---|
| Product | Freshly verified | “RetinaSathi is a research screening-support prototype with human review.” | “It is a diagnostic device” or “clinically approved” |
| Cloud | Freshly verified | “The current site uses an authenticated InsForge function to reach a protected Azure Container Apps V3.4 service.” | “All patient data remains on device” |
| Model | Freshly verified identity; historical evaluation | “The deployed candidate is partial DINOv2 ViT-S/14, version `classifier-v3.4-seed26038`.” | “EfficientNet-B3 is the deployed V3.4 model” |
| Referral performance | Historical verified result | “On a reused, patient-separated 400-image DeepDRiD source holdout, the selected seed measured 92.22% sensitivity and 91.36% specificity.” | “92.22% five-grade accuracy” or “prospective clinical accuracy” |
| Three seeds | Historical verified result | “Three fine-tuning seeds from the same V3.2 initialization averaged 93.89% sensitivity and 90.61% specificity.” | “Three fully independent model initializations” |
| Five grades | Historical verified result | “The model outputs grades 0–4, but exact grading is weaker than referral; selected-seed macro-F1 was 0.5339.” | “Reliable autonomous five-grade diagnosis” |
| DeepDRiD | Historical verified result | “DeepDRiD supplied a patient-separated source-validation partition; the source also contributed development images.” | “Untouched independent external clinical test” |
| XAI | Fresh code status and historical failure | “V3.4 explanation is disabled because the prior map failed technical validation.” | “Clinically validated heatmap” or “lesion localization” |
| DME/lesions/NV | Fresh runtime status | “These modules are unavailable in the deployed V3.4 path.” | “DME, hemorrhage, exudate or neovascularization detection is implemented” |
| Quality/OOD | Fresh engineering tests | “A conservative heuristic rejects known poor and synthetic controls; it is not a validated OOD detector.” | “It recognizes every non-retinal image” |
| MATLAB | Historical verified and locally reproducible | “MATLAB matched 25 stored V3.4 parity cases and passed 8 engineering tests.” | “MATLAB parity proves medical accuracy” |
| SimEvents | Historical verified simulation | “SimEvents models queues and capacity under declared assumptions.” | “127,000 patients were screened” |
| Distillation | Future work | “Distillation may be evaluated after safety and accuracy validation.” | “V3.4 was trained by knowledge distillation” |
| Clinical use | Future work | “Independent intended-camera and prospective ophthalmologist evaluation are next.” | “Hospital validated,” “doctor approved,” or “ready for patient care” |

Evidence must remain attached to the same model hash, threshold and code revision. If those change, re-run the relevant checks before reusing the claim.
