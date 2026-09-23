# RetinaSathi V3.4: validation, clinical-pilot, and application integration plan

Date: 21 September 2026

## Decision in one page

V3.4 is ready for **local teammate testing and a controlled clinician-review pilot**. It is not ready for autonomous screening, diagnosis, treatment decisions, or unsupervised use with patients.

The strongest supported use is: a trained operator uploads a gradable colour fundus image; V3.4 estimates whether referable diabetic retinopathy may be present; an ophthalmologist reviews the image and the AI output; the clinician remains responsible for the decision.

The exact five-grade result must remain secondary. Across three seeds, referable-DR screening was stable, but mean recall for Grades 1–4 was only 53.62%, 51.45%, 57.35%, and 43.33%. The product should therefore lead with **referable / non-referable screening support**, then show the proposed 0–4 grade as a review aid.

## What external testing has actually been done

The V3 manifest contains:

| Role | Source | Images | How it was used |
|---|---|---:|---|
| Development pool | APTOS 2019 | 3,501 | Training, internal validation, and calibration folds |
| Development pool | IDRiD | 405 | Training, internal validation, and calibration folds |
| Development pool | DeepDRiD | 1,189 | Training, internal validation, and calibration folds |
| Source validation | DeepDRiD | 400 | Held outside the development pool and evaluated after training |
| Historical test excluded from V3 selection | IDRiD | 102 | Not used for V3 training or V3 selection; previously seen during V2 work |

The 400-image DeepDRiD source-validation set contains Grade 0/1/2/3/4 counts of 174/46/92/68/20.

This is **external-source validation** because the 400 images were kept outside V3 development. It is not a fully independent clinical validation because:

1. DeepDRiD also contributes different images to the development pool.
2. DeepDRiD source-validation results have been inspected while comparing V3.0–V3.4.
3. The images do not represent RetinaSathi's complete intended Indian rural workflow, camera mix, operators, prevalence, and referral pathway.
4. Only 20 source-validation images are Grade 4, so the Grade 4 estimate is imprecise.

Do not describe the result as “clinically validated.” Say: **“Retrospectively source-validated on 400 held-out DeepDRiD images; prospective Indian clinical validation is pending.”**

## V3.4 evidence

Three identical training repeats produced:

| Metric | Three-seed mean | Worst seed |
|---|---:|---:|
| Referable sensitivity | 93.89% | 92.22% |
| Referable specificity | 90.61% | 89.55% |
| AUROC | 0.9781 | 0.9756 |
| AUPRC | 0.9763 | 0.9745 |
| QWK | 0.8247 | 0.7910 |
| Macro-F1 | 0.5666 | 0.5339 |

These results support research screening assistance. They do not prove safety at a hospital, because real performance can change with camera type, image quality, patient population, disease prevalence, operator skill, and workflow.

## Which checkpoint is loaded in the application

The local application defaults to seed 26038:

- Checkpoint: `runs/v3_4_seed_26038_mps_progress_20260920_gpu/best.pt`
- Validation report: `runs/v3_4_seed_26038_mps_progress_20260920_gpu/final_validation.json`
- Checkpoint SHA-256: `0407cbf0402e9e791307045e8d42b38ad188bec19b5ddc58f454d14f3f677f39`
- Selected epoch: 5
- Referable threshold: 0.207326

Seed 26038 is used because it was the first predeclared V3.4 run. Choosing seed 26039 only because it scored highest on DeepDRiD would turn source-validation performance into a model-selection signal.

## Run V3.4 in the web application

From the repository root:

```bash
./scripts/run_v3_4_local.sh
```

The script starts:

- V3.4 inference at `http://127.0.0.1:8000`
- the web application at the Vite address printed in the terminal
- local-only processing mode
- the UI label `Local V3.4 candidate`

Check the model service:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model-card
```

To run another explicitly selected V3.4 checkpoint, pass both bound files:

```bash
./scripts/run_v3_4_local.sh PATH_TO_BEST_PT PATH_TO_FINAL_VALIDATION_JSON
```

The runtime refuses to load if the checkpoint hash and validation report do not match.

## What the V3.4 application now returns

1. A deterministic image-quality gate runs first.
2. The V3.4 classifier returns calibrated 0–4 grade probabilities.
3. The separate binary head returns the referable-DR score and uses the saved calibration threshold.
4. DME is reported as **not assessed**, because V3.4 has no validated DME head.
5. The application creates a gradient-times-activation patch map for the predicted grade.
6. The explanation is labelled as model influence, not a lesion map or anatomical confirmation.
7. Every result requires human review.

## Safe rollout stages

### Stage 0 — teammate and engineering test

- Use public, licensed, de-identified fundus images.
- Verify upload, quality rejection, grade probabilities, referable decision, explicit explanation-unavailable state, PDF, history, and deletion.
- Build a fixed regression set containing good, blurred, dark, overexposed, cropped, and non-fundus images.
- Record the app version, checkpoint hash, device, latency, and result for every case.

Exit condition: no crashes, no missing safeguards, and reproducible output on the same model package.

### Stage 1 — retrospective clinician review

- Use a new de-identified Indian-camera dataset that has not influenced model development.
- Freeze code, weights, preprocessing, threshold, and analysis plan before opening labels.
- Obtain reference grades from qualified ophthalmologists using a documented adjudication process.
- Report sensitivity, specificity, PPV, NPV, AUROC, AUPRC, calibration, confusion matrix, ungradable rate, and 95% confidence intervals.
- Stratify by camera, site, image quality, age group, sex where appropriate, and disease grade.
- Record false negatives, especially referable DR, severe NPDR, and proliferative DR.

Exit condition: the clinical and technical team accepts the predefined safety targets and failure analysis.

### Stage 2 — prospective silent pilot

- Run RetinaSathi in the real workflow without showing its result to the treating team.
- Compare AI output with normal clinical decisions and reference grading.
- Measure capture failures, latency, operator mistakes, missing follow-up, and subgroup performance.
- Do not allow the software to change patient care during this stage.

Exit condition: prospective evidence shows stable performance under the intended cameras, users, and clinical conditions.

### Stage 3 — supervised decision-support pilot

- Show results only to trained clinical users.
- Keep clinician confirmation mandatory before referral or routine disposition.
- Route poor-quality, low-confidence, out-of-distribution, and model-disagreement cases to manual review.
- Log model version, prediction, override, reason, and final outcome.
- Monitor false negatives, overrides, drift, downtime, and adverse events.

Exit condition: institutional ethics, information-security, clinical-governance, and applicable regulatory requirements are satisfied.

### Stage 4 — regulated operational release

- Define intended use, intended users, population, camera requirements, contraindications, and workflow.
- Complete the applicable Indian medical-device-software pathway with CDSCO and qualified regulatory advice.
- Establish quality management, risk management, software lifecycle, cybersecurity, usability, clinical evaluation, post-market monitoring, incident handling, and controlled updates.
- Release only the exact approved model package and monitor it after deployment.

## Data needed next

The highest-value next dataset is not another random internet benchmark. It is a governed, untouched Indian cohort matching the intended use:

- multiple Indian hospitals or screening camps
- at least two or three camera models, including the intended portable camera
- consecutive or prevalence-aware sampling
- both eyes linked at patient level without train/test patient leakage
- image-quality labels and reasons for ungradability
- five-grade DR reference labels with adjudication
- enough Grades 2–4 to estimate clinically important errors
- DME reference only when the required clinical examination or OCT evidence is available
- consent/waiver, de-identification, access control, retention policy, and data-use permission

Sample size must be planned from the desired confidence-interval width and expected prevalence with a statistician. A single fixed number should not be invented before the intended-use target is defined.

## Model improvements after the frozen evaluation

1. Preserve V3.4 as the frozen benchmark.
2. Collect new Indian-camera data and keep a final site/patient-level test cohort untouched.
3. Improve image-quality and out-of-distribution detection before changing the grader.
4. Target Grade 2–4 errors with verified data, not repeated oversampling of uncertain labels.
5. Compare patient-level two-eye fusion with single-eye inference.
6. Test camera-aware normalization only on development data.
7. If latency or hosting cost is high, distil V3.4 into a smaller model, then validate the student independently.
8. Build lesion segmentation only with proper pixel-level labels; never rename an attention map as lesion detection.
9. Add DME only through a separately trained and validated head with suitable ground truth.
10. Keep every challenger separate from the frozen candidate until its evaluation plan is complete.

## Compute and deployment opinion

No extra GPU is needed for teammate testing on the current Mac. The tested local CPU path loads the model and can process one image after startup. MPS can be used when available.

A GPU may help later for concurrent cloud inference, large prospective batches, retraining, or distillation. Do not rent one only to put V3.4 into the application. First measure:

- cold-start time
- CPU and MPS latency on real fundus images
- peak memory
- expected screenings per hour
- required concurrent users
- acceptable response time and monthly budget

For the SIH demo, run V3.4 locally and keep the lightweight cloud model only as a labelled backup. For a hospital pilot, deploy inside an approved private environment with encryption, access controls, audit logs, backups, and an operational support plan.

## Required wording in the product

Use:

- “Research screening support”
- “Referable DR suspected / not suspected”
- “Proposed DR grade for clinician review”
- “Model influence map; not a confirmed lesion map”
- “DME not assessed by this model”
- “Human review required”

Do not use:

- “Diagnosis confirmed”
- “Clinically validated” before the clinical study is complete
- “FDA/CDSCO approved” without an applicable authorization
- “The heatmap shows hemorrhage/exudate” without a validated lesion model
- “Safe for autonomous use”

## Regulatory and engineering references

- CDSCO Medical Devices Rules and notices: <https://cdsco.gov.in/opencms/opencms/en/Acts-and-rules/Medical-Devices-Rules/>
- CDSCO medical-device software guidance area: <https://www.cdsco.gov.in/opencms/opencms/en/Medical-Device-Diagnostics/Medical-Device-Diagnostics/>
- FDA/IMDRF Good Machine Learning Practice principles: <https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles>
- ML-enabled medical-device transparency principles: <https://www.fda.gov/medical-devices/software-medical-device-samd/transparency-machine-learning-enabled-medical-devices-guiding-principles>

These references guide planning; the team should obtain qualified clinical, ethics, security, statistical, and regulatory review before a real patient-care release.
