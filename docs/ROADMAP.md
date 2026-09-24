# RetinaSathi research roadmap

The roadmap is organized by dependency and evidence, without calendar promises.
Major changes follow:

**Research → Hypothesis → Experiment → Validation → Integration**

## Current verified prototype

### MATLAB

- Frozen V3.4 ONNX execution
- Matched preprocessing, calibration and referral threshold
- 8 passing tests and 25/25 Python-reference parity cases

### Simulink/SimEvents

- Capture, recapture, network, AI and human-review workflow
- Seven scenarios and 14/14 software checks
- Queue, utilization, waiting-time and throughput planning

### Supporting web deployment

- Authenticated InsForge interface and protected history
- InsForge server function to protected Azure Container Apps inference
- Explicit assessment states, model provenance and human-review requirement

## Priority 1: stabilize the verified prototype

- Reproduce MATLAB and SimEvents verification on a clean reference environment.
- Verify one authenticated Azure screening, persistence and reload.
- Record model/config hashes, Git commit and active Azure revision.
- Keep V3.4 XAI disabled and unavailable modules visible.
- Maintain a tested local inference fallback.
- Replace unsupported 5,000-case claims with measured benchmarks and simulation.

## Priority 2: strengthen research and clinical understanding

- Review current DR grading, quality/OOD and transformer-attribution literature.
- Interview ophthalmologists about capture quality, dangerous errors, referral
  criteria, report content, heatmap usefulness and expected review time.
- Review false negatives, false positives, grade disagreements and uncertain
  cases with qualified clinicians when a governed collaboration is available.
- Complete the EyePACS admission audit and controlled baseline experiment only
  if provenance and usage rights are acceptable.
- Define a frozen independent evaluation protocol before further model choice.
- Load-test Azure cold/warm latency, concurrency, memory, failures and cost.

No document may state “doctors validated RetinaSathi” until supporting evidence
exists. The current goal is ophthalmologist collaboration and workflow research.

## Priority 3: build independent data and safety evidence

- Prepare an Indian intended-camera cohort with governance, patient grouping,
  two qualified graders and adjudication.
- Train or adopt a fundus quality/OOD gate using intended-camera images and
  legally usable negatives.
- Report confidence intervals, calibration and results by site, camera, image
  quality, age, sex and grade.
- Improve Grade-2 stability and Grade-4 recall without altering the frozen V3.4
  evidence or tuning against the final test cohort.
- Validate lesion modules before using them as clinical evidence.

## Validated explainability research

1. Select transformer-appropriate attribution methods from literature.
2. Test non-degeneracy and sensitivity to target class and model parameters.
3. Run perturbation/insertion/deletion tests and verify input geometry.
4. Compare against lesion annotations where rights and labels allow.
5. Ask ophthalmologists whether maps are useful, misleading or unnecessary.
6. Integrate a map only after technical and clinical acceptance criteria pass.

## Knowledge distillation and edge-model research

Goal: reduce memory, latency and cloud dependence while preserving safety.

1. Benchmark teacher candidates under one frozen protocol; do not assume V3.4,
   RETFound or a larger foundation model is automatically the best teacher.
2. Benchmark compact students such as EfficientNet, MobileNet or a compact ViT.
3. Compare ground-truth training with referral-logit, grade-logit, ordinal and,
   where justified, feature-level distillation.
4. Evaluate against a non-distilled student using sensitivity, specificity,
   AUROC, AUPRC, QWK, macro-F1, per-grade recall and calibration.
5. Measure model size, peak RAM, CPU latency, energy and cost.
6. Reject a faster student if it fails the referral-safety gate.
7. Export accepted candidates to ONNX and repeat Python↔ONNX↔MATLAB parity.

Distillation is future research. No distilled RetinaSathi model exists today.

## Priority 4: prepare for a controlled pilot

- Implement encrypted local inference, resumable synchronization, idempotency,
  secure model updates and recovery from interrupted connectivity.
- Conduct clinician report-comprehension and timed-review studies.
- Replace SimEvents estimates with measured capture, recapture, bandwidth,
  inference and reviewer-time distributions.
- Complete threat modelling, access audits, backup/restore drills and monitoring.
- Prepare a controlled pilot only with institutional, ethical and clinical support.

## Priority 5: multi-centre and regulatory research

- Evaluate multiple clinics, cameras, populations and acquisition conditions.
- Monitor calibration, data drift, referral completion and safety incidents.
- Establish model-change control and clinical governance.
- Investigate relevant medical-device, privacy and health-economic requirements.

## Research questions

- **RQ1:** Does admitted EyePACS development data improve Grade-2 recall and
  macro-F1 without weakening Indian-source referral performance?
- **RQ2:** Can distillation reduce V3.4 size and latency while preserving
  referral sensitivity and calibration?
- **RQ3:** Which attribution method is technically stable and clinically useful?
- **RQ4:** How does performance vary across intended fundus cameras and sites?
- **RQ5:** Which edge/cloud design is reliable under rural connectivity?
- **RQ6:** What reviewer capacity is needed for selective versus universal review?

## Highest priorities

1. Ophthalmologist workflow research and governed collaboration.
2. Independent Indian intended-camera evaluation.
3. Quality/OOD safety and false-negative analysis.
4. Validated lesion evidence and explainability.
5. Exact-grade improvement, especially Grades 2 and 4.
6. Controlled EyePACS and multi-camera data experiments.
7. Distillation and secure offline/edge deployment.
