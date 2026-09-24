# Repository guidance

RetinaSathi is a MathWorks-integrated diabetic-retinopathy screening-support
research prototype for SIH26038. MATLAB and Simulink/SimEvents are core project
components. The React, InsForge and Azure path is the supporting demonstration
and deployment interface.

## Current release

- Model: `classifier-v3.4-seed26038`
- Architecture: partially adapted DINOv2 ViT-S/14
- Input: 392 × 392 RGB after the frozen V3.4 preprocessing contract
- Artifact metadata: `models/retinasathi-v3-4.manifest.json`
- Model binaries, datasets, credentials and patient images never belong in Git

## Safety invariants

1. A rejected or ungradeable image must never become non-referable or Routine.
2. Referral comes from the dedicated binary referral head and frozen threshold.
3. Preserve model, preprocessing, calibration and threshold provenance.
4. Never fabricate or relabel an attention map as lesion evidence.
5. Never train on, tune on or repeatedly inspect a frozen evaluation cohort.
6. Never publish secrets, credentials, patient data, datasets or licensed images.
7. Do not automatically retrain or replace the frozen model.
8. Research major model changes before implementation.
9. Knowledge distillation and edge deployment are future research until tested.
10. Do not claim ophthalmologist or clinical validation without recorded evidence.

## Research workflow

Use: **Research → Hypothesis → Experiment → Validation → Integration**.

Every model or dataset proposal must identify the measured problem, protected
evaluation data, success criteria, comparison baseline and rollback decision.

## Verification

```bash
npm ci
./scripts/run_tests.sh
./scripts/verify_v3_4_matlab.sh
```

In MATLAB, verify SimEvents with:

```matlab
cd('simulink')
report = verify_simevents_workflow(true, "results/simevents");
```

MATLAB parity proves implementation consistency. SimEvents results are planning
estimates. Neither establishes clinical validity.
