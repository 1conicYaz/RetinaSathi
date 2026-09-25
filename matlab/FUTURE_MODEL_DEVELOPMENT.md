# RetinaSathi future model-development path

This document separates the working research prototype from proposed
development. It does not claim that future modules are trained, validated or
clinically available.

## What the MATLAB prototype demonstrates now

- Image-quality screening stops unusable photographs and requests recapture.
- V3.4 produces a separate referral score and five-grade DR probabilities.
- V3.2 produces experimental candidate masks for microaneurysms,
  haemorrhages, hard exudates and soft exudates.
- The lesion module uses its frozen ONNX hash, per-class thresholds,
  Gaussian overlap blending and retinal-border margin.
- The lesion result cannot change V3.4's referral or grade decision.
- Every result requires human review; DME and validated classifier
  explainability remain unavailable.
- SimEvents represents patient flow, cameras, network delay, inference
  capacity and reviewer queues. It does not execute pixel inference for every
  simulated entity.

V3.2 passed its internal 11-image full-resolution engineering gate with mean
Dice 0.3937. The current overlay still produces border and optic-disc false
positives, so it is an experimental prototype rather than clinical evidence.

## Governed data expansion

1. Record the licence, source, camera, field of view, resolution, patient key,
   eye and label definition for every dataset.
2. Use patient-level splits. Keep images from the same patient in one split.
3. Deduplicate within and across datasets before splitting.
4. Keep one locked test set that is never used for model choice, thresholds or
   preprocessing decisions.
5. Combine suitable public sources such as IDRiD, EyePACS, APTOS, Messidor-2
   and lesion-labelled resources only after checking licence and label
   compatibility.
6. Add de-identified images from intended Indian cameras under consent,
   governance and ophthalmologist-reviewed labels.
7. Report results separately by dataset, camera, site, image quality and DR
   grade so a large mixed score cannot hide a weak subgroup.

## Classification development

The screening decision and exact grade remain separate tasks:

- quality and out-of-distribution gate;
- binary referable-DR head;
- ordinal severity head that respects the order G0 to G4;
- nominal five-grade head for exact-grade probabilities;
- DME-risk module only when suitable labels and clinical validation exist;
- calibrated uncertainty routing to human review.

EfficientNet, DINO and retinal foundation models should be compared using the
same patient split, preprocessing, augmentation, calibration and evaluation
gate. Model selection should consider referral sensitivity and specificity,
AUROC, calibration, quadratic weighted kappa, macro-F1, subgroup stability,
latency and memory. A model is not selected from accuracy alone.

## Segmentation development

The next lesion model should address V3.2's visible failure modes directly:

1. Train with larger-context patches while scoring only the reliable centre
   crop. This prevents patch edges from being treated as retinal structures.
2. Continue model-mined hard negatives from the retinal rim, optic disc,
   vessels, reflections and camera artefacts.
3. Use anatomy modules or verified masks to distinguish the optic disc and
   vessels from bright and red lesions.
4. Preserve native lesion detail; do not shrink a full 4K image into one small
   square for microaneurysm training.
5. Compare a shared four-class model with class-specific red-lesion and
   bright-lesion heads under the same locked split.
6. Measure pixel Dice/IoU together with lesion sensitivity, false positives per
   image, precision-recall performance and per-image visual review.
7. Reject a candidate that passes aggregate metrics but shows tile patterns,
   retinal-border leakage or systematic optic-disc confusion.

## Teacher comparison and distillation

Distillation is conditional, not automatic:

1. Train or adapt candidate teachers under one frozen protocol.
2. Select a teacher only if it improves locked referral, grading or lesion
   evidence without unacceptable calibration or subgroup failures.
3. Train a compact student using ground-truth loss plus teacher logits or
   features. Keep classification and segmentation objectives explicit.
4. Calibrate and validate the student independently. It cannot inherit the
   teacher's evidence or safety claims.
5. Distil only when the smaller model materially improves edge latency,
   memory, power or offline deployment.

## MATLAB and deployment gate

1. Freeze PyTorch weights, preprocessing, thresholds, class order and hashes.
2. Export ONNX and reject unsupported or numerically unstable operations.
3. Compare Python, ONNX Runtime and MATLAB on fixed images and intermediate
   tensors, not only the final label.
4. Check segmentation masks, per-class pixel counts and connected regions
   within declared tolerances.
5. Measure local CPU/GPU latency and memory on the intended district hardware.
6. Package an offline MATLAB path for weak connectivity and a protected cloud
   path for connected sites.
7. Use SimEvents measurements to size cameras, local compute, bandwidth and
   reviewers without presenting simulated throughput as hospital evidence.

## Evidence required before clinical use

- untouched independent multi-camera evaluation;
- prospective evaluation in the intended workflow;
- ophthalmologist review of false negatives and false positives;
- quality/OOD and failure-recovery testing;
- documented privacy, access control, audit and retention policies;
- versioned model, dataset and threshold provenance;
- regulatory and institutional approval appropriate to the intended use.

Until these gates pass, RetinaSathi remains research screening support and
must not be described as an autonomous diagnosis system.
