# RetinaSathi suggested six-slide deck

Keep the deck to six slides including the title. Use screenshots from the verified release and cite the measured-result documents linked from `README.md`.

## Slide 1 — Problem and vision

**RetinaSathi · SIH26038**

- DR screening access is limited in resource-constrained settings.
- Vision: affordable, human-in-the-loop screening support.
- Position clearly: research prototype; not a diagnosis.

## Slide 2 — Proposed solution and MATLAB architecture

Show: `Fundus image → quality gate → V3.4 ONNX → MATLAB calibration/referral + grade → human review`.

- Frozen preprocessing, model hash, calibration and referral threshold
- 8 MATLAB tests and 25/25 Python-reference parity cases
- DME, lesion analysis and V3.4 XAI marked unavailable/future

## Slide 3 — SimEvents and AI evidence

- Show the executable capture → recapture → network → AI → reviewer workflow
- Explain camera, network, compute and reviewer bottlenecks
- V3.4: partial DINOv2 ViT-S/14, 392 × 392, 84.27 MiB ONNX
- Selected 400-image reused DeepDRiD source holdout: sensitivity 92.22%, specificity 91.36%, AUROC 0.9756, QWK 0.7910, macro-F1 0.5339
- 14/14 SimEvents checks; label throughput as simulation
- Footnote: retrospective source validation; not prospective clinical validation

## Slide 4 — Safety and honesty

- Bad image → not assessed / retake, never Routine
- Dedicated referral score + frozen threshold shown
- Every result requires human review
- V3.4 XAI disabled after failed technical validation
- Known gaps: independent cohort, exact grading, OOD breadth, clinical governance

## Slide 5 — Supporting deployment and feasibility

Show: `Browser → InsForge Auth → server function → Azure V3.4 ONNX`.

- Private storage and owner-scoped history
- Scale-to-zero: lower idle cost, possible cold-start delay
- State that the present one-replica service has not demonstrated 5,000 cases in two minutes

## Slide 6 — Research roadmap, impact and ask

- Potential earlier prioritization and more organized screening workflows
- SimEvents supports capacity planning; it is not field throughput evidence
- Research → ophthalmologists → independent Indian data → validated XAI → distillation → secure edge/offline → controlled pilot
- Ask for clinical/data partnership, intended-camera evaluation and MathWorks mentoring
- Add GitHub, demo URL and contact QR codes only if permitted
