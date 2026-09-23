# RetinaSathi official six-slide content

Keep the deck to six slides including the title. Use screenshots from the verified release and cite the claim register.

## Slide 1 — Problem and vision

**RetinaSathi · SIH26038**

- DR screening access is limited in resource-constrained settings.
- Vision: affordable, human-in-the-loop screening support.
- Position clearly: research prototype; not a diagnosis.

## Slide 2 — Working prototype

Show: `Fundus image → quality gate → V3.4 → referral + grade → human review → protected history`.

- React + InsForge authentication/storage
- authenticated function → protected Azure Container App
- MATLAB parity and SimEvents planning
- DME, lesion analysis and V3.4 XAI marked unavailable/future

## Slide 3 — Technical evidence

- V3.4: partial DINOv2 ViT-S/14, 392 × 392, 84.27 MiB ONNX
- Selected 400-image reused DeepDRiD source holdout: sensitivity 92.22%, specificity 91.36%, AUROC 0.9756, QWK 0.7910, macro-F1 0.5339
- Fresh engineering checks: 8 MATLAB tests, 25/25 parity cases, 14/14 SimEvents checks
- Footnote: retrospective source validation; not prospective clinical validation

## Slide 4 — Safety and honesty

- Bad image → not assessed / retake, never Routine
- Dedicated referral score + frozen threshold shown
- Every result requires human review
- V3.4 XAI disabled after failed technical validation
- Known gaps: independent cohort, exact grading, OOD breadth, clinical governance

## Slide 5 — Deployment and roadmap

Show: `Browser → InsForge Auth → server function → Azure V3.4 ONNX`.

- Private storage and owner-scoped history
- Scale-to-zero: lower idle cost, possible cold-start delay
- Next: independent intended-camera evaluation → stronger quality/OOD → validated XAI/pathology → edge optimization → controlled pilot

## Slide 6 — Potential impact and ask

- Potential earlier prioritization and more organized screening workflows
- SimEvents supports capacity planning; it is not field throughput evidence
- Ask for clinical/data partnership, intended-camera evaluation and deployment mentoring
- Add GitHub, demo URL and contact QR codes only if permitted
