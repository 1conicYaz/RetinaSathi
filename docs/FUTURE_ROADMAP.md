# RetinaSathi future roadmap

Planning estimates below are priorities, not promises.

## Immediate: SIH stabilization

- Release the explicit assessment-state and referral-score contracts.
- Verify one signed-in website screening through Azure, save, History and reload.
- Record immutable model/config hashes, build commit and Azure revision.
- Keep V3.4 XAI disabled and keep unavailable modules visible.
- Rehearse the six-slide deck and local fallback demo.

## 1–3 months: safety and measurement

- Train or adopt a fundus-domain/OOD gate using legally usable negatives and intended-camera images.
- Build a governed clinician review queue with reviewer roles and immutable events.
- Define transformer attribution acceptance tests; show no heatmap until they pass.
- Measure cold/warm cloud latency, memory, failure rate and device-specific image quality.
- Freeze a pristine evaluation protocol before further model selection.

## 3–6 months: independent evidence

- Collect or partner for an Indian intended-camera cohort with consent and governance.
- Use independent ophthalmologist grading with adjudication and patient-level partitions.
- Report sensitivity, specificity, CIs, calibration, QWK, macro-F1 and subgroup/device results.
- Evaluate exact-grade improvements without changing the locked referral threshold on test data.
- Run clinician report-comprehension and review-time studies.

## 6–12 months: controlled pilot readiness

- Complete security threat modelling, backup/restore drills and monitoring.
- Validate camera/operator workflow, recapture rate and referral completion.
- Evaluate ONNX optimization, quantization or a distilled student only after safety gates.
- Add encrypted edge/offline operation for managed clinic workstations.
- Update SimEvents with measured arrival, capture, network and review distributions.

## 12+ months: multi-centre pathway

- Multi-device and multi-centre validation with drift monitoring.
- Prospective controlled pilot under institutional and regulatory guidance.
- Production clinical governance, incident response and model-change control.
- Regulatory and health-economic research for the intended market.

## Three highest priorities after SIH

1. **Independent Indian intended-camera clinical evaluation** because the current source holdout cannot establish real-world safety or generalization.
2. **Tested quality/OOD and governed human-review workflow** because unsafe inputs and unclear reviewer responsibility can harm users even when the classifier is accurate.
3. **Validated explainability and pathology evidence** because an attractive map is not useful unless it is technically sound and clinically interpretable.
