# External validation protocol

Messidor is reserved for one-way external evaluation after the classifier architecture, preprocessing, checkpoint, temperature, and referable threshold are locked on APTOS/IDRiD development data. Messidor must never influence model selection, early stopping, calibration, threshold tuning, or preprocessing choices.

The compatible primary endpoint is binary referable diabetic retinopathy. RetinaSathi maps Messidor retinopathy grades 0–1 to non-referable and grades 2–3 to referable. Because Messidor has four grades while the training tasks use five, no five-class QWK or invented grade-4 mapping will be reported. Sensitivity, specificity, balanced accuracy, confusion counts, and bootstrap confidence intervals will be reported for this fixed endpoint.

The 0–2 macular-edema risk label is reported separately and only if the locked model includes a DME head. The 12 workbooks and 1,200 local images are pairing-audited by `scripts/audit_messidor.py`; only the aggregate audit JSON is committed. Images, filenames, predictions, and row-level annotations stay outside Git.

The one-way evaluation was completed on 2026-09-10 after the immutable model lock. On 1,200 images the frozen grade≥2 endpoint produced sensitivity 0.4551 (95% CI 0.4125–0.5000), specificity 0.9757 (0.9641–0.9863), and balanced accuracy 0.7154 (0.6924–0.7387). This indicates material domain shift and under-referral; it is negative but important evidence, not clinical validation. The authoritative hash-bound result is `artifacts/locked_evaluation.json`. No Messidor result may be used to retune this locked model.
