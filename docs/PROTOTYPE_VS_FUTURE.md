# Current prototype versus future vision

| Capability | Current prototype | Evidence | Future target |
|---|---|---|---|
| Image intake | JPEG/PNG/WebP, 15 MiB encoded limit, decoded-size gate | Code and tests | Camera-specific acquisition guidance and DICOM/device integration |
| Quality | Conservative deterministic gate with known limitations | Negative controls and demo fixtures | Validated fundus/OOD model across intended cameras |
| DR referral | Dedicated calibrated binary head | Retrospective source holdout and runtime contract | Independent intended-camera validation and threshold governance |
| DR grade 0–4 | Calibrated ordinal/nominal fusion | Retrospective source holdout | Better minority-grade performance with independent test |
| DME | Unavailable | Runtime model card | Validated task using suitable labels/OCT or clinical reference standard |
| Lesions and NV | Unavailable in deployed path | Runtime module status | Clinician-validated lesion/NV model with localization metrics |
| Explainability | Disabled for V3.4 | Prior zero-map audit; current safe state | Target-sensitive transformer attribution plus sanity and clinician studies |
| Human review | Operator-owned review event | Source and RLS | Verified clinician roles, assignment queue and audit governance |
| Cloud inference | Authenticated InsForge proxy to Azure | Fresh cloud inspection | Monitoring, per-user quota, explicit probes and restore drills |
| Offline | Browser queue and local inference option | Source; limited lifecycle evidence | Encrypted managed-device queue and robust two-way reconciliation |
| MATLAB | Native V3.4 preprocessing/inference/parity app | Historical 8 tests and 25 parity cases | Automated cross-runtime release gate |
| SimEvents | Executable capacity model | Historical 14 checks | Parameters measured in real clinics and uncertainty analysis |
| Clinical validity | None claimed | No prospective study | Ethics-approved, independently graded, multi-site evaluation |

Current outputs always require qualified human review and must not be used as an autonomous diagnosis.
