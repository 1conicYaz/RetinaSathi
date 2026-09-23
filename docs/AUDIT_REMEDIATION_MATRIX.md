# RetinaSathi audit remediation matrix

Evidence labels used here:

- **Freshly verified:** reproduced on 23 September 2026 during this remediation.
- **Historical verified result:** supported by the independent audit or checked-in evidence, but not rerun in this remediation.
- **Unverified:** insufficient current evidence.
- **Future work:** intentionally not claimed as implemented.

| Audit finding | Historical evidence | Current code or service | Reproduced? | Fix | Regression evidence | Current status |
|---|---|---|---|---|---|---|
| Rejected image could be stored as `referable=false` and shown as Routine | Audit S1 | Runtime, persistence and History | Yes, by source inspection | Added explicit `assessment_state`; retake/not-assessed cases carry null grade and null referral decision | Python ungradeable contract; TypeScript schema rejects negative serialization; history label helper tests | Fixed in release branch; production migration pending |
| Referral explanation used sum of grade 2–4 instead of dedicated binary head | Audit S2; 43 disagreements in selected 400-image holdout | Runtime and result panel | Yes | Return, persist and display `referable_score`, threshold and decision from the binary head | Deliberate head-disagreement tests in Python and TypeScript | Fixed in release branch |
| V3.4 explanation map was all zero | Audit S3; explanation probe min=max=0 | Local PyTorch and cloud ONNX | Yes, historical probe reviewed | Disabled V3.4 explanation in both runtimes; UI states that it failed technical validation | V3.4 runtime requires unavailable status and no map | Safely disabled; new XAI is future work |
| Synthetic red noise passed quality | Audit S4 | Deterministic quality gate | Yes | Added circular-field, boundary-coverage and excessive-texture rejection | Negative-control test plus approved retinal demo fixtures | Improved, still heuristic; broader OOD model is future work |
| Inference API lacked authentication and restrictive CORS | Audit S5 | Azure API and InsForge proxy | Freshly rechecked | Azure prediction requires secret header; browser uses signed-in InsForge function; origins restricted | Unauthorized request check; deployed function/source inspection | Pass for intended path; rate limiting remains partial |
| No decoded-pixel cap; synchronous inference blocked event loop | Audit S6 | FastAPI | Yes, by source inspection | Added decoded pixel/dimension limits, one-slot concurrency, thread-pool execution and timeout | API dimension regression test | Fixed in release branch |
| Client can submit model fields to database | Audit S7 | Browser persistence | Yes | Added runtime provenance and result snapshot, but server attestation/signature is not implemented | Contract tests | Partial; trusted persistence is future work |
| Reviewer identity equals record owner | Audit S8 | Review workflow | Yes | Kept wording as “human review”; no clinician role claim | Claim register | Future governed reviewer roles |
| IndexedDB queue lacks encryption and hard expiry | Audit S9 | Offline queue | Yes | Seven-day retention remains; disclosure retained | Source inspection | Partial; managed-device/encryption policy required before pilot |
| Offline replay lacked durable idempotency | Audit S10 | Offline queue and database | Yes | Stable operation UUID, deterministic object key and unique `(user_id, operation_id)` | Pure contract tests; branch migration validation | Fixed in release branch; crash-recovery E2E still required |
| History could not reconstruct complete report provenance | Audit S11 | Screening row | Yes | Added compact `result_snapshot`, model/config hashes, calibration and explanation status | Migration/schema inspection | Fixed for new records after migration |
| Storage/database deletion is not transactional | Audit S12 | Delete flow | Yes | No risky pre-SIH rewrite | Documented limitation | Future reconciliation job |
| API responses used TypeScript assertions only | Audit S13 | Browser boundary | Yes | Added Zod runtime validation and cross-field safety rules | Vitest contract tests | Fixed for prediction responses; history schema validation remains P1 |
| No CI | Audit S14 | Repository | Yes | Added GitHub Actions for frontend checks and Python contracts | Local equivalent commands pass | Fixed in release branch |
| Homepage contained stale V1/DME wording | Audit S15 | Public website | Fresh website inspection | V3.4 wording, unavailable DME/XAI and authenticated Azure route displayed | Live page accessibility snapshot | Freshly verified |

The matrix does not convert engineering tests into clinical validation. Independent intended-camera and prospective clinical evaluation remain required.
