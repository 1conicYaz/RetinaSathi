# RetinaSathi SIH judge Q&A

Every answer separates evidence available now from planned work.

## 1. What actually works today?
**CURRENT:** Authenticated upload, safety-oriented quality gate, V3.4 referral and grade outputs, protected history, MATLAB parity and SimEvents planning. **FUTURE:** Prospective clinical deployment.

## 2. What is experimental?
**CURRENT:** The whole product is a research prototype; five-grade output and heuristic quality require human review. XAI/DME/lesions are unavailable. **FUTURE:** Validate each module independently.

## 3. Which model is deployed?
**CURRENT:** `classifier-v3.4-seed26038`, partial DINOv2 ViT-S/14, ONNX at 392 × 392. **FUTURE:** Change it only after a frozen evaluation gate.

## 4. How do you prove Azure is V3.4?
**CURRENT:** Runtime loading trace, `/health`, `/model-card`, prediction identity and SHA256 tie the service to the manifest. **FUTURE:** Signed release attestation.

## 5. Why Azure Container Apps?
**CURRENT:** Managed HTTPS containers, revision control and scale-to-zero suit an ONNX API. **FUTURE:** Compare cost and latency at pilot load.

## 6. What is scale-to-zero?
**CURRENT:** Minimum replicas are zero, reducing idle cost but causing cold-start delay. **FUTURE:** Set warm capacity if clinical service levels require it.

## 7. How is the API protected?
**CURRENT:** The browser authenticates to InsForge; its server function calls Azure with a server-held key. **FUTURE:** Add stronger rate limits, rotation and managed identity where suitable.

## 8. Can someone call the model directly?
**CURRENT:** The URL is reachable, but prediction requires the inference key; unauthorized valid uploads are rejected. **FUTURE:** Add network/private ingress if operationally justified.

## 9. What happens during cold start?
**CURRENT:** The first request may take longer while the container and 84.27 MiB model load. The UI retries transient gateway failures. **FUTURE:** Measure a latency distribution and tune minimum replicas.

## 10. What happens without internet?
**CURRENT:** A verified local V3.4 runtime can support a demonstration, and pending saves can queue in browser storage. **FUTURE:** Encrypted, resilient offline-first deployment.

## 11. Can it run locally?
**CURRENT:** Yes, FastAPI + ONNX Runtime and MATLAB both run locally with the frozen artifact. **FUTURE:** Package and benchmark target clinic hardware.

## 12. How much memory does it use?
**CURRENT:** Azure is configured for 1 vCPU and 2 GiB; the ONNX artifact is about 84.27 MiB. This is configuration, not measured peak RAM. **FUTURE:** Profile peak and sustained memory.

## 13. What does 92.22% mean?
**CURRENT:** Sensitivity: 166 of 180 referable cases in the selected retrospective holdout were flagged. **FUTURE:** Estimate it on an independent intended-camera cohort.

## 14. Is 92.22% accuracy?
**CURRENT:** No. It is sensitivity. Five-grade macro-F1 was 0.5339. **FUTURE:** Report endpoint-specific metrics with confidence intervals.

## 15. Is it clinically validated?
**CURRENT:** No; evidence is retrospective and engineering-focused. **FUTURE:** Ethics-governed, prospective, multi-site evaluation.

## 16. Why is exact grading weaker?
**CURRENT:** Five classes are imbalanced and visually overlapping, with few severe examples. **FUTURE:** Better labels, diverse data, ordinal objectives and hard-case review.

## 17. Why separate referral and grade heads?
**CURRENT:** Referral is a binary screening endpoint; exact grade is a harder ordered classification task. A dedicated head optimizes the actual referral decision. **FUTURE:** Clinically validate both endpoints separately.

## 18. What happens with bad images?
**CURRENT:** They receive `retake_required`, null grade and null referral decision. **FUTURE:** Replace heuristic quality with a validated multi-device model.

## 19. How do you detect non-retinal inputs?
**CURRENT:** Conservative geometry, brightness, texture, size and decoding checks reject known controls; coverage is incomplete. **FUTURE:** Train and validate retinal/OOD detection on broad negatives.

## 20. Does V3.4 explainability work?
**CURRENT:** No verified V3.4 explanation is displayed. **FUTURE:** Transformer attribution with sanity, perturbation and expert-usefulness tests.

## 21. Why disable the heatmap?
**CURRENT:** The prior map was zero/degenerate and could mislead users. **FUTURE:** Re-enable only after technical and clinical acceptance tests.

## 22. How will you validate XAI?
**CURRENT:** Acceptance criteria are documented; no validity claim is made. **FUTURE:** Test target/input dependence, geometry, stability and lesion/expert agreement.

## 23. Have ophthalmologists approved it?
**CURRENT:** No approval or endorsement is claimed. **FUTURE:** Seek qualified partners for protocol and usefulness review.

## 24. What datasets were used?
**CURRENT:** Project experiments document EyePACS, APTOS, IDRiD and DeepDRiD under their respective labels/licenses. **FUTURE:** Add governed intended-camera Indian data.

## 25. Is DeepDRiD an untouched independent test?
**CURRENT:** No. The patient-separated holdout is source validation and the source also contributed development images. **FUTURE:** Freeze a truly independent cohort.

## 26. What is MATLAB doing?
**CURRENT:** It reproduces preprocessing, ONNX inference, calibration and decision logic; 25 parity cases matched. **FUTURE:** Automate parity in a licensed CI environment.

## 27. Why SimEvents?
**CURRENT:** It models queues, resources, network delay and review capacity. **FUTURE:** Calibrate assumptions with real workflow observations.

## 28. Is 127,000 patients/year real?
**CURRENT:** No. It is 508 simulated daily completions × 250 assumed days. **FUTURE:** Replace assumptions with pilot measurements.

## 29. What is innovative?
**CURRENT:** One reproducible system combines calibrated referral, safe failure states, authenticated cloud delivery, MATLAB parity and workflow simulation. **FUTURE:** Prove real-world value with independent data.

## 30. Did you perform knowledge distillation?
**CURRENT:** No. **FUTURE:** Evaluate distillation only after safety and accuracy are frozen.

## 31. Why DINOv2?
**CURRENT:** It provided strong transferable features and practical ViT-S size under our comparison process. **FUTURE:** Reassess on independent data rather than popularity.

## 32. Why not a giant foundation model?
**CURRENT:** Cost, latency and reproducibility matter; ViT-S is a practical compromise. **FUTURE:** Compare larger teachers for research without assuming they deploy better.

## 33. How will it work in rural areas?
**CURRENT:** Cloud prototype plus local fallback and operational simulation. **FUTURE:** Validate weak-network, power, camera, device and staff workflows in the intended setting.

## 34. How is patient data protected?
**CURRENT:** Authentication, private storage, owner RLS, HTTPS and server-side secrets. **FUTURE:** Formal retention, encryption, consent, access audit and incident policies.

## 35. What data leaves the device?
**CURRENT:** In cloud mode, the retinal image is sent through InsForge to Azure and stored in the protected workspace after analysis. **FUTURE:** Offer a validated edge mode with explicit data policy.

## 36. What are the next three milestones?
**CURRENT:** Prototype stabilization is complete enough for a controlled demo. **FUTURE:** Independent intended-camera evaluation; tested quality/OOD and reviewer governance; validated explanation/pathology research.

## 37. How will you clinically validate it?
**CURRENT:** No clinical validation claim. **FUTURE:** Freeze model/threshold, obtain approvals, use patient-level reference grading, report confidence intervals, calibration, subgroups and false negatives.

## 38. What happens after SIH?
**CURRENT:** Preserve the frozen release and evidence. **FUTURE:** Execute the staged roadmap before optimization or pilot claims.

## 39. What would SIH support enable?
**CURRENT:** A reproducible base exists. **FUTURE:** Clinical/data partnerships, intended-camera testing, security hardening and deployment evaluation.

## 40. What is the biggest limitation?
**CURRENT:** Lack of independent prospective intended-camera clinical validation. **FUTURE:** Address that before positioning the system for patient care.
