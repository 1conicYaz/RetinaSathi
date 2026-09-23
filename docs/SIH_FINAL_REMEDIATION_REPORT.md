# RetinaSathi SIH26038 Final Remediation Report

Evidence labels: **Freshly verified**, **Historical verified result**, **Inference / recommendation**, **Unverified**, **Contradicted**, and **Future work**.

## 1. Executive Summary

RetinaSathi is **demo-ready with documented limitations**, except that the final browser file-upload → save → History path is not yet verified. The V3.4 model, secured Azure runtime, InsForge proxy, database safety migration, frontend result semantics, MATLAB parity, SimEvents model and release checks are verified. The product remains a research screening-support prototype requiring human review.

## 2. What Changed Since the Independent Audit

- **Freshly verified:** Azure serves V3.4 image tag `6` through the protected prediction route.
- **Freshly verified:** InsForge authenticates users and proxies to Azure with a server-side key.
- **Freshly verified:** explicit assessment states prevent rejected images becoming Routine.
- **Freshly verified:** the dedicated binary referral score/threshold drive referral wording.
- **Freshly verified:** V3.4 XAI is disabled after the historical zero-map failure.
- **Freshly verified:** runtime identity now includes model/config hashes, threshold, calibration, Git commit and release.
- **Freshly verified:** the production database migration and updated website/function are deployed.

## 3. Audit Findings Matrix

| Finding | Severity | Historical Status | Current Status | Fix | Test |
|---|---|---|---|---|---|
| Retake saved as negative/Routine | P0 | Failed | Fixed | Explicit state + null clinical fields + DB checks | Python and TypeScript contracts |
| Referral explained from grade sum | P0 | Failed | Fixed | Dedicated score, threshold and decision | Deliberate disagreement tests |
| Degenerate V3.4 XAI | P0 | Failed | Safely disabled | No V3.4 map displayed | Runtime/UI tests |
| Red noise passed quality | P0 | Failed | Improved | Field geometry and texture controls | Negative-control tests |
| Weak API bounds/concurrency | P1 | Partial | Improved | Pixel/dimension limit, timeout, semaphore/threadpool | API tests |
| Response assertions only | P1 | Failed | Fixed for predictions | Zod boundary validation | Vitest |
| Offline replay duplication | P1 | Partial | Improved | Stable operation UUID + unique index | Schema/contract tests |
| Missing provenance/history snapshot | P1 | Failed | Fixed for new rows | Hashes, calibration, state and compact snapshot | Branch migration verification |
| No CI | P1 | Failed | Fixed | GitHub Actions | Local equivalent passes |

The detailed mapping is in `docs/AUDIT_REMEDIATION_MATRIX.md`.

## 4. Azure Deployment Verification

- **Freshly verified:** Container App `retinasathi-v34`, resource group `retinasathi-azure`, UAE North.
- **Freshly verified:** image `retinasathiv34uae.azurecr.io/retinasathi-v34:6`, healthy and provisioned.
- **Freshly verified:** 1 vCPU, 2 GiB, min 0, max 1, HTTP concurrency trigger 10.
- **Freshly verified:** external HTTPS ingress; insecure HTTP disabled.
- **Freshly verified:** model SHA `8d5373211b8651e1a3c787a19bfd52a64383d8a2296f370cbf2615467422b2ec`.
- **Freshly verified:** commit `89615dc`, release `retinasathi-v34--latest`.
- **Not run:** controlled cold-versus-warm latency distribution.
- **Partial:** no explicit Container Apps liveness/readiness/startup probes.

## 5. Website / InsForge Verification

- **Freshly verified:** <https://69exmaqk.insforge.site/> loads and displays current V3.4/limitation wording.
- **Freshly verified:** GitHub OAuth reached the authenticated screening page.
- **Freshly verified:** function `retinasathi-inference` is deployed and checks `auth.getCurrentUser()`.
- **Freshly verified:** private storage and owner-scoped database policies exist.
- **Freshly verified:** production build, deployment and proxy update completed.

## 6. Complete E2E Test

**NOT RUN — browser upload permission blocked the final step.** Login passed, but Chrome file upload required its extension setting “Allow access to file URLs.” Therefore image → Azure prediction → display → save → History → reload is not claimed as passed. Execute `DEMO.md` after enabling that setting.

## 7. Model Status

| Field | Current value |
|---|---|
| Model | `classifier-v3.4-seed26038` |
| Architecture | partial DINOv2 ViT-S/14 |
| Artifact | 84.27 MiB ONNX, excluded from Git |
| SHA256 | `8d5373211b8651e1a3c787a19bfd52a64383d8a2296f370cbf2615467422b2ec` |
| Config SHA256 | `a0254e268417590cc4437af06824ed5ea7cc67c7c4bccb4292263d723818f9f1` |
| Input | 392 × 392 RGB |
| Threshold | `0.20732617378234863` |
| Calibration | manifest-bound binary, ordinal and nominal temperatures |
| Local | ONNX Runtime and MATLAB available with local artifact |
| Cloud | Azure Container Apps tag `6` |

## 8. Fresh Test Results

### Passed

- 3 frontend contract tests; ESLint; TypeScript; Vite production build.
- 23 compute tests and 61 ML tests.
- npm audit: 0 known vulnerabilities.
- MATLAB: 8 tests and 25/25 parity cases; grade/referral agreement 1.000.
- SimEvents: 14/14 checks across seven scenarios.
- Production migration branch validation and merge.
- Valid retinal multipart without inference key: HTTP 401.
- Live `/health` and `/model-card` identity checks.

### Failed

- Initial tag `4` Azure rollout failed because the legacy builder emitted an unsupported image format. It was replaced by Buildx OCI tags `5` and final `6`; tag `6` is healthy.

### Not Run

- Final authenticated browser upload/save/History/reload.
- Controlled cold/warm latency benchmark.
- Prospective clinical or intended-camera validation.

## 9. Safety Status

- Ungradeable images have null grade/referral and `retake_required`.
- Quality/OOD rejection is stronger but remains heuristic.
- Referral uses the dedicated binary head.
- New History rows preserve state and provenance.
- Human review is required for every result; reviewer role governance is still prototype-level.
- Storage and database deletion are not a single transaction.

## 10. Explainability Status

**DISABLED / NOT VERIFIED.** The prior V3.4 map failed technical validation. No decorative or V2 map is presented as V3.4 evidence.

## 11. MATLAB Status

**Freshly verified:** 8 tests passed and all 25 frozen parity cases matched Python/ONNX for grade and referral. This is implementation parity, not clinical validation.

## 12. SimEvents Status

**Freshly verified:** 14/14 software checks passed across seven scenarios. The 127,000 annual-equivalent district result is 508 simulated completions × 250 assumed days, not observed patient throughput.

## 13. Security Status

- Azure key absent from frontend and held by the InsForge function.
- Direct valid unauthorized prediction blocked with 401.
- HTTPS, restrictive origins, owner RLS and private storage verified.
- Byte/pixel/dimension/time/concurrency limits implemented.
- Secret values and model binaries excluded from Git.
- **Partial:** no per-user quota, signed result attestation or formal operational threat model.

## 14. Files Changed

- `compute/`: safety states, identity, input limits, timeout/concurrency and tests.
- `src/lib/`: Zod contract, persistence provenance and idempotent operation IDs.
- `src/components/`: correct referral, retake, uncertainty and XAI-unavailable wording.
- `migrations/20260923233000_screening-result-contract-v3.sql`: production safety schema.
- `functions/retinasathi-inference.ts`: authenticated, bounded Azure proxy.
- `.github/workflows/ci.yml`: frontend and Python contract gates.
- `README.md`, `DEMO.md`, `docs/`: corrected claims, deployment evidence, presentation and team guides.

## 15. GitHub Readiness

### SAFE TO COMMIT

Source, tests, migrations, manifests, aggregate evidence, diagrams and honest documentation in PR #1.

### DO NOT COMMIT

`.env.local`, `.insforge/project.json`, Azure/InsForge keys, licensed retinal images, patient data, `.onnx`, `.pt/.pth/.ckpt`, generated import packages or local run directories.

## 16. SIH P0 Blockers

1. Enable Chrome extension file URL access and complete one production valid-image and one retake-image E2E test.
2. Confirm PR CI and merge PR #1 before presenting a GitHub main-branch link.
3. Rehearse the local V3.4 fallback with the exact hash-matching artifact.

## 17. P1 Improvements

- Explicit Azure health/startup/readiness probes and cold/warm benchmark.
- Runtime schemas for History/review responses.
- Per-user rate limit and result attestation.
- Stronger offline queue recovery/encryption and storage reconciliation.

## 18. P2 Engineering Work

- Bundle splitting, observability dashboards, backup/restore drill, model registry automation, reviewer assignment and immutable audit events.

## 19. Clinical / Research Validation

Freeze model and threshold before a patient-level, independent, intended-camera study with qualified reference grading, ethics/governance, sensitivity/specificity CIs, calibration, QWK, subgroup/device analysis and false-negative review.

## 20. Exact SIH-Safe Claims

- “RetinaSathi is a human-in-the-loop DR screening-support research prototype.”
- “V3.4 is partial DINOv2 ViT-S/14 and is deployed through an authenticated InsForge-to-Azure path.”
- “On a reused patient-separated 400-image DeepDRiD source holdout, selected-seed sensitivity was 92.22% and specificity 91.36%.”
- “MATLAB parity and SimEvents are engineering and planning evidence, not clinical validation.”
- “V3.4 explanation, DME and lesion modules are unavailable.”

## 21. Claims We Must NOT Make

- Clinically approved, doctor approved, hospital validated or ready for autonomous diagnosis.
- 92.22% five-grade accuracy.
- DeepDRiD was an untouched independent final test.
- Heatmap detects lesions, V3.4 XAI works, or DME/NV is assessed.
- 127,000 patients were screened.
- Knowledge distillation trained V3.4.
- All cloud-mode data stays on device.

## 22. Current Prototype vs Future Vision

Current: authenticated upload, heuristic quality gate, calibrated referral, grade support, review/history, MATLAB parity and operational simulation. Future: independently validated quality, DR, pathology and explanation; governed ophthalmologist review; secure edge/offline workflow; controlled rural pilot.

## 23. Future Roadmap

### Immediate
Complete E2E, merge release, rehearse cloud/local demo.

### 1–3 months
Quality/OOD dataset, reviewer governance, attribution acceptance tests and deployment monitoring.

### 3–6 months
Independent intended-camera cohort and stronger grade analysis without test-set tuning.

### 6–12 months
Controlled pilot readiness, security operations and validated edge optimization.

### 12+ months
Multi-centre validation, drift monitoring and regulatory/product pathway research.

## 24. Six-Slide PPT Content

1. Problem and human-in-the-loop vision.
2. Working quality → V3.4 → review → history prototype.
3. Retrospective and engineering evidence with limitations.
4. Safety states, honest XAI status and gaps.
5. Secure deployment and staged roadmap.
6. Potential impact, partnership ask and links.

Exact wording is in `docs/SIH_SIX_SLIDE_CONTENT.md`.

## 25. Video Presentation Plan

Show problem, architecture, signed-in valid screening, poor-quality safety, History, security path, MATLAB, SimEvents, evidence limitations and roadmap. Use `docs/SIH_VIDEO_SCRIPT.md`.

## 26. Judge Q&A

Forty CURRENT-versus-FUTURE answers are in `docs/SIH_JUDGE_QA.md`. The most important answer is that the largest limitation is missing independent prospective intended-camera validation.

## 27. Demo Checklist

Verify live identity → sign in → approved valid fixture → score/threshold/grade → History/reload → poor fixture → retake state → MATLAB parity → SimEvents → limitations/roadmap. Follow `DEMO.md`.

## 28. Commands to Reproduce Everything

```bash
npm ci
npm test
npm run lint
npm run typecheck
npm run build
python3.11 -m venv .venv
.venv/bin/python -m pip install -r compute/requirements.txt
.venv/bin/python -m unittest discover -s compute/tests -v
.venv/bin/python -m unittest discover -s ml/tests -v
./scripts/verify_v3_4_matlab.sh
```

```matlab
cd('simulink')
verify_simevents_workflow(true, "results/simevents");
```

```bash
curl -fsS https://retinasathi-v34.proudcoast-d5c4449c.uaenorth.azurecontainerapps.io/health
curl -fsS https://retinasathi-v34.proudcoast-d5c4449c.uaenorth.azurecontainerapps.io/model-card
```

## 29. Current Final Verdict

**Demo-ready with documented limitations, pending one manual browser-upload E2E gate and PR merge.** The core safety defects are fixed and tested; V3.4 identity and protected cloud route are fresh; MATLAB/SimEvents are reproducible; unsupported medical capabilities remain visibly unavailable. No clinical-use claim is justified.

The three most important post-SIH priorities are: (1) independent Indian intended-camera clinical evaluation, (2) robust quality/OOD plus governed human review, and (3) technically and clinically validated explanation/pathology evidence.
