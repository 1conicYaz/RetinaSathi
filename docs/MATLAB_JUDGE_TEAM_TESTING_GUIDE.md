# RetinaSathi V3.4 — MATLAB, Team and Judge Testing Guide

## What is being tested

RetinaSathi V3.4 is a five-grade diabetic-retinopathy research screening
candidate. It receives one colour fundus photograph and returns:

- image-quality feedback;
- DR Grade 0, 1, 2, 3 or 4;
- calibrated grade probabilities;
- a separate referable-DR decision;
- a safe next-step message;
- `DME not assessed`, because V3.4 has no DME head.

It supports screening and demonstrations. It is not a diagnosis and has not
completed a prospective clinical study or the one-time locked external test.

## Files to keep together

- `models/retinasathi-v3-4.onnx` — portable V3.4 weights, 84.27 MiB.
- `models/retinasathi-v3-4.manifest.json` — input, preprocessing, calibration,
  output and SHA-256 contract.
- `runs/v3_4_matlab_parity/` — fixed Python reference cases for MATLAB parity.
- `matlab/` — native preprocessing, ONNX inference, calibration, tests and app.
- `simulink/` — screening workflow model and scenario runner.

Never test a model copied from another location unless its SHA-256 matches the
manifest. The MATLAB pipeline checks this before import.

## One-command technical verification

From the repository root:

```bash
./scripts/verify_v3_4_matlab.sh
```

The expected result is:

- 8 MATLAB tests pass;
- 25/25 fixed cases have the same grade as Python;
- 25/25 fixed cases have the same referral decision as Python;
- no model-hash, preprocessing, tensor or calibration failure.

This proves implementation parity. It does not prove clinical safety or
accuracy on a new hospital population.

## Manual MATLAB test

Open MATLAB and run:

```matlab
cd('/path/to/X-Retina/matlab')
setup
status = check_toolboxes()
tests = runtests("tests")
assert(all([tests.Passed]))

report = runV34Parity("../runs/v3_4_matlab_parity", ...
    "../models/retinasathi-v3-4.onnx", ...
    "../models/retinasathi-v3-4.manifest.json", ...
    "../runs/v3_4_matlab_parity/matlab_parity_report.json")
assert(report.passed)
```

Test one image:

```matlab
result = runRetinaPipelineV34("../runs/demo_cases/referable.jpg")
disp(result)
```

Open the MATLAB app:

```matlab
cfg = struct("modelGeneration", "v3.4", ...
    "modelPath", "../models/retinasathi-v3-4.onnx", ...
    "manifestPath", "../models/retinasathi-v3-4.manifest.json");
launchRetinaSathiApp(cfg)
```

Choose a retinal image and confirm that the app shows quality, grade,
probabilities, referral status and the research-use warning. Confirm that it
does not invent DME, lesions or an explanation that the exported model cannot
produce.

## Simulink test

In MATLAB:

```matlab
cd('/path/to/X-Retina/simulink')
report = verify_simevents_workflow(true, "results/simevents");
model = launch_simevents_demo(20);
set_param(model,'SimulationCommand','start')
```

Show the judge the live queue/completion displays, the capture/AI/review queue
scopes and the 14/14 verification report. SimEvents demonstrates operational
capacity and routing; real V3.4 numerical inference is verified separately in
MATLAB.

## Teammate acceptance test

Each teammate should use a fresh browser session and record results in
`docs/TEAM_JUDGE_TEST_RESULTS.csv`. Do not enter a real patient's name.

Test these cases:

1. **Good non-referable image:** a result appears, all five probabilities are
   visible, and the output still asks for human review.
2. **Referable image:** the next step clearly asks for ophthalmologist review.
3. **Severe image:** the grade and warning are visible without calling the
   result a diagnosis.
4. **Poor-quality image:** inference stops and asks for recapture.
5. **Uncertain image:** low confidence routes to review instead of presenting
   a confident routine result.
6. **Invalid file:** a clear error appears for a non-image or corrupt image.
7. **DME field:** it says `Not assessed — classifier has no DME head`.
8. **Cloud explanation:** if the cloud ONNX runtime is used, it says the
   attention map is unavailable. The local PyTorch runtime is required for the
   experimental gradient-based attention view.
9. **History and privacy:** the saved record uses a patient code, appears only
   for the signed-in user, and contains the correct model version.
10. **Refresh/retry:** refresh the page and repeat one screening to check that
    the service wakes from scale-to-zero and completes normally.

For every failure, record the image ID, runtime (`local` or `cloud`), displayed
model version, expected behaviour, actual behaviour and a screenshot. Do not
share patient-identifying data in screenshots.

## How a judge can test it

Use this order so the demonstration remains understandable:

1. Open the web app and show the research-use and consent wording.
2. Upload one clean retinal image using a patient reference code.
3. Show image quality before discussing the disease grade.
4. Show the five grade probabilities and explain that the highest calibrated
   probability chooses the grade.
5. Show the separate referable-DR score and the safe next step.
6. Point out that DME is not assessed and requires a separate validated model,
   OCT or clinical examination.
7. Open reviewer details and show model version, calibration and module status.
8. Run a poor-quality case to prove that unsafe input is rejected.
9. Run the MATLAB parity command and show 25/25 grade and referral agreement.
10. Open Simulink and explain the quality gate, grading, uncertainty and human
    review path.

If the judge supplies a new image, first ask whether it is a colour fundus
photograph and whether the team may process it for the demonstration. Record it
as a blind demo case. Do not claim correctness until a qualified reviewer
provides the reference grade.

## What counts as a pass

| Area | Pass condition |
|---|---|
| Artifact identity | ONNX SHA-256 matches the manifest |
| MATLAB parity | 100% grade and referral agreement on all fixed parity cases |
| Quality safety | Poor image produces no DR prediction |
| DR output | Exactly five probabilities that sum to approximately 1 |
| Referral | Uses the saved calibrated binary threshold |
| DME | Clearly unavailable; no fabricated score |
| Explainability | Correctly labels whether an attention method is available |
| Privacy | Patient code is used and another user cannot read the record |
| Traceability | Model version and runtime appear in reviewer details |
| Clinical wording | Says research screening support and requires human review |

## Current measured evidence

- ONNX size: **84.27 MiB** (88,366,785 bytes).
- PyTorch-to-ONNX maximum logit difference: **2.71e-6**.
- MATLAB parity: **25/25 grades and 25/25 referral decisions**.
- MATLAB test suite: **8/8 passed**.
- Low-memory cloud-compatible benchmark after warm-up: **508.2 ms median** total.
- Measured peak process memory: **314.52 MiB** after disabling transformer
  weight prepacking and ONNX memory-pattern caching.
- Source-validation referable DR: sensitivity **93.16%**, specificity **87.02%**.
- Source-validation five-grade QWK: **0.8592** and macro-F1 **0.6180**.

The accuracy values are source-validation results, not a final independent
clinical claim. Grade 1, Grade 3 and Grade 4 remain the harder classes.

## Hosting decision

An older deployment attempted to run FP32 V3.4 inside InsForge's 512 MiB compute
limit and returned HTTP 502 during real prediction. That configuration is no
longer the current architecture. The current website uses InsForge for auth,
storage and an authenticated proxy, while the V3.4 ONNX runtime runs in Azure
Container Apps with 1 vCPU and 2 GiB. Verify `/health`, `/model-card` and one
signed-in screening before each judge demonstration because cloud state can
change independently of this guide.

Use `compute/Dockerfile.v3_4` for Azure with at least **1 GiB RAM** and one CPU;
2 GiB is preferred for demonstrations and concurrent requests. A GPU is not
required for V3.4 ONNX inference. The default `compute/Dockerfile` remains the
verified local V3.4 fallback image.

An INT8 experiment was rejected: it reduced the ONNX file to about 24 MiB but
changed 2 of 25 grade decisions and shifted a grade probability by up to 11.4
percentage points. Do not deploy that quantized artifact.

## Verified Simulink scenario outputs

`run_simevents_scenarios` completed all seven executable SimEvents capacity
simulations and wrote `simulink/results/simevents/scenarios_simevents.json` and
`.csv`. The current outputs include:

| Scenario | Patients/day | Annual throughput | Average wait | Maximum queue |
|---|---:|---:|---:|---:|
| Baseline rural clinic | 51 | 12,750 | 3.79 min | 3 |
| Increased patient load | 68 | 17,000 | 81.47 min | 38 |
| Add a second camera | 105 | 26,250 | 2.42 min | 4 |
| District 100k annual | 508 | 127,000 | 10.78 min | 30 |

This simulation shows that image capture becomes the main bottleneck in the
tested higher-load setup: adding a camera improves throughput and waiting time
more than adding only a reviewer. Treat these as planning simulations based on
assumed arrival and service times, not measured hospital performance.

## If something fails

- **Model hash failure:** restore the exact ONNX file named in the manifest.
- **MATLAB cannot import ONNX:** run `check_toolboxes`, install Deep Learning
  Toolbox and the ONNX model import support package, then rerun `setup`.
- **Different MATLAB result:** stop the demo and rerun the full parity gate.
- **Cloud timeout after inactivity:** retry once; the service may be waking from
  scale-to-zero.
- **Cloud out of memory:** roll back to the previous image or increase the
  compute memory to 1 GB. Do not silently change preprocessing or quantize the
  model during a judged demonstration.
- **No cloud heatmap:** this is expected for the current V3.4 ONNX export. Use
  the local PyTorch V3.4 runtime when the experimental attention view is needed.

## What the team should say

“The same frozen V3.4 model runs through ONNX in Python and MATLAB. We verified
the software port on 25 fixed cases and obtained identical grade and referral
decisions. The model is still a research candidate: image quality can stop the
pipeline, every result requires human review, DME is not assessed, and final
clinical use needs locked external and prospective evaluation.”
