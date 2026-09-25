# MATLAB and Simulink/SimEvents

MATLAB and Simulink/SimEvents are core SIH26038 components. The web application
is a supporting interface for the same screening workflow.

## MATLAB image path

```text
Fundus image
  → quality gate and recapture decision
  → retinal crop and square padding
  → 392 px Ben Graham enhancement
  → ImageNet normalization
  → imported V3.4 ONNX network
  → referral, ordinal and five-class logits
  → saved calibration and referral threshold
  → Grade 0–4, referral state and required human review
```

MATLAB imports the frozen ONNX graph as a `dlnetwork`. It verifies the expected
artifact hash and reproduces preprocessing and postprocessing rather than
training a second model. MathWorks documents `importNetworkFromONNX` as the
supported ONNX-to-`dlnetwork` route:
<https://www.mathworks.com/help/deeplearning/ref/importnetworkfromonnx.html>.

Run one image:

```matlab
cd('/path/to/RetinaSathi/matlab')
setup
cfg = struct( ...
    "modelGeneration", "v3.4", ...
    "modelPath", "../models/retinasathi-v3-4.onnx", ...
    "manifestPath", "../models/retinasathi-v3-4.manifest.json" ...
);
result = runRetinaPipelineV34("../runs/demo_cases/referable.jpg", cfg);
disp(result)
launchRetinaSathiApp(cfg)
```

The ONNX file is intentionally excluded from Git. Its hash must match the
manifest before a result is presented as V3.4.

## MATLAB verification

```bash
./scripts/verify_v3_4_matlab.sh
```

Verified release evidence:

- 8 MATLAB tests passed.
- 25/25 reference cases matched Python grade and referral decisions.
- V3.4 warm inference measured approximately 0.636 seconds per image on the
  tested Mac environment.

Parity shows that two runtimes implement the same frozen decision contract. It
does not provide new patients, independent accuracy evidence or clinical
validation.

## SimEvents operational model

One entity represents one screening visit. It moves through:

```text
arrival → capture queue → camera → quality/recapture → network transfer
        → AI queue → V3.4 service-time block → decision routing
        → routine completion or priority clinical-review queue
```

Queues represent waiting. Servers represent limited resources and service
time. Routing blocks represent recapture, routine, referable and uncertain
paths. This follows the documented SimEvents queue/server abstraction for
discrete-event capacity analysis:
<https://www.mathworks.com/help/simevents/queuing-and-service.html>.

The AI block uses a measured/configured service-time distribution. It does not
execute ONNX for every synthetic visit. The simulation measures throughput,
waiting time, queue depth and utilization; model accuracy comes from the
separate evaluation pipeline.

## Run and inspect the simulation

```matlab
cd('/path/to/RetinaSathi/simulink')
report = verify_simevents_workflow(true, "results/simevents");
results = run_simevents_scenarios("results/simevents");
disp(struct2table(results))
open_system('RetinaSathiScreeningWorkflow')
```

Verified release evidence: 14/14 software checks passed across seven scenarios.
The district scenario completed 508 simulated visits in ten hours, equivalent
to 127,000 visits over 250 identical days. These are simulation outputs based
on declared assumptions, not observed clinical throughput.

## Current limits

- V3.4 does not assess DME, vessels, optic disc or fovea. A separate V3.2 four-class lesion candidate can be shown in the MATLAB prototype, but it is experimental and does not determine V3.4 referral.
- V3.4 explanation maps remain disabled after failing technical validation.
- The quality gate is partly heuristic.
- MATLAB parity does not prove clinical accuracy.
- SimEvents capacity does not prove hospital throughput.
