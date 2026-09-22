# RetinaSathi SimEvents Workflow Results

These are reproducible planning estimates from the executable SimEvents R2026a
model in `simulink/RetinaSathiScreeningWorkflow.slx`. They are not clinical
outcomes or measurements from a hospital.

One entity represents one screening visit. The model executes patient arrivals,
capture queues, camera service, one quality recapture, network transfer, AI
service, referable/uncertain/routine routing, optional priority review, reviewer
service and final outcomes. The 1.2-minute AI station allowance includes image
handling and preprocessing; measured warm V3.4 MATLAB inference alone was 0.636
seconds.

| Scenario | Arrivals | Completed | Annual equivalent | Capture wait | Review wait | Max total queue | Camera util. | Reviewer util. | Recaptures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline rural clinic | 51 | 51 | 12,750 | 3.67 min | 0.32 min | 3 | 71.7% | 32.1% | 6 |
| Network constrained | 51 | 49 | 12,250 | 3.67 min | 0.32 min | 4 | 71.7% | 31.7% | 6 |
| Increased patient load | 108 | 68 | 17,000 | 81.39 min | 0.25 min | 38 | 98.5% | 38.5% | 9 |
| Additional camera | 108 | 105 | 26,250 | 1.69 min | 1.84 min | 4 | 74.0% | 63.5% | 12 |
| Additional reviewer | 108 | 68 | 17,000 | 81.39 min | 0.00 min | 38 | 98.5% | 19.2% | 9 |
| High-referral priority review | 108 | 73 | 18,250 | 1.69 min | 66.86 min | 32 | 74.0% | 96.4% | 12 |
| District 100k configuration | 521 | 508 | 127,000 | 10.50 min | 0.04 min | 30 | 96.0% | 60.5% | 59 |

Annual equivalent multiplies completed visits by 250 identical operating days.
The district scenario uses a ten-hour operating period; the other scenarios use
eight hours.

## What these results show

- At increased load, the camera is the bottleneck: utilization reaches 98.5%,
  the capture wait rises to 81.39 minutes, and 40 visits remain unfinished.
- Adding a second camera cuts the maximum total queue from 38 to 4 and increases
  completions from 68 to 105 under the same arrival stream.
- Adding a reviewer alone does not help that scenario because patients are
  waiting for capture, not review.
- In the high-referral scenario, the reviewer becomes the bottleneck: reviewer
  utilization reaches 96.4% and the review queue reaches 32.
- The district configuration exceeds the 100,000 annual planning target, but
  its 127,000 estimate must not be presented as observed field capacity.

## What the AI block represents

The SimEvents AI server models AI station capacity and service time. Its routing
decisions are reproducible samples from the configured referable and uncertainty
rates. It does not load a retinal image and execute the ONNX model for every
synthetic patient. Real-image V3.4 inference is demonstrated in the MATLAB app
and parity pipeline.

## Reproduce the result

```matlab
cd('/path/to/X-Retina/simulink')
results = run_simevents_scenarios("results/simevents");
disp(struct2table(results))
```

Portable evidence is stored in:

- `simulink/results/simevents/scenarios_simevents.csv`
- `simulink/results/simevents/scenarios_simevents.json`

The older `simulateWorkflow.m` remains an independent base-MATLAB reference.
Its random process and recapture assumptions differ, so its exact totals are not
expected to match the SimEvents model.
