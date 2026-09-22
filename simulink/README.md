# RetinaSathi SimEvents Workflow

`RetinaSathiScreeningWorkflow.slx` is an executable SimEvents R2026a model.
One entity represents one screening visit. The generated model contains:

- seeded exponential patient arrivals;
- capture and AI queues;
- capacity-limited camera, AI, and reviewer servers;
- a quality gate with one recapture loop;
- network-transfer delay;
- reproducible referable, uncertain, and routine AI routing;
- FIFO or priority clinical review selected by `priorityReview`;
- separate routine and reviewed outcomes;
- live queue displays/scopes and exported time-series statistics.

The model uses planning assumptions from `parameters.m`. It does not run the
V3.4 ONNX image classifier for every simulated patient. The AI server models
the measured workflow time and routes entities using configured rates. Use the
MATLAB screening app to demonstrate real-image ONNX inference.

## Run one visible scenario

```matlab
cd('/path/to/X-Retina/simulink')
p = parameters();
model = build_retinasathi_workflow(p);
open_system(model)
set_param(model,'SimulationCommand','start')
```

For a slower judge demonstration that opens all queue scopes:

```matlab
model = launch_simevents_demo(20);
set_param(model,'SimulationCommand','start')
```

At pacing rate 20, the 480-minute clinic day takes approximately 24 seconds.

Open the three queue scopes or watch **Live Total Queue** and **Live Completed**.

## Run and export one scenario

```matlab
metrics = run_simevents_workflow(parameters(), "results/simevents");
```

## Run all seven scenarios

```matlab
results = run_simevents_scenarios("results/simevents");
disp(struct2table(results))
```

The verified output is `results/simevents/scenarios_simevents.csv` and JSON.
The baseline produced 51 arrivals and 51 completions. Increased load produced
a maximum capture queue of 38; adding a second camera increased completions
from 68 to 105. The district configuration completed 508 visits in a ten-hour
simulation, equivalent to 127,000 visits across 250 identical operating days.
These are engineering estimates, not observed hospital outcomes.

`simulateWorkflow.m` and `run_scenarios.m` remain as an independent base-MATLAB
reference implementation. Their random process and recapture assumptions are
not identical to the SimEvents implementation, so exact totals are not
expected to match.
