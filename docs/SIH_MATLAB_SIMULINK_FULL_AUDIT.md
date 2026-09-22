# RetinaSathi MATLAB, Simulink and SIH26038 complete explanation

**Updated:** 22 September 2026
**Purpose:** Help a new teammate run, interpret, test and present the MATLAB and
SimEvents work without making unsupported clinical claims.

> RetinaSathi is a research screening-support prototype. Software verification
> does not make it a clinically validated medical device.

## 1. What the two screenshots mean

### Scope screenshot

The horizontal axis is simulated clinic time in minutes. The baseline ends at
480 minutes, representing one eight-hour operating day. The vertical axis is a
queue length, not accuracy, confidence or DR grade.

- `0` means nobody is waiting in that queue.
- `1` means one screening visit is waiting.
- A vertical jump means the queue changed at that event time.
- Yellow circles are displayed event samples because markers are enabled.
- `T=478.2745` means the last displayed event occurred near the 480-minute end.
- A mostly 0/1 AI or review queue is expected in the baseline because those
  resources are not saturated.

To demonstrate a visible queue bottleneck, use the increased-load scenario. Its
capture queue reaches 38, making the graph much easier to explain.

### Full Simulink screenshot

The connected blocks are the actual executable patient-flow model. The model
is displayed at 36% zoom because the full workflow is wide. The star after the
model name means a display/runtime setting such as zoom or pacing has changed
since the last save; it does not mean the simulation failed.

The small labels have standard SimEvents meanings:

| Label | Meaning |
|---|---|
| `d` | Number of entities that departed a block |
| `n` | Number currently waiting/stored |
| `w` | Average waiting time |
| `l` | Average queue length |
| `util` | Fraction of simulated time a server is occupied |
| `E` / entity connection | Event-driven patient entity path |

## 2. Three different systems that must not be confused

### MATLAB retinal pipeline

This runs a real retinal image through V3.4 preprocessing, ONNX inference,
calibration and referral logic. It answers: **What does the frozen AI model
output for this image?**

### Simulink/SimEvents workflow

This simulates visits moving through cameras, recapture, upload, AI capacity and
human review. It answers: **How many people can the service handle, where will
queues form, and which resource should be added?**

The SimEvents AI server uses measured/configured processing time and configured
route rates. It does not load a different image and execute ONNX for every
synthetic patient.

### Web application

This is the operator/reviewer product interface with upload, history, privacy,
reporting and local/cloud routing. MATLAB does not need to run inside the web
browser.

## 3. SimEvents workflow under the hood

```text
Patient generator
  → Capture merge
  → Capture queue
  → Fundus camera server
  → Quality gate
      ├─ poor quality → Recapture server → Capture merge
      └─ accepted
          → Network transfer server
          → AI queue
          → V3.4 AI service-time server
          → AI decision router
              ├─ routine → Routine outcome
              ├─ uncertain ┐
              └─ referable ┴→ Clinical review queue
                                  → Human review server
                                  → Reviewed outcome
```

One **entity** represents one screening visit. The patient image itself is not
stored inside the entity.

### Patient arrivals

The Entity Generator uses a seeded exponential inter-arrival process. The seed
makes the scenario reproducible. The baseline arrival rate is `0.10/minute`, or
an average of one arrival every ten minutes.

### Capture queue and camera server

The queue stores patients waiting for a fundus camera. Camera capacity is the
number of cameras. Service time represents positioning, capture and basic
handling, not only shutter time.

### Quality gate and recapture

The configured poor-quality rate sends some entities through one recapture
cycle. The real MATLAB image pipeline separately performs an actual image
quality assessment and withholds DR output when the image is ungradeable.

### Network transfer

Transfer delay is calculated from:

```text
network overhead + (image size in megabits / bandwidth in megabits per second)
```

The baseline uses a 5 MB image, 10 Mbps bandwidth and fixed workflow overhead,
giving 1.5 minutes. The constrained scenario uses 0.1 Mbps and produces 8.1
minutes per image.

### AI queue and inference server

The AI station allowance is 1.2 minutes. It includes preprocessing, device
handling and workflow buffer; the neural network alone is faster. The number of
AI devices controls simultaneous capacity.

### AI routing and priority

Configured rates produce routine, referable and uncertain cases. Within the
referable group, a configured severe-case fraction receives the highest entity
priority. Other referable cases are next, uncertain cases follow, and routine
cases bypass human review. These are workflow assumptions, not synthetic
medical diagnoses.

### Clinical review

The queue is FIFO in ordinary scenarios. When `priorityReview=true`, it becomes
a priority queue. Reviewer count and review time determine specialist capacity.

### Outcomes and statistics

Routine and reviewed outcomes are counted separately. The runner exports:

- arrivals and completions;
- annual-equivalent throughput;
- capture, AI and review waiting time;
- estimated total screening time;
- maximum queue lengths;
- camera, AI and reviewer utilization;
- automatic bottleneck name;
- referable, uncertain and recapture counts;
- image size, bandwidth and network contribution;
- unfinished visits at the stop time.

## 4. Essential terminology

| Term | Layman meaning |
|---|---|
| Entity | One simulated screening visit |
| Arrival rate | How frequently patients reach the clinic |
| Inter-arrival time | Time between two arriving patients |
| Queue | Patients waiting for a resource |
| Server | A resource that spends time processing a patient |
| Capacity | Number processed simultaneously |
| Service time | Time one resource occupies one patient |
| Recapture rate | Fraction needing another photograph |
| Throughput | Completed screenings in the simulated period |
| Waiting time | Time spent in queues |
| Total screening time | Estimated queue plus service time from capture to outcome |
| Utilization | Percentage of available resource time that is busy |
| Bottleneck | Resource with the highest utilization/queue pressure |
| Bandwidth | Network transfer capacity in Mbps |
| Priority queue | Urgent entities are reviewed before lower-priority ones |
| Seed | Number that makes random-looking events reproducible |
| Stop time | End of the simulated clinic period |
| VariableStepDiscrete | Solver mode that jumps between discrete events |
| Scope | Time graph of one statistic |
| Calibration | Converting model scores into more reliable probabilities |
| Sensitivity | Fraction of true referable cases correctly referred |
| Specificity | Fraction of true non-referable cases correctly kept non-referable |
| Grad-CAM/attention | Regions influencing a model, not confirmed lesions |
| Parity | Two runtimes produce matching decisions from the same frozen model |

## 5. Exact testing procedure

### A. Start cleanly

In MATLAB:

```matlab
close all force
clear functions
clearvars
clc
```

### B. Verify installed products

```matlab
cd('/path/to/X-Retina/matlab')
setup
status = check_toolboxes();
disp(status.SimEvents)
```

Pass condition:

```text
installed: 1
licenceAvailable: 1
```

### C. Test the real V3.4 image model

```matlab
result = runRetinaPipelineV34("../runs/demo_cases/referable.jpg");
disp(result.quality)
disp(result.dr)
disp(result.dme)
```

Expected:

- a gradeable image returns five grade probabilities;
- probabilities sum to approximately one;
- referral uses the frozen calibrated binary threshold;
- DME says not assessed;
- every output requires human review.

Test the safety gate:

```matlab
poor = runRetinaPipelineV34("../runs/demo_cases/poor_quality.jpg");
disp(poor)
```

Expected: quality is poor and DR inference is withheld with recapture guidance.

### D. Run MATLAB tests and parity

From Terminal:

```bash
cd "/path/to/X-Retina"
./scripts/verify_v3_4_matlab.sh
```

Expected evidence:

- 8/8 MATLAB tests pass;
- 25/25 grade decisions match the Python reference;
- 25/25 referral decisions match;
- ONNX hash verification passes.

Parity proves software consistency on fixed engineering cases. It does not
prove clinical accuracy on a new hospital population.

### E. Run the automatic SimEvents verifier

```matlab
cd('/path/to/X-Retina/simulink')
report = verify_simevents_workflow(true, "results/simevents");
```

Expected final line:

```text
PASS: 14/14 SimEvents software checks.
```

This validates installation, licence, required blocks, accounting, utilization,
bandwidth behavior, bottleneck output, total screening time, seven scenarios,
camera resource sensitivity and the 100k district target.

### F. Run the understandable live demo

Baseline:

```matlab
model = launch_simevents_demo(20);
set_param(model,'SimulationCommand','start')
```

The full eight-hour day takes about 24 seconds at pacing rate 20.

Visible bottleneck example:

```matlab
p = parameters();
p.name = "increased_patient_load";
p.arrivalRatePerMinute = 0.22;
model = launch_simevents_demo(20, p);
set_param(model,'SimulationCommand','start')
```

Explain that the capture queue increases because one camera cannot keep up.

### G. Open the scenario dashboard

```matlab
results = readtable("results/simevents/scenarios_simevents.csv", "TextType", "string");
plot_simevents_dashboard(results, "../docs/images/retinasathi_simevents_results.png");
```

## 6. How to know whether the simulation passed

| Check | Expected result |
|---|---|
| Required executable blocks | 13/13 found |
| Baseline accounting | routine + reviewed = completed |
| Utilization | Every value between 0% and 100% |
| Baseline bandwidth | 1.5 minutes/image |
| Constrained bandwidth | 8.1 minutes/image |
| Baseline bottleneck | Camera |
| Increased-load completed | 68 |
| Increased-load max queue | 38 |
| Additional-camera completed | 105 |
| Additional-camera max queue | 4 |
| Priority scenario bottleneck | Human reviewer |
| District annual equivalent | 127,000, above the 100k planning target |

If an additional camera does not improve the high-load scenario, or the
district scenario falls below 100,000 without an explained parameter change,
stop and investigate before presenting.

## 7. Current scenario interpretation

| Scenario | What it proves |
|---|---|
| Baseline | Normal rural load completes without a backlog |
| Network constrained | Low bandwidth increases transfer time and unfinished cases |
| Increased load | One camera saturates and becomes the bottleneck |
| Additional camera | A second camera raises completion from 68 to 105 and cuts queue 38 to 4 |
| Additional reviewer | More reviewers do not solve a camera bottleneck |
| Priority review | High referral demand saturates the reviewer, exposing specialist capacity limits |
| District 100k | Configured resources produce 127,000 annual-equivalent completions |

The 127,000 value means the model completed 508 visits in a ten-hour simulated
day and multiplied by 250 identical operating days. It does not mean 127,000
real people were screened.

## 8. SIH26038 requirement audit

The requirement wording is based on the preserved project source and the
public SIH26038 archive. The team must still compare it with the exact current
official SIH portal/template before submission.

| Requirement | Current evidence | Status | Remaining gap |
|---|---|---|---|
| Quality assessment and enhancement | MATLAB quality gate, cropping, square padding and V3.4 Ben Graham preprocessing; poor input withholds prediction | Implemented, partly heuristic | Validate focus/illumination/FOV thresholds on portable Indian cameras |
| Grade 0–4 | Frozen V3.4 three-head DINOv2 path in Python, ONNX and MATLAB | Implemented | Harder exact grades remain variable; needs independent validation |
| Referable sensitivity >90%, specificity >85% | V3.4 seed 26038 source validation: 93.16% sensitivity and 87.02% specificity; three-seed mean also exceeds targets | Meets source-validation target | Not yet proven prospectively or on independent Indian multi-camera data |
| Calibration and uncertainty | Frozen temperatures, binary threshold and human-review routing | Implemented | Clinical threshold validation remains pending |
| Optic disc/fovea | Experimental localization artifacts exist outside the V3.4 MATLAB app | Partial | Integrate and externally validate before presenting as clinical evidence |
| Vessel segmentation | Experimental local model exists | Partial | Small internal validation; not integrated into V3.4 MATLAB result |
| Microaneurysm/exudate/haemorrhage lesions | Lesion experiment was measured and disabled because quality was inadequate | Failed/disabled honestly | Train a reliable native-resolution lesion model with valid masks |
| Neovascularization | Interface/status only | Not trained | Explicit labels and ophthalmologist-verified evaluation required |
| DME | V3.4 correctly reports not assessed | Missing by design | Separate validated DME head and clinical/OCT confirmation pathway |
| Grad-CAM/attention | Local Python influence evidence exists; MATLAB V3.4 attention parity is withheld | Partial | Validate transformer explanation in MATLAB and conduct clinician usefulness study |
| Annotated report under 30 seconds | Printable screening-support report exists | Implemented technically | Timed ophthalmologist usability evaluation is not completed |
| Simulink acquisition/queues/resources | Executable Entity Generator, queues, servers, recapture and outcomes | Implemented |
| Bandwidth/image size | Explicit 5 MB, Mbps and overhead model; constrained scenario verified | Implemented | Replace assumptions with field measurements |
| Processing throughput | AI devices, service time and throughput exported | Implemented | Load-test actual deployment service under concurrency |
| Review capacity and priority | Reviewer count/time plus severe/referable/uncertain priorities | Implemented as simulation | Validate routing policy with ophthalmologists |
| Bottleneck and total time | Automatically exported | Implemented |
| 100,000+ annual planning | District scenario verifies 127,000 annual equivalent | Implemented as planning evidence | Pilot measurements required before operational claims |
| MATLAB-based pipeline | Native MATLAB preprocessing, ONNX inference, calibration, app and tests | Implemented |
| Published benchmark superiority | Mixed: V3.4 source validation is strong, but independent benchmark/clinical superiority is not established | Not complete | Locked external V3.4 evaluation and prospective multi-site study |

## 9. What is still missing and priority order

### Critical before claiming clinical readiness

1. Prospective evaluation using Indian portable-camera images from multiple
   clinics with ophthalmologist reference grades.
2. Independent V3.4 external evaluation frozen before viewing results.
3. A reliable lesion segmentation model, especially for tiny microaneurysms.
4. Clinician-rated explanation usefulness and report review time.
5. Fairness analysis by camera, site, age, sex and relevant population groups.

### Important for stronger SIH alignment

1. Integrate validated optic-disc, fovea and vessel outputs into the MATLAB app.
2. Build and validate a separate DME pathway; do not infer DME from the DR grade.
3. Obtain neovascularization labels or continue displaying `not trained`.
4. Measure real camera time, retake rate, image size, bandwidth and reviewer time
   at a clinic, then replace simulation assumptions.
5. Run actual Azure V3.4 deployment concurrency, memory and latency tests.

### Presentation work

1. Replace old simulation numbers in every slide with the verified SimEvents
   CSV and dashboard.
2. Show one poor-quality MATLAB case, one referable case and the SimEvents
   increased-load versus additional-camera comparison.
3. Keep “attention” separate from “lesion segmentation.”
4. Say “127,000 annual-equivalent planning estimate,” never “127,000 patients
   screened.”

## 10. Judge demonstration sequence

1. State the rural problem and screening-only boundary.
2. Run one clean retinal image in the MATLAB app.
3. Explain quality, Grade 0–4, separate referable probability and human review.
4. Run a poor-quality image and show that inference stops.
5. Show the parity evidence: 25/25 grade and referral agreement.
6. Open the SimEvents model and explain that one entity is one visit.
7. Run the increased-load scenario and show the camera queue growing.
8. Show the dashboard comparison: adding a camera changes 68 completions to 105
   and reduces the queue from 38 to 4.
9. Show the district 127,000 annual-equivalent scenario and state the assumptions.
10. End with limitations and the clinical-validation plan.

## 11. Questions judges may ask

**Why use MATLAB if training was done in PyTorch?**
Training used the research ecosystem required by DINOv2. The frozen graph was
exported to ONNX. MATLAB independently reproduces preprocessing, executes the
same frozen inference graph, verifies its hash, applies calibration and matches
the Python decisions on fixed engineering cases.

**Why use Simulink?**
Model accuracy alone does not reveal how many cameras, AI workers or reviewers
a district needs. SimEvents exposes operational queues and bottlenecks before
resources are purchased.

**Does the SimEvents AI block diagnose DR?**
No. It models processing capacity and configured routing proportions. Real
image inference is demonstrated in MATLAB and the application.

**Why is the scope switching between zero and one?**
It is showing the number waiting. The baseline queue is short because capacity
is adequate. The increased-load scenario produces a much larger capture queue.

**Does the heatmap detect lesions?**
No. It shows model influence. A validated lesion segmentation model must be a
separate output.

**Has the >90% sensitivity target been met?**
It was met on V3.4 source validation. It has not yet been proven in prospective
multi-camera Indian clinical deployment, so the team must state both facts.

**Is the 100k target real?**
It is an executable planning scenario based on declared assumptions. It proves
the model can calculate a resource configuration, not that the program has
already screened that population.

**What would make this clinically deployable?**
Locked external testing, multi-site prospective evaluation, camera-specific
quality validation, clinician-reviewed explanations, fairness testing,
regulatory assessment and monitored pilot operation.

## 12. Safe one-minute explanation for teammates

> MATLAB tests the real frozen retinal model on images. SimEvents tests the
> service around that model. Each SimEvents entity is one screening visit. It
> waits for a camera, may require recapture, experiences network and AI time,
> and may enter a priority ophthalmologist queue. We compare seven scenarios to
> find bottlenecks. Under high load, one camera completes 68 visits and creates
> a queue of 38; adding a camera completes 105 and reduces the queue to 4. The
> district configuration produces a 127,000 annual-equivalent planning estimate.
> These are operational simulations, while the V3.4 MATLAB pipeline performs
> real image inference. The prototype still needs independent clinical and
> lesion-level validation.
