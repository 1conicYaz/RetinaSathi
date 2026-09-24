# RetinaSathi / X-Retina

RetinaSathi is a MathWorks-integrated diabetic-retinopathy screening-support
research prototype for Smart India Hackathon problem statement SIH26038. It
combines a frozen retinal AI model, reproducible MATLAB execution, an executable
Simulink/SimEvents healthcare-workflow model, mandatory human review and a
supporting authenticated web/cloud demonstration.

> Research screening support only. RetinaSathi is not a diagnosis or a
> clinically approved medical device.

## SIH26038 — MathWorks problem statement

SIH26038 asks for a MATLAB-based pipeline covering image quality, retinal
structure and lesion analysis, ICDR Grade 0–4, referable-DR sensitivity above
90% and specificity above 85%, clinically meaningful explainability and
Simulink capacity planning for rural programs serving more than 100,000 people
annually.

[Read the requirement-by-requirement status](docs/SIH26038.md). RetinaSathi
meets the referral targets on retrospective source validation and implements
the MATLAB and SimEvents foundations. Validated lesion-level explainability,
complete retinal structure analysis and prospective clinical evidence remain
future work.

## Problem

Rural screening is constrained by variable portable-camera image quality,
limited connectivity and specialist capacity. A useful system must reject
unsafe inputs, prioritize referable or uncertain cases and help planners
understand camera, compute, network and reviewer bottlenecks.

## Proposed solution

~~~text
Fundus camera
    → image quality / recapture
    → RetinaSathi V3.4 AI
         ├─ dedicated referable-DR score
         └─ ICDR Grade 0–4 estimate
    → MATLAB-compatible calibrated decision contract
    → mandatory human review / referral

The same clinic assumptions feed an executable SimEvents capacity model.
The React + InsForge + Azure application is the supporting deployment interface.
~~~

Poor-quality images receive no DR result. DME, lesions, vessels, optic disc,
fovea and V3.4 explainability are reported as unavailable rather than inferred
from an unvalidated output.

## MathWorks architecture

### MATLAB implementation

MATLAB imports the frozen 84.27 MiB V3.4 ONNX graph and reproduces retinal
cropping, square padding, 392 px Ben Graham enhancement, normalization,
calibration, thresholding and grade/referral postprocessing.

Verified release evidence:

- 8 MATLAB tests passed.
- 25/25 Python-reference cases matched grade and referral decisions.
- The manifest binds model identity, SHA-256, preprocessing and threshold.

Parity demonstrates implementation consistency. It is not clinical validation.

### Simulink / SimEvents workflow

~~~text
arrival → capture queue → camera → quality/recapture → network transfer
        → AI queue → decision routing → routine outcome or clinical review
~~~

One entity is one screening visit. The model measures queue depth, waiting time,
utilization, throughput and bottlenecks across seven scenarios. It does not run
ONNX for every synthetic entity.

Verified release evidence:

- 14/14 SimEvents software checks passed.
- Under the configured high-load scenario, adding a second camera increased
  completions from 68 to 105 and reduced maximum queue from 38 to 4.
- The district scenario estimated 127,000 visits across 250 identical operating
  days.

Every capacity number is a **simulation**, not observed field throughput.

![SimEvents workflow](docs/images/retinasathi_simevents_workflow.png)

[Run and interpret MATLAB/SimEvents](docs/MATLAB_SIMEVENTS.md).

## AI model — V3.4

The selected candidate partially adapts the final two blocks of DINOv2-S/14 and
uses three heads: dedicated referable DR, ordinal severity and direct five-class
grading. The release was selected after controlled EfficientNet-B3, frozen
DINOv2 and frozen RETFound comparisons.

Three seeded fine-tuning runs from the same V3.2 initialization produced:

| Metric | Three-seed mean | Range |
|---|---:|---:|
| Referable sensitivity | 93.89% | 92.22–95.00% |
| Referable specificity | 90.61% | 89.55–91.36% |
| AUROC | 0.9781 | 0.9756–0.9807 |
| AUPRC | 0.9763 | 0.9745–0.9787 |
| Quadratic weighted kappa | 0.8247 | 0.7910–0.8635 |
| Macro-F1 | 0.5666 | 0.5339–0.6017 |

These are patient-separated DeepDRiD **source-validation** results. They are not
prospective clinical evidence. Grade 2 varies most between seeds and Grade 4
recall remains weak; V3.4 is stronger for referral screening than autonomous
exact grading.

[Model rationale](docs/MODEL_SELECTION_AND_APPROACH_REPORT.md) ·
[selected-seed results](docs/V3_4_RESULTS.md) ·
[three-seed stability](docs/V3_4_THREE_SEED_STABILITY.md)

## Deployment and web demonstration

~~~text
React interface → InsForge authentication → authenticated server function
                → protected Azure Container Apps → V3.4 ONNX Runtime
                → private screening record and required human review
~~~

- Demo: <https://69exmaqk.insforge.site/>
- Runtime identity: **classifier-v3.4-seed26038**
- Input: 392 × 392 RGB after the frozen preprocessing contract
- Referral threshold: **0.2073261738**
- Azure configuration at verification: minimum 0, maximum 1 replica

Scale-to-zero reduces idle cost but can add cold-start delay. The current
one-replica release has not demonstrated a 5,000-case/two-minute workload.
[Deployment verification](docs/AZURE_DEPLOYMENT_VERIFICATION.md).

## Dataset strategy

V3.4 development used APTOS 2019, IDRiD and patient-grouped DeepDRiD under the
recorded manifest policy. EyePACS was not used. Its archived copy was never
admitted to the verified V3 pipeline.

The current EyePACS decision is **EXPERIMENT FIRST**: verify competition terms,
archive integrity, subject grouping, labels, quality and cross-source duplicates
before a controlled development-only comparison. Dataset images never belong in
this repository. [Read the dataset and EyePACS decision](docs/DATASET_STRATEGY.md).

## Repository structure

| Path | Purpose |
|---|---|
| **matlab/** | Quality, preprocessing, ONNX inference, calibration, app and tests |
| **simulink/** | Executable SimEvents model, scenarios and verification |
| **ml/** | Dataset, model, training, calibration and evaluation code |
| **compute/** | FastAPI/ONNX inference service and contract tests |
| **src/** | Supporting React screening and review interface |
| **functions/**, **migrations/** | InsForge proxy and protected data contract |
| **configs/** | Reproducible experiment definitions |
| **models/** | Safe manifests only; binary weights are excluded |
| **artifacts/** | Small aggregate evidence and provenance records |
| **docs/** | Requirements, evidence, runbooks and research roadmap |

Raw datasets, retinal photographs, credentials, checkpoints, ONNX binaries,
generated MATLAB packages and local experiment outputs are excluded from Git.

## Quick start

### Web and Python checks

~~~bash
cp .env.example .env.local
npm ci
python3.11 -m venv .venv
.venv/bin/python -m pip install -r compute/requirements.txt
./scripts/run_tests.sh
~~~

Run the local stack:

~~~bash
./scripts/run_full_stack_local.sh
~~~

### MATLAB V3.4 parity

Place the hash-matching model at **models/retinasathi-v3-4.onnx**, then:

~~~bash
./scripts/verify_v3_4_matlab.sh
~~~

### SimEvents

~~~matlab
cd('simulink')
report = verify_simevents_workflow(true, "results/simevents");
~~~

## Current status and limitations

| Area | Status |
|---|---|
| V3.4 referral and Grade 0–4 | Implemented research candidate |
| MATLAB ONNX execution and parity | Verified software implementation |
| SimEvents patient-flow model | Verified simulation software |
| Authenticated web/Azure demonstration | Implemented; verify live revision before demo |
| Image-quality gate | Implemented, partly heuristic |
| DME and retinal structure/lesion outputs | Unavailable in V3.4 |
| V3.4 explainability | Disabled pending technical and clinician validation |
| Indian intended-camera multi-site evaluation | Not performed |
| Prospective clinical validation/regulatory readiness | Not established |
| Encrypted edge/offline pilot | Future research |

## Research roadmap

Future work follows **Research → Hypothesis → Experiment → Validation →
Integration**. Priorities are ophthalmologist workflow research, independent
Indian intended-camera evaluation, quality/OOD safety, validated explainability,
exact-grade improvement, controlled EyePACS experiments, knowledge distillation
and secure offline/edge operation.

[Read the staged roadmap](docs/ROADMAP.md).

## Documentation

[Documentation index](docs/README.md) · [demo runbook](DEMO.md) ·
[architecture](docs/ARCHITECTURE.md) · [judge Q&A](docs/SIH_JUDGE_QA.md)
