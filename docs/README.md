# Documentation guide

The [main README](../README.md) is the shortest overview and setup path. This page helps you find a detailed guide without reading every file.

## Choose your path

| If you want to… | Start with | Then read |
|---|---|---|
| Understand the project with no ML background | [Complete project explanation](COMPLETE_PROJECT_EXPLANATION.md) | [Main README](../README.md) |
| Try the web and local candidate | [Main README](../README.md#run-the-project) | [V3.4 app integration](V3_4_CLINICAL_PILOT_AND_APP_INTEGRATION.md) |
| Follow a screening from upload to report | [Architecture](ARCHITECTURE.md) | [API contract](API_V2.md), [deployment verification](AZURE_DEPLOYMENT_VERIFICATION.md) |
| Check SIH problem coverage | [SIH26038 alignment](SIH26038.md) | [Research roadmap](ROADMAP.md) |
| Understand the model and training | [Model training guide](MODEL_TRAINING_GUIDE.md) | [Model selection report](MODEL_SELECTION_AND_APPROACH_REPORT.md), [data card](V3_DATA_CARD.md) |
| Inspect measured results and limits | [V3.4 results](V3_4_RESULTS.md) | [Three-seed stability](V3_4_THREE_SEED_STABILITY.md), [deployment verification](AZURE_DEPLOYMENT_VERIFICATION.md) |
| Inspect lesion segmentation | [Lesion implementation](LESION_SEGMENTATION_V3_IMPLEMENTATION.md) | [MATLAB prototype](../matlab/README.md) |
| Understand datasets and licensing | [Dataset strategy](DATASET_STRATEGY.md) | [Dataset inventory](DATASET_INVENTORY.md), [data validation](DATA_VALIDATION.md) |
| Run MATLAB | [Step-by-step MATLAB and Simulink user guide](MATLAB_SIMULINK_USER_GUIDE.md) | [MATLAB README](../matlab/README.md), [parity report](V3_4_MATLAB_PARITY_REPORT.md) |
| Run clinic simulation | [Step-by-step MATLAB and Simulink user guide](MATLAB_SIMULINK_USER_GUIDE.md#5-open-the-simulink-clinic-workflow) | [Simulink README](../simulink/README.md), [simulation results](SIMULATION_RESULTS.md) |
| Prepare a demo | [MATLAB and Simulink user guide](MATLAB_SIMULINK_USER_GUIDE.md) | [SIH26038 alignment](SIH26038.md) |
| Understand clinical next steps | [Pilot plan](V3_4_CLINICAL_PILOT_AND_APP_INTEGRATION.md) | [External validation protocol](EXTERNAL_VALIDATION_PROTOCOL.md) |
| Check the cloud release | [Azure deployment verification](AZURE_DEPLOYMENT_VERIFICATION.md) | [Research roadmap](ROADMAP.md) |

## How to interpret the files

- **Current V3.4 candidate:** files named `V3_4_*`, the [data card](V3_DATA_CARD.md), [deployment verification](AZURE_DEPLOYMENT_VERIFICATION.md), and [complete project explanation](COMPLETE_PROJECT_EXPLANATION.md). These describe the research candidate, cloud release, and measured limits.
- **Executable components:** [`matlab/`](../matlab/) and [`simulink/`](../simulink/) have their own READMEs. Web and API code live in [`src/`](../src/) and [`compute/`](../compute/).
- **Historical evidence:** [Model selection](MODEL_SELECTION_AND_APPROACH_REPORT.md) traces V1/V2 and V3.0–V3.3 decisions. Their metrics belong to those versions.
- **Presentation material:** use the [user guide](MATLAB_SIMULINK_USER_GUIDE.md) for a runnable demonstration and [SIH26038 alignment](SIH26038.md) for requirement status. For performance claims, use evaluation reports.
- **Images:** web screenshots are in [`playwright/evidence/`](playwright/evidence/); the supplied V3.4 MATLAB prototype screenshot and SimEvents diagrams are in [`images/`](images/). The MATLAB lesion overlay shows experimental candidates, not confirmed lesions.

The [public website](https://69exmaqk.insforge.site) was [verified on 24 September](AZURE_DEPLOYMENT_VERIFICATION.md) with a protected V3.4 ONNX cloud runtime. V3.4 also has a local engineering path. Its retrospective DeepDRiD results, MATLAB parity, and SimEvents capacity estimates answer different questions. None alone establishes safe clinical performance in Indian screening settings.
