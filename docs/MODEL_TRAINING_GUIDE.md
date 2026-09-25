# Model training and evidence guide

This guide explains where RetinaSathi's model work lives and how a teammate can reproduce a **new research run**. Training is separate from using the [MATLAB screening window](MATLAB_SIMULINK_USER_GUIDE.md) or the public web app. Those use frozen artifacts.

## What was trained

| Version or module | Purpose | Current interpretation |
|---|---|---|
| V1 MobileNetV3 | Small initial DR/DME web baseline | Historical engineering baseline. |
| V2 EfficientNet-B3 | Controlled five-grade DR experiment | Locked IDRiD and Messidor results exposed source shift; do not use its numbers as V3.4 evidence. |
| V3.1–V3.3 | EfficientNet, frozen DINOv2, and RETFound comparisons | Research candidates used to select the next experiment. |
| V3.4 partial DINOv2-S/14 | Dedicated referable-DR head plus ordinal and five-grade outputs | Current frozen research candidate. Three adaptation seeds were studied; seed 26038 is the selected runtime identity. |
| V3.2 four-class lesion U-Net | Pixel candidate masks for four IDRiD lesion classes | Optional **experimental** MATLAB overlay; internal full-image mean Dice 0.3937 on 11 images, with visible false positives. |
| Quality, vessel, and landmark experiments | Capture checks and anatomical research | Separately evaluated. They are not validated V3.4 clinical outputs. |

Read [model selection](MODEL_SELECTION_AND_APPROACH_REPORT.md), [V3.4 results](V3_4_RESULTS.md), [seed stability](V3_4_THREE_SEED_STABILITY.md), and [lesion report](LESION_SEGMENTATION_V3_IMPLEMENTATION.md) before comparing numbers.

## Files and data

```text
configs/     experiment settings and declared thresholds
ml/datasets/ input parsing, labels, and patient-aware grouping
ml/models/   classifier and segmentation networks
ml/training/ training entry points
ml/evaluation/metrics.py and scripts/audit_v3_candidate.py
             evaluation and candidate gates
models/*.manifest.json
             small artifact identity and preprocessing records
runs/        local outputs and checkpoints; excluded from Git
```

Dataset images, licensed annotations, checkpoints, and ONNX binaries are not in the repository. Use the providers' terms and the [data strategy](DATASET_STRATEGY.md), [data card](V3_DATA_CARD.md), and [inventory](DATASET_INVENTORY.md). Set `RETINASATHI_DATA_ROOT` to your authorized local dataset directory. Never upload raw retinal data or patient information to a public Git repository.

## Research run sequence

1. **Prepare the environment.** Use Python 3.11 and install `ml/requirements.txt` in a virtual environment.
2. **Audit the data.** Run `scripts/build_v3_data_manifest.py` on the authorized data root. Write its new manifest, audit, and data card under a local `runs/` directory so published evidence is not overwritten.
3. **Choose a declared config.** The V3.4 settings are in `configs/classifier_v3_4.yaml`; lesion settings are in `configs/lesions_v3_2.yaml`. A changed config is a new experiment, not a reproduction of the frozen candidate.
4. **Train on development data.** `scripts/train_classifier_v3_4.sh` launches the classifier; `ml.training.train_lesion_segmentation_v3` launches the lesion model. V3.4 depends on its documented starting checkpoint and data manifest. Keep patient groups and held-out roles intact.
5. **Evaluate before export.** Use the validation and candidate-audit scripts. Do not use the official test partition for tuning. Report sensitivity, specificity, per-grade recall, calibration, and failure cases. For lesions, run full-image overlap-tile evaluation after patch validation and inspect border and optic-disc false positives.
6. **Freeze and export.** Export only a selected, hash-bound checkpoint with its matching validation record. The ONNX file stays out of Git; commit only safe manifests and aggregate reports.
7. **Check MATLAB parity.** Compare the frozen ONNX and native MATLAB preprocessing/calibration before claiming the two runtimes agree. The [parity report](V3_4_MATLAB_PARITY_REPORT.md) explains what was checked.

For command options, use `--help` on the relevant Python entry point and inspect the config. Training uses substantial local compute and requires datasets and starting weights that a fresh clone cannot supply.

## Minimal command map

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r ml/requirements.txt
export RETINASATHI_DATA_ROOT=/absolute/path/to/authorized/datasets

.venv/bin/python scripts/build_v3_data_manifest.py --help
./scripts/train_classifier_v3_4.sh --help
.venv/bin/python -m ml.training.train_lesion_segmentation_v3 --help
.venv/bin/python -m ml.evaluate_lesion_full_images --help
.venv/bin/python scripts/audit_v3_candidate.py --help
```

The **published V3.4 scores are evidence from the frozen historical run**. A new run may produce different results and needs its own manifest, checkpoint hash, evaluation report, and status. The clinical question remains open until an independent intended-camera Indian cohort and prospective workflow evaluation are completed.
