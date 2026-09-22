# RetinaSathi V3.4 — PyTorch, ONNX and MATLAB Parity Report

Execution date: 2026-09-22
MATLAB: R2026a Update 5 on Apple Silicon
Scope: engineering interoperability, not clinical validation

## Outcome

The frozen V3.4 seed-26038 checkpoint was exported to a fixed-input ONNX graph,
imported by MATLAB, and executed behind native MATLAB retinal preprocessing and
the saved calibration/referral policy.

| Gate | Result |
|---|---:|
| MATLAB safety and V3.4 contract tests | 8/8 passed |
| PyTorch vs ONNX max absolute logit difference | `2.71e-6` |
| MATLAB grade agreement on parity set | 25/25 (100%) |
| MATLAB referral agreement on parity set | 25/25 (100%) |
| End-to-end good referable case | Grade 2, referable, DME not assessed |
| End-to-end good non-referable case | Grade 0, non-referable, DME not assessed |
| End-to-end poor-quality case | Inference stopped as ungradeable |
| First MATLAB import plus inference | 11.36 s |
| Warm MATLAB inference | 0.636 s |
| V3.4 MATLAB app and report | Hidden render smoke passed |

The first native attempt used MATLAB's Gaussian kernel and achieved 24/25 grade
agreement. Investigation showed that Pillow's `GaussianBlur` uses three passes
of fractional extended-box filters. `pillowGaussianBlur.m` now reproduces that
algorithm in MATLAB. The corrected implementation achieved 25/25 grade and
referral agreement.

## Frozen artifact contract

- Model: partially fine-tuned DINOv2-S/14 with referable, ordinal and nominal heads.
- Checkpoint SHA-256: `0407cbf0402e9e791307045e8d42b38ad188bec19b5ddc58f454d14f3f677f39`.
- ONNX SHA-256: `8d5373211b8651e1a3c787a19bfd52a64383d8a2296f370cbf2615467422b2ec`.
- Input: float32 NCHW `[1, 3, 392, 392]`.
- Outputs: one referable logit, four ordinal logits and five nominal logits.
- Temperatures and referral threshold come from the hash-bound validation report.
- MATLAB verifies the ONNX SHA-256 from the manifest before first import.
- Final grade distribution is `0.5 × ordinal + 0.5 × nominal`.
- DME is not assessed.

The ONNX model is 84 MB and remains ignored by Git. The small portable manifest
at `models/retinasathi-v3-4.manifest.json` records hashes, preprocessing,
calibration, foundation provenance and limitations.

## MATLAB preprocessing contract

`preprocessV34Fundus.m` performs:

1. RGB conversion.
2. Visible-retina crop using the same threshold and 1% padding as Python.
3. Centered black square padding without stretching.
4. Lanczos resize to 392 × 392.
5. Ben Graham enhancement.
6. Pillow-compatible three-pass extended-box Gaussian approximation.
7. ImageNet RGB normalization.

Across the 25 cases, the normalized-tensor mean absolute difference averaged
`0.00649` and was at most `0.01160` per case. The largest individual normalized
pixel difference was `0.5480`. Differences remain because MATLAB and Pillow use
different Lanczos implementations. Final decisions still matched on every case.

The maximum observed MATLAB-preprocessing vs Python-preprocessing logit changes
were `0.4666` for the referable head, `0.5305` for the ordinal head and `0.3908`
for the nominal head. These are documented engineering differences; they are not
presented as proof of broader robustness.

## Parity cohort

The local ignored parity folder contains 25 images:

- five prepared behavior cases: good non-referable, referable, severe,
  uncertain and synthetic poor quality;
- twenty deterministic development-pool APTOS images, four from each mapped
  grade 0–4.

This cohort checks framework and preprocessing behavior. It must not be used to
claim accuracy, external validation or clinical performance. Licensed images,
reference tensors and MATLAB-generated adapters stay outside Git.

## Reproduction

```bash
.venv/bin/python scripts/export_v3_4_onnx.py \
  --checkpoint runs/v3_4_seed_26038_mps_progress_20260920_gpu/best.pt \
  --validation runs/v3_4_seed_26038_mps_progress_20260920_gpu/final_validation.json \
  --output models/retinasathi-v3-4.onnx \
  --manifest models/retinasathi-v3-4.manifest.json

.venv/bin/python scripts/build_v3_4_matlab_parity_fixture.py \
  --onnx models/retinasathi-v3-4.onnx \
  --output-dir runs/v3_4_matlab_parity \
  --manifest runs/v3_data/v3_initial_manifest.csv \
  --data-root "$RETINASATHI_DATA_ROOT" --per-grade 4 \
  runs/demo_cases/good_nonreferable.jpg \
  runs/demo_cases/referable.jpg runs/demo_cases/severe.jpg \
  runs/demo_cases/uncertain.jpg runs/demo_cases/poor_quality.jpg
```

```matlab
cd matlab
setup
results = runtests("tests");
report = runV34Parity("../runs/v3_4_matlab_parity", ...
    "../models/retinasathi-v3-4.onnx", ...
    "../models/retinasathi-v3-4.manifest.json", ...
    "../runs/v3_4_matlab_parity/matlab_parity_report.json");
assert(all([results.Passed]));
assert(report.passed);
```

## Remaining limits

- The 25-case parity set is an engineering gate, not an independent clinical set.
- V3.4 source validation used DeepDRiD and is not prospective Indian clinical validation.
- Exact grade remains weaker than referable screening; Grade 4 recall is a known weakness.
- The MATLAB V3.4 attention map has not been validated. The UI must not show an
  unverified map or describe attention as lesion localization.
- The public cloud remains the lightweight V1 backup. V3.4 is intended for a
  controlled local clinician-review demonstration with mandatory human review.
