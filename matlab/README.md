# RetinaSathi MATLAB Layer

This is an executable MATLAB implementation of the retinal pipeline, not a UI mock. Run `setup`, then `check_toolboxes`, and call `runRetinaPipeline(imagePath, config)`. The pipeline crops the fundus, enhances luminance, evaluates deterministic capture safety, stops clinical-looking inference when recapture is required, and imports the ONNX classifier when Deep Learning Toolbox supports the exported graph.

```matlab
cd matlab
setup
status = check_toolboxes()
cfg = struct("modelPath", "../compute/artifacts/idrid_multitask.onnx");
result = runRetinaPipeline("/path/to/fundus.jpg", cfg);
launchRetinaSathiApp(cfg)
```

The Python side has genuine experimental lesion/vessel/localization artifacts, but the MATLAB layer reports those modules unavailable unless a compatible MATLAB network artifact is explicitly supplied. The baseline ONNX feature activation is not presented as MATLAB Grad-CAM. See `TOOLBOX_STATUS.md` for the current machine audit.

`importNetworkFromONNX` can generate a model-named `+package` of MathWorks adapter classes beside this folder. That machine-generated package is ignored and should be regenerated from the privately transferred ONNX model rather than committed.

## V3.4 candidate path

V3.4 is trained and frozen in PyTorch, exported as a fixed batch-1 `392 x 392`
ONNX model, and processed natively in MATLAB. It is a research candidate and
does not replace the older verified baseline until the MATLAB parity gate passes.

From the repository root:

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
  --data-root /path/to/SIH26038_Datasets \
  --per-grade 4 \
  runs/demo_cases/good_nonreferable.jpg \
  runs/demo_cases/referable.jpg \
  runs/demo_cases/severe.jpg \
  runs/demo_cases/uncertain.jpg \
  runs/demo_cases/poor_quality.jpg
```

Then run in MATLAB:

```matlab
cd matlab
setup
tests = runtests("tests");
report = runV34Parity("../runs/v3_4_matlab_parity", ...
    "../models/retinasathi-v3-4.onnx", ...
    "../models/retinasathi-v3-4.manifest.json", ...
    "../runs/v3_4_matlab_parity/matlab_parity_report.json");
result = runRetinaPipelineV34("../runs/demo_cases/referable.jpg");
cfg = struct("modelGeneration", "v3.4", ...
    "modelPath", "../models/retinasathi-v3-4.onnx", ...
    "manifestPath", "../models/retinasathi-v3-4.manifest.json");
launchRetinaSathiApp(cfg)
```

Accept the MATLAB port only after every fixed case has the same grade and
referral decision. Keep tensor and logit differences in the generated report.
The current V3.4 model does not assess DME, and MATLAB attention-map equivalence
has not yet been validated.

After the local-only ONNX and parity fixtures exist, the complete noninteractive
gate is one command:

```bash
./scripts/verify_v3_4_matlab.sh
```
