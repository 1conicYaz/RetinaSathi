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

The lesion module now points to the V3.2 experimental four-class ONNX candidate
and its hash-bound manifest. It uses native-resolution tiles, Gaussian overlap
blending, the frozen retinal-border margin, class-specific thresholds and
connected-component regions. V3.2 passed the internal engineering gate with
full-image mean Dice 0.3937, but visual QA still finds border and optic-disc
false positives. MATLAB therefore labels every overlay experimental and keeps
it separate from V3.4's grade and referral decision. The baseline ONNX feature
activation is not presented as MATLAB Grad-CAM. See `TOOLBOX_STATUS.md` for the
current machine audit.

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

To launch the complete local prototype with the current artifacts:

```matlab
cd('/path/to/X-Retina/matlab')
launchV34LesionPrototype
```

To configure it manually, add both lesion artifact paths:

```matlab
cfg.lesionModelPath = "../models/retinasathi-lesions-v3-2.onnx";
cfg.lesionManifestPath = "../models/retinasathi-lesions-v3-2.manifest.json";
result = runRetinaPipelineV34("../runs/demo_cases/referable.jpg",cfg);
launchRetinaSathiApp(cfg)
```

The **Lesion evidence** image tab then shows candidate masks in four colours:
microaneurysms, haemorrhages, hard exudates and soft exudates. The app labels
them experimental and requires ophthalmologist confirmation. A poor-quality
image stops both grading and lesion analysis.

The app presents the result in two levels:

- **What this means** shows the referral decision, suggested DR grade, image
  quality, five-grade probability chart, a plain-language explanation and the
  next clinical action.
- **Technical status** shows which research modules are available. DME,
  lesion, vessel, optic-disc, fovea and attention outputs stay out of the main
  clinical summary when they have not been validated.

The referral score and grade confidence are intentionally shown separately.
Referral score answers whether specialist review is recommended. Grade
confidence is the largest of the five grade probabilities and describes
certainty about the exact severity level; it is not model accuracy.

The **Enhanced retinal view** and **Exact model input** are preprocessing
views, not heatmaps. V3.4 square-pads the crop and applies Ben Graham-style
contrast enhancement; the neutral-grey padding and coloured border halo in the
exact input are expected consequences of that frozen preprocessing contract.
The display-only enhanced view masks those padding artefacts without changing
the array sent to the model.

The **Proposed explainability** tab documents the future validation gate. It
does not create a patient-specific attention map. A map should be enabled only
after faithfulness, stability, lesion-mask comparison, clinician-usefulness and
Python/ONNX/MATLAB parity checks pass.

The **Future model roadmap** tab separates planned work from current evidence.
It covers governed multi-dataset and Indian-camera expansion, larger-context
segmentation, multi-task classification, teacher comparison, conditional
knowledge distillation, independent student calibration, MATLAB parity,
district-capacity simulation and prospective ophthalmologist evaluation. The
full technical sequence is in `FUTURE_MODEL_DEVELOPMENT.md`.

Accept the MATLAB port only after every fixed case has the same grade and
referral decision. Keep tensor and logit differences in the generated report.
The current V3.4 model does not assess DME, and MATLAB attention-map equivalence
has not yet been validated.

After the local-only ONNX and parity fixtures exist, the complete noninteractive
gate is one command:

```bash
./scripts/verify_v3_4_matlab.sh
```
