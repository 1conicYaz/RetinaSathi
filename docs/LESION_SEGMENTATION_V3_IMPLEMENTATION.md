# RetinaSathi lesion segmentation candidate

## Purpose

This module proposes pixel-level evidence for four IDRiD lesion classes:
microaneurysms, haemorrhages, hard exudates and soft exudates. It is separate
from the frozen V3.4 DR classifier. It cannot change the V3.4 grade or referral
decision, and it is not a diagnosis.

## Why the previous model was rejected

The previous whole-image model achieved mean Dice **0.0192** on its internal
validation split. A 4K retinal image was reduced to one 1024-pixel square, tiny
microaneurysms were lost, all classes used a fixed 0.50 threshold, and the loss
was dominated by background. That artifact remains historical evidence and is
not enabled in the application.

## Implemented pipeline

1. Keep the existing patient/image-level IDRiD train/validation manifest. The
   official 27-image segmentation test set remains locked.
2. Read the four official pixel masks at native resolution.
3. Train on 256 × 256 lesion-centred patches plus retinal-background patches.
4. Balance sampling across all four classes, including uncommon soft exudates.
5. Apply aligned flips, 90-degree rotations, brightness and contrast changes.
6. Use an ImageNet-pretrained MobileNetV3-small encoder with a U-Net decoder,
   then optimize with focal Tversky loss for severe foreground/background imbalance.
7. Select one threshold per lesion class on the frozen internal validation patches.
8. Run full images with overlapping native-resolution tiles and average overlap logits.
9. Convert masks into connected candidate regions with class, bounding box,
   centroid, area, mean probability and maximum probability.
10. Draw a four-colour overlay on the original image and keep the wording
    “candidate lesion evidence — clinician confirmation required.”

## Safety and release gate

- Poor image quality stops lesion inference and requests recapture.
- The model and manifest are bound by SHA-256 before use.
- A failed candidate cannot be exported by the release command.
- A smoke run never passes the release gate.
- The internal gate requires mean Dice ≥ 0.20 **and** class Dice of at least
  0.05 MA, 0.15 haemorrhages, 0.30 hard exudates and 0.15 soft exudates.
  Passing this engineering gate permits experimental integration only; it does
  not establish clinical validity.
- Official test evaluation, cross-runtime parity and independent Indian-camera
  validation are still required before any external claim.

## Current candidate result (25 September 2026)

The MobileNetV3-small U-Net run completed all 30 epochs. Epoch 27 was selected
from the frozen internal patch validation split with mean Dice **0.5216**.
It then passed the separate 11-image full-resolution overlap-tile engineering
gate with mean Dice **0.3738**:

| Lesion class | Full-image Dice |
|---|---:|
| Microaneurysms | 0.1958 |
| Haemorrhages | 0.3869 |
| Hard exudates | 0.5886 |
| Soft exudates | 0.3238 |

The exported candidate is `models/retinasathi-lesions-v3.onnx` (5.6 MiB), with
its thresholds, validation evidence and SHA-256 recorded in the adjacent
manifest. The official IDRiD test set was not used.

Runtime loading and overlay generation succeeded on a full-size image. Visual
review also found tile-edge and retinal-border false positives. For that reason,
the artifact remains `ready_experimental` and must not be presented as verified
lesion localization or enabled as a clinical feature. The next iteration must
improve overlap blending and border handling, repeat full-image validation, and
complete clinician and independent-camera evaluation.

## Hard-negative refinement result

The V3.2 refinement mined **1,643** high-confidence false candidates from all
43 training images, then fine-tuned the selected V3.1 checkpoint. Gaussian
overlap inference and a retinal-field safety margin were applied consistently
in validation and deployment. The selected epoch 3 checkpoint achieved a
full-image internal mean Dice of **0.3937** on the same 11-image validation set:

| Lesion class | Full-image Dice |
|---|---:|
| Microaneurysms | 0.2033 |
| Haemorrhages | 0.3453 |
| Hard exudates | 0.6553 |
| Soft exudates | 0.3708 |

This improves the previous full-image mean Dice of 0.3738. Runtime visual QA
also reduced the haemorrhage-labelled fraction on IDRiD_11 from 7.34% in the
original V3 candidate to 4.52% in V3.2. Repeated border candidates and optic-disc
false positives remain visible, so V3.2 remains an experimental artifact and is
not enabled as verified lesion evidence. A larger-context or centre-crop
inference design should be evaluated next rather than hiding the remaining
errors with display-only filtering.

## Commands

From the repository root:

```bash
.venv/bin/python -m ml.training.train_lesion_segmentation_v3 \
  --data-root "$RETINASATHI_DATA_ROOT" \
  --run-dir runs/20260925_lesions_v3_256_candidate
```

Keep each run under a new local `runs/` directory. The training script selects
MPS, CUDA, or CPU according to the available machine; inspect its logged
device and saved config before interpreting results.

Patch validation selects the checkpoint. Release requires a separate
full-resolution overlap-tile check:

```bash
.venv/bin/python -m ml.evaluate_lesion_full_images \
  --checkpoint runs/20260925_lesions_v3_256_candidate/best.pt \
  --data-root "$RETINASATHI_DATA_ROOT" \
  --output runs/20260925_lesions_v3_256_candidate/full_image_validation.json
```

After the full-image report passes its gate:

```bash
.venv/bin/python -m ml.export_segmentation \
  --checkpoint runs/20260925_lesions_v3_256_candidate/best.pt \
  --validation runs/20260925_lesions_v3_256_candidate/full_image_validation.json \
  --output models/retinasathi-lesions-v3.onnx \
  --version lesion-segmentation-v3
```

Enable MATLAB with `lesionModelPath` and `lesionManifestPath` in the app
config; see the [user guide](MATLAB_SIMULINK_USER_GUIDE.md). The current
protected cloud V3.4 release does not claim validated lesion output.

## Interpretation

The colour overlay answers **where the separate segmentation model sees
candidate lesion-shaped pixels**. It does not prove that the pixels are true
lesions, and it does not explain why the V3.4 classifier selected a grade.
Classifier explainability and lesion segmentation are different validation
problems and must stay visually and verbally separate.
