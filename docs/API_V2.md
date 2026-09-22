# Inference API V2

`POST /predict` accepts one JPEG, PNG, or WebP image of at most 15 MiB. The service returns HTTP errors for invalid media, oversized input, unreadable images, or an unavailable model.

## Stable envelope

Every successful V2 response contains:

```json
{
  "api_version": "2.0",
  "model_version": "idrid-mobilenetv3-multitask-v0.1",
  "quality": { "status": "heuristic" },
  "dr": { "status": "baseline_v1", "confidence_calibrated": null },
  "dme": { "status": "baseline_v1" },
  "structures": { "status": "not_trained" },
  "lesions": { "status": "not_trained", "items": [] },
  "explainability": { "status": "baseline_feature_activation" },
  "recommendation": { "requires_human_review": true },
  "runtime": { "processing_mode": "cloud" }
}
```

The complete runtime values are typed in `src/lib/screenings.ts`. Existing flat V1 fields remain temporarily where their meaning is unchanged so the deployed client can transition safely.

## Status semantics

| Status | Meaning |
|---|---|
| `ready` | Trained and evaluated for the stated experimental use |
| `candidate` | Trained on development data but not yet locked/tested; not for claims |
| `baseline_v1` | Preserved baseline with documented limitations |
| `heuristic` | Deterministic logic, not a learned model |
| `not_trained` | Interface exists but no result is produced |
| `data_limited` | Required verified labels are unavailable |
| `unavailable` | Expected runtime dependency is unavailable |

`baseline_feature_activation` is explicitly not Grad-CAM and not a lesion map. Lesion results stay empty until a genuine mask-trained model passes its acceptance tests.

## Compatibility and safety

- `model_version`, `dr_grade`, `dme_risk`, `confidence`, `referable_dr`, and `explanation_image` remain available for the V1 client.
- The recommendation text is present as `recommendation.text`; storage persists that text only.
- No attention region is stored in `screening_lesions` because attention is not lesion evidence.
- New clients must inspect each module's `status` before displaying a result.

## Ungradeable images

When the hard quality gate labels an image `poor`, inference stops before the ONNX classifier. The response remains a successful V2 safety result so the operator receives capture reasons and retake guidance, but `dr_grade`, `dme_risk`, and `confidence` are `null`; their module statuses are `unavailable`; probability arrays and attention regions are empty; and `explainability.method` is `not_run_quality_gate`. A poor image must never be presented as a clinical grade.

For a V2 candidate, local PyTorch mode reports `gradcam_autograd` and uses true gradients. The lightweight ONNX mode reports `gradcam_exact_linear_gap` only for a global-average-pooling backbone with an identity post-pool transform, where the exported analytic class weights are tested to be mathematically equivalent to Grad-CAM up to the normalization constant. V1 remains explicitly `channel_mean_feature_activation_not_gradcam`.

Optional learned modules are loaded only when their versioned ONNX paths and manifests are configured. Lesion output contains a separate four-class experimental overlay and per-class predicted pixel fractions; vessel output contains its own overlay; localization returns padded-frame normalized coordinates and relative uncertainty-derived confidence. Missing models return `not_trained` and never fabricate coordinates or masks. These outputs are omitted entirely when the quality gate stops inference.

Calibrated classifier confidence is routed separately from the disease endpoint. Confidence below 0.60 returns `uncertainty.label = high` and `recommendation.urgency = review`; it does not change `referable_dr` to true. Uncalibrated V1 and quality-gated results report uncertainty as `unavailable`. Every path still requires human review.

## Structured errors

HTTP errors use `detail.code`, `detail.message`, and `detail.next_action`. Current codes include `INVALID_FILE`, `CORRUPT_IMAGE`, `IMAGE_TOO_LARGE`, `MODEL_UNAVAILABLE`, and `INFERENCE_FAILED`. Client-side routing additionally surfaces `LOCAL_MODEL_UNAVAILABLE`, `NETWORK_UNAVAILABLE`, `STORAGE_FAILED`, and the explicit `SYNC PENDING` state.
