"""Non-clinical spatial sanity metrics for attention and real lesion masks."""

from __future__ import annotations

import numpy as np


def attention_mask_metrics(heatmap: np.ndarray, lesion_mask: np.ndarray, retina_mask: np.ndarray) -> dict[str, float | bool | None]:
    if heatmap.shape != lesion_mask.shape or heatmap.shape != retina_mask.shape:
        raise ValueError("Heatmap, lesion mask, and retina mask must share a shape")
    heatmap = np.clip(heatmap.astype(np.float64), 0.0, 1.0)
    lesion = lesion_mask.astype(bool); retina = retina_mask.astype(bool)
    retinal_values = heatmap[retina]
    threshold = float(np.quantile(retinal_values, 0.80)) if retinal_values.size else 1.0
    attention = (heatmap >= threshold) & retina
    intersection = int((attention & lesion).sum()); union = int((attention | lesion).sum())
    heat_total = float(heatmap.sum())
    maximum = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    has_lesion = bool(lesion.any())
    return {
        "lesion_present": has_lesion,
        "top_20pct_attention_lesion_iou": float(intersection / union) if union else 0.0,
        "pointing_game_hit": bool(lesion[maximum]) if has_lesion else None,
        "attention_inside_retina_fraction": float(heatmap[retina].sum() / heat_total) if heat_total else 0.0,
        "attention_outside_retina_fraction": float(heatmap[~retina].sum() / heat_total) if heat_total else 0.0,
    }
