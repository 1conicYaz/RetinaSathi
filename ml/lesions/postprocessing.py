"""Shared lesion thresholding, connected regions and visualization helpers."""

from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

LESION_COLORS = [(255, 65, 54), (255, 149, 0), (255, 214, 10), (175, 82, 222)]


def gaussian_importance_map(size: int, sigma_scale: float = .125, minimum: float = 1e-3) -> np.ndarray:
    """Weight tile centres more than unreliable tile edges during reconstruction."""
    if size < 1 or sigma_scale <= 0:
        raise ValueError("size and sigma_scale must be positive")
    coordinates = np.arange(size, dtype=np.float32) - (size - 1) / 2
    sigma = size * sigma_scale
    axis = np.exp(-.5 * (coordinates / sigma) ** 2)
    weights = np.outer(axis, axis)
    weights /= weights.max()
    return np.maximum(weights, minimum).astype(np.float32)


def retinal_field_mask(rgb: np.ndarray, border_margin_fraction: float = .015) -> np.ndarray:
    """Return the visible retinal field with an interior safety margin.

    The margin removes camera-border reflections from candidate overlays. It is
    deliberately geometric and must also be used during validation.
    """
    field = np.asarray(rgb).max(axis=2) > 12
    if not field.any() or border_margin_fraction <= 0:
        return field
    margin = max(1, round(min(field.shape) * border_margin_fraction))
    return distance_transform_edt(field) > margin


def thresholds_from_manifest(manifest: dict[str, object], classes: list[str]) -> np.ndarray:
    value = manifest.get("thresholds", manifest.get("threshold", .5))
    if isinstance(value, dict):
        return np.asarray([float(value.get(name, .5)) for name in classes], dtype=np.float32)
    if isinstance(value, (list, tuple)):
        if len(value) != len(classes):
            raise ValueError("Lesion threshold count does not match class count")
        return np.asarray(value, dtype=np.float32)
    return np.full(len(classes), float(value), dtype=np.float32)


def _components(mask: np.ndarray, minimum_pixels: int) -> list[dict[str, int]]:
    mask = np.asarray(mask, dtype=bool)
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    regions: list[dict[str, int]] = []
    for y, x in np.argwhere(mask):
        if visited[y, x]:
            continue
        queue = deque([(int(y), int(x))]); visited[y, x] = True
        xs: list[int] = []; ys: list[int] = []
        while queue:
            cy, cx = queue.popleft(); xs.append(cx); ys.append(cy)
            for ny, nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny,nx] and not visited[ny,nx]:
                    visited[ny,nx] = True; queue.append((ny,nx))
        if len(xs) >= minimum_pixels:
            regions.append({"x":min(xs),"y":min(ys),"width":max(xs)-min(xs)+1,"height":max(ys)-min(ys)+1,"area_pixels":len(xs)})
    return sorted(regions, key=lambda item: item["area_pixels"], reverse=True)


def extract_lesion_regions(
    probabilities: np.ndarray,
    classes: list[str],
    thresholds: np.ndarray,
    minimum_pixels: int = 3,
    maximum_regions_per_class: int = 50,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    if probabilities.ndim != 3 or probabilities.shape[0] != len(classes):
        raise ValueError("Expected lesion probabilities in CHW order")
    masks = probabilities >= thresholds[:, None, None]
    regions: list[dict[str, object]] = []
    for channel, name in enumerate(classes):
        for region in _components(masks[channel], minimum_pixels)[:maximum_regions_per_class]:
            x, y, width, height = (region[key] for key in ("x","y","width","height"))
            values = probabilities[channel, y:y+height, x:x+width][masks[channel, y:y+height, x:x+width]]
            regions.append({"type":name, **region, "mean_probability":round(float(values.mean()),5), "max_probability":round(float(values.max()),5)})
    return masks, regions


def overlay_lesions(image: Image.Image, masks: np.ndarray, colors: list[tuple[int,int,int]] = LESION_COLORS) -> Image.Image:
    display = image.convert("RGB")
    rgb = np.asarray(display, dtype=np.float32)
    tint = np.zeros_like(rgb); alpha = np.zeros(rgb.shape[:2], dtype=np.float32)
    for channel, color in enumerate(colors[: masks.shape[0]]):
        resized = Image.fromarray((masks[channel] * 255).astype(np.uint8)).resize(display.size, Image.Resampling.NEAREST)
        selected = np.asarray(resized) > 0
        tint[selected] = color; alpha[selected] = .58
    return Image.fromarray(np.clip(rgb*(1-alpha[...,None])+tint*alpha[...,None],0,255).astype(np.uint8))
