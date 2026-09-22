"""Fundus-preserving preprocessing shared by training and local inference."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def crop_fundus(image: Image.Image, threshold: int = 8, padding_fraction: float = 0.01) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    visible = rgb.max(axis=2) > threshold
    rows = np.flatnonzero(visible.any(axis=1)); columns = np.flatnonzero(visible.any(axis=0))
    if not rows.size or not columns.size:
        return image.convert("RGB")
    padding = round(min(rgb.shape[:2]) * padding_fraction)
    top = max(0, int(rows[0]) - padding); bottom = min(rgb.shape[0], int(rows[-1]) + padding + 1)
    left = max(0, int(columns[0]) - padding); right = min(rgb.shape[1], int(columns[-1]) + padding + 1)
    return Image.fromarray(rgb[top:bottom, left:right])


def pad_square(image: Image.Image, fill: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    size = max(image.size)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def ben_graham_enhance(image: Image.Image, sigma_fraction: float = 0.035) -> Image.Image:
    """Local contrast enhancement used in retinal pipelines, bounded to uint8."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    sigma = max(1.0, min(image.size) * sigma_fraction)
    blurred = np.asarray(image.convert("RGB").filter(ImageFilter.GaussianBlur(sigma)), dtype=np.float32)
    enhanced = np.clip(4.0 * rgb - 4.0 * blurred + 128.0, 0, 255).astype(np.uint8)
    return Image.fromarray(enhanced)


def _clip_histogram(histogram: np.ndarray, limit: int) -> np.ndarray:
    clipped = np.minimum(histogram, limit)
    excess = int((histogram - clipped).sum())
    clipped += excess // len(clipped)
    clipped[: excess % len(clipped)] += 1
    return clipped


def clahe_enhance(image: Image.Image, tiles: int = 8, clip_factor: float = 2.0) -> Image.Image:
    """Dependency-free contrast-limited adaptive equalization on luminance."""
    ycbcr = np.asarray(image.convert("YCbCr")).copy()
    luminance = ycbcr[:, :, 0]
    output = np.empty_like(luminance)
    row_edges = np.linspace(0, luminance.shape[0], tiles + 1, dtype=int)
    column_edges = np.linspace(0, luminance.shape[1], tiles + 1, dtype=int)
    for row in range(tiles):
        for column in range(tiles):
            ys = slice(row_edges[row], row_edges[row + 1]); xs = slice(column_edges[column], column_edges[column + 1])
            tile = luminance[ys, xs]
            histogram = np.bincount(tile.ravel(), minlength=256)
            limit = max(1, round(clip_factor * tile.size / 256))
            cumulative = _clip_histogram(histogram, limit).cumsum()
            mapping = np.clip(255 * cumulative / max(cumulative[-1], 1), 0, 255).astype(np.uint8)
            output[ys, xs] = mapping[tile]
    ycbcr[:, :, 0] = output
    return Image.fromarray(ycbcr, "YCbCr").convert("RGB")


def gray_world_normalize(image: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    means = rgb.reshape(-1, 3).mean(axis=0)
    target = float(means.mean())
    scaled = rgb * (target / np.maximum(means, 1e-6))[None, None, :]
    return Image.fromarray(np.clip(scaled, 0, 255).astype(np.uint8))


def preprocess_fundus(
    image: Image.Image,
    image_size: int,
    enhancement: str = "ben_graham",
    color_normalization: bool = False,
) -> Image.Image:
    cropped = crop_fundus(image)
    square = pad_square(cropped)
    resized = square.resize((image_size, image_size), Image.Resampling.LANCZOS)
    if color_normalization:
        resized = gray_world_normalize(resized)
    if enhancement == "ben_graham":
        return ben_graham_enhance(resized)
    if enhancement == "clahe":
        return clahe_enhance(resized)
    if enhancement == "none":
        return resized
    raise ValueError(f"Unknown enhancement: {enhancement}")
