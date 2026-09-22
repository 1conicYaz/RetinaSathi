"""Leakage-free IDRiD lesion and DRIVE vessel datasets."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as functional


LESION_PATHS = {
    "microaneurysms": ("1. Microaneurysms", "MA"),
    "haemorrhages": ("2. Haemorrhages", "HE"),
    "hard_exudates": ("3. Hard Exudates", "EX"),
    "soft_exudates": ("4. Soft Exudates", "SE"),
}


def manifest_ids(path: Path, split: str) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [row["image_id"] for row in csv.DictReader(handle) if row["split"] == split]


class SegmentationDataset(Dataset):
    def __init__(self, data_root: Path, manifest: Path, split: str, task: str, image_size: int, augment: bool) -> None:
        self.data_root, self.ids, self.task, self.image_size, self.augment = data_root, manifest_ids(manifest, split), task, image_size, augment

    def __len__(self) -> int: return len(self.ids)

    def _idrid(self, image_id: str) -> tuple[Image.Image, list[Image.Image]]:
        base = self.data_root / "IDRiD/original_extracted/A. Segmentation"
        image = Image.open(base / f"1. Original Images/a. Training Set/{image_id}.jpg").convert("RGB")
        masks = []
        for folder, suffix in LESION_PATHS.values():
            path = base / f"2. All Segmentation Groundtruths/a. Training Set/{folder}/{image_id}_{suffix}.tif"
            masks.append(Image.open(path).convert("L") if path.exists() else Image.new("L", image.size))
        return image, masks

    def _drive(self, image_id: str) -> tuple[Image.Image, list[Image.Image]]:
        base = self.data_root / "DRIVE/training"
        image = Image.open(base / f"images/training/images/{image_id}_training.tif").convert("RGB")
        mask = Image.open(base / f"masks/training/1st_manual/{image_id}_manual1.gif").convert("L")
        fov = Image.open(base / f"masks/training/mask/{image_id}_training_mask.gif").convert("L")
        return image, [Image.fromarray((np.asarray(mask) * (np.asarray(fov) > 0)).astype(np.uint8))]

    def __getitem__(self, index: int):
        image_id = self.ids[index]; image, masks = self._idrid(image_id) if self.task == "lesions" else self._drive(image_id)
        if self.augment and random.random() < 0.5:
            image = functional.hflip(image); masks = [functional.hflip(mask) for mask in masks]
        if self.augment:
            angle = random.uniform(-10, 10); image = functional.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            masks = [functional.rotate(mask, angle, interpolation=InterpolationMode.NEAREST) for mask in masks]
        rgb = np.asarray(image); visible = rgb.max(axis=2) > 8
        rows = np.flatnonzero(visible.any(axis=1)); columns = np.flatnonzero(visible.any(axis=0))
        if rows.size and columns.size:
            padding = round(min(rgb.shape[:2]) * 0.01)
            box = (max(0,int(columns[0])-padding),max(0,int(rows[0])-padding),min(rgb.shape[1],int(columns[-1])+padding+1),min(rgb.shape[0],int(rows[-1])+padding+1))
            image=image.crop(box);masks=[mask.crop(box) for mask in masks]
        square_size=max(image.size);offset=((square_size-image.width)//2,(square_size-image.height)//2)
        image_square=Image.new("RGB",(square_size,square_size));image_square.paste(image,offset);image=image_square
        padded_masks=[]
        for mask in masks:
            canvas=Image.new("L",(square_size,square_size));canvas.paste(mask,offset);padded_masks.append(canvas)
        masks=padded_masks
        image = functional.resize(image, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        masks = [functional.resize(mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST) for mask in masks]
        tensor = functional.normalize(functional.to_tensor(image), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        target = torch.stack([(functional.pil_to_tensor(mask)[0] > 0).float() for mask in masks])
        return tensor, target, image_id


def tiled_prediction(model, image: torch.Tensor, tile_size: int = 512, overlap: int = 128) -> torch.Tensor:
    """Overlap-tile inference preserving microaneurysm-scale resolution."""
    if image.ndim != 4 or image.shape[0] != 1: raise ValueError("tiled_prediction expects one BCHW image")
    stride = tile_size-overlap; height, width = image.shape[-2:]
    output = None; weights = torch.zeros((1, 1, height, width), device=image.device)
    for top in list(range(0, max(height-tile_size, 0)+1, stride)) + ([height-tile_size] if height > tile_size and (height-tile_size)%stride else []):
        for left in list(range(0, max(width-tile_size, 0)+1, stride)) + ([width-tile_size] if width > tile_size and (width-tile_size)%stride else []):
            tile = image[:, :, top:top+tile_size, left:left+tile_size]; prediction = model(tile)
            if output is None: output = torch.zeros((1, prediction.shape[1], height, width), device=image.device)
            output[:, :, top:top+tile_size, left:left+tile_size] += prediction; weights[:, :, top:top+tile_size, left:left+tile_size] += 1
    if output is None: return model(image)
    return output / weights.clamp_min(1)
