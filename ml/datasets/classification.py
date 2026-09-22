"""Licensed-local manifest datasets for APTOS and IDRiD classification."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset


class ManifestFundusDataset(Dataset):
    def __init__(self, manifest: Path, image_root: Path, dataset: str, split: str, transform: Callable) -> None:
        self.image_root = image_root
        self.dataset = dataset
        self.transform = transform
        with manifest.open(encoding="utf-8", newline="") as handle:
            self.rows = [row for row in csv.DictReader(handle) if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No {dataset} rows for split {split!r} in {manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def image_path(self, image_id: str) -> Path:
        suffix = ".png" if self.dataset == "APTOS_2019" else ".jpg"
        return self.image_root / f"{image_id}{suffix}"

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = self.image_path(row["image_id"])
        with Image.open(path) as source:
            image = source.convert("RGB")
        dme = int(row["dme"]) if row.get("dme") not in (None, "") else -1
        return self.transform(image), int(row["grade"]), dme, row["image_id"]
