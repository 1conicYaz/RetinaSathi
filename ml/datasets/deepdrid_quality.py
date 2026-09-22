"""Official patient-partitioned DeepDRiD quality dataset."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


QUALITY_LEVELS = {
    "Artifact": (0, 1, 4, 6, 8, 10),
    "Clarity": (1, 4, 6, 8, 10),
    "Field definition": (1, 4, 6, 8, 10),
}


class DeepDRiDQualityDataset(Dataset):
    def __init__(self, release_root: Path, split: str, transform) -> None:
        folder = f"regular-fundus-{split}"
        self.root = release_root / "regular_fundus_images" / folder
        self.transform = transform
        with (self.root / f"{folder}.csv").open(encoding="utf-8-sig", newline="") as handle:
            self.rows = list(csv.DictReader(handle))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = self.root / "Images" / row["patient_id"] / f"{row['image_id']}.jpg"
        with Image.open(path) as source:
            image = source.convert("RGB")
        attributes = [int(row[name]) / 10 for name in QUALITY_LEVELS]
        return self.transform(image), int(row["Overall quality"]), attributes, row["patient_id"], row["image_id"]
