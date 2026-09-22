"""Source-aware V3 retinal grading dataset."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Collection

from PIL import Image
from torch.utils.data import Dataset


class V3ManifestDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        data_root: Path,
        transform: Callable,
        roles: Collection[str],
        folds: Collection[str] | None = None,
        maximum_records: int | None = None,
    ) -> None:
        self.data_root = data_root
        self.transform = transform
        role_set = set(roles)
        fold_set = set(folds) if folds is not None else None
        with manifest.open(encoding="utf-8", newline="") as handle:
            rows = [
                row
                for row in csv.DictReader(handle)
                if row["role"] in role_set and (fold_set is None or row["cv_fold"] in fold_set)
            ]
        self.rows = sorted(rows, key=lambda row: row["image_uid"])
        if maximum_records is not None:
            self.rows = self.rows[:maximum_records]
        if not self.rows:
            raise ValueError(f"No V3 manifest rows for roles={sorted(role_set)} folds={fold_set}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = self.data_root / row["relative_path"]
        with Image.open(path) as source:
            image = source.convert("RGB")
        return (
            self.transform(image),
            int(row["mapped_icdr_grade"]),
            float(row["referable_label"]),
            row["source_name"],
            row["image_uid"],
        )
