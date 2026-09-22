"""IDRiD disc/fovea localizations with aspect-safe coordinate transforms."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def coordinates(path: Path) -> dict[str, tuple[float, float]]:
    """Read the populated partition from IDRiD's shared train/test CSV table."""
    output: dict[str, tuple[float, float]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            image_id = str(row["Image No"]).strip()
            x_value = str(row["X- Coordinate"]).strip()
            y_value = str(row["Y - Coordinate"]).strip()
            if not x_value and not y_value:
                continue
            if not image_id or not x_value or not y_value:
                raise ValueError(f"Partial localization row in {path} at line {row_number}")
            if image_id in output:
                raise ValueError(f"Duplicate localization label for {image_id} in {path}")
            output[image_id] = (float(x_value), float(y_value))
    return output


class IDRiDLocalizationDataset(Dataset):
    def __init__(self,data_root:Path,manifest:Path,split:str,image_size:int,augment:bool)->None:
        with manifest.open(encoding="utf-8",newline="") as handle:self.ids=[row["image_id"] for row in csv.DictReader(handle) if row["split"]==split]
        ground=data_root/"IDRiD/grading/C. Localization/2. Groundtruths"; self.disc=coordinates(ground/"1. Optic Disc Center Location/a. IDRiD_OD_Center_Training Set_Markups.csv")
        self.fovea=coordinates(ground/"2. Fovea Center Location/IDRiD_Fovea_Center_Training Set_Markups.csv")
        missing = sorted(set(self.ids) - (set(self.disc) & set(self.fovea)))
        if missing:
            raise ValueError(
                f"Localization labels missing for {len(missing)} manifest image(s)"
            )
        self.images=data_root/"IDRiD/images/B. Disease Grading/1. Original Images/a. Training Set"; self.size=image_size
        operations=[]
        if augment:operations=[transforms.ColorJitter(0.1,0.1,0.08)]
        operations += [transforms.ToTensor(),transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))]; self.transform=transforms.Compose(operations)

    def __len__(self)->int:return len(self.ids)

    def __getitem__(self,index:int):
        image_id=self.ids[index]
        with Image.open(self.images/f"{image_id}.jpg") as source:image=source.convert("RGB")
        width,height=image.size; side=max(width,height); offset_x=(side-width)//2; offset_y=(side-height)//2
        canvas=Image.new("RGB",(side,side)); canvas.paste(image,(offset_x,offset_y)); canvas=canvas.resize((self.size,self.size),Image.Resampling.LANCZOS)
        disc=self.disc[image_id]; fovea=self.fovea[image_id]
        target=torch.tensor([(disc[0]+offset_x)/side,(disc[1]+offset_y)/side,(fovea[0]+offset_x)/side,(fovea[1]+offset_y)/side],dtype=torch.float32)
        return self.transform(canvas),target,image_id
