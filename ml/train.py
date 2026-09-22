from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model import RetinaSathiNet


IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class IDRiDDataset(Dataset):
    def __init__(self, image_dir: Path, labels_csv: Path, transform: transforms.Compose) -> None:
        self.image_dir = image_dir
        self.transform = transform
        self.rows: list[tuple[str, int, int]] = []
        with labels_csv.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                normalized = {key.strip(): value.strip() for key, value in row.items() if key}
                self.rows.append(
                    (
                        normalized["Image name"],
                        int(normalized["Retinopathy grade"]),
                        int(normalized["Risk of macular edema"]),
                    )
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int, str]:
        image_id, grade, dme = self.rows[index]
        with Image.open(self.image_dir / f"{image_id}.jpg") as source:
            image = source.convert("RGB")
        return self.transform(image), grade, dme, image_id


def resolve_layout(root: Path) -> dict[str, Path]:
    base = root / "images" / "B. Disease Grading" / "1. Original Images"
    labels = root / "grading" / "B. Disease Grading" / "2. Groundtruths"
    paths = {
        "train_images": base / "a. Training Set",
        "test_images": base / "b. Testing Set",
        "train_labels": labels / "a. IDRiD_Disease Grading_Training Labels.csv",
        "test_labels": labels / "b. IDRiD_Disease Grading_Testing Labels.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing IDRiD files:\n" + "\n".join(missing))
    return paths


def class_weights(rows: list[tuple[str, int, int]], label_index: int, classes: int) -> torch.Tensor:
    counts = Counter(row[label_index] for row in rows)
    total = sum(counts.values())
    return torch.tensor([total / (classes * max(counts.get(i, 0), 1)) for i in range(classes)])


def metrics_for(y_true: list[int], y_pred: list[int], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def run_epoch(
    model: RetinaSathiNet,
    loader: DataLoader,
    device: torch.device,
    grade_loss: nn.Module,
    dme_loss: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    grade_true: list[int] = []
    grade_pred: list[int] = []
    dme_true: list[int] = []
    dme_pred: list[int] = []

    for images, grades, dme, _ in loader:
        images = images.to(device)
        grades = grades.to(device)
        dme = dme.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            grade_logits, dme_logits = model(images)
            loss = grade_loss(grade_logits, grades) + 0.35 * dme_loss(dme_logits, dme)
            if training:
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        grade_true.extend(grades.cpu().tolist())
        grade_pred.extend(grade_logits.argmax(1).detach().cpu().tolist())
        dme_true.extend(dme.cpu().tolist())
        dme_pred.extend(dme_logits.argmax(1).detach().cpu().tolist())

    values = {"loss": float(np.mean(losses))}
    values.update(metrics_for(grade_true, grade_pred, "grade"))
    values.update(metrics_for(dme_true, dme_pred, "dme"))
    values["grade_qwk"] = float(cohen_kappa_score(grade_true, grade_pred, weights="quadratic"))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the RetinaSathi IDRiD baseline")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("compute/artifacts/idrid_multitask.pt"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=26038)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    paths = resolve_layout(args.data_root)

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.78, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    train_data = IDRiDDataset(paths["train_images"], paths["train_labels"], train_transform)
    test_data = IDRiDDataset(paths["test_images"], paths["test_labels"], eval_transform)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = RetinaSathiNet(pretrained=True).to(device)
    model.freeze_backbone()
    grade_loss = nn.CrossEntropyLoss(weight=class_weights(train_data.rows, 1, 5).to(device))
    dme_loss = nn.CrossEntropyLoss(weight=class_weights(train_data.rows, 2, 3).to(device))
    history: list[dict[str, float | int]] = []
    best_score = float("-inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        if epoch == 3:
            model.unfreeze_last_blocks(3)
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=2e-4 if epoch < 3 else 5e-5,
            weight_decay=1e-4,
        )
        train_metrics = run_epoch(model, train_loader, device, grade_loss, dme_loss, optimizer)
        test_metrics = run_epoch(model, test_loader, device, grade_loss, dme_loss, None)
        record = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"test_{k}": v for k, v in test_metrics.items()}}
        history.append(record)
        print(json.dumps(record), flush=True)
        score = test_metrics["grade_qwk"]
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_version": "idrid-mobilenetv3-multitask-v0.1",
                    "image_size": IMAGE_SIZE,
                    "mean": MEAN,
                    "std": STD,
                    "dataset": "IDRiD",
                    "train_samples": len(train_data),
                    "test_samples": len(test_data),
                    "metrics": test_metrics,
                },
                args.output,
            )

    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps({"best_grade_qwk": best_score, "history": history}, indent=2))
    print(f"Saved best checkpoint to {args.output}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
