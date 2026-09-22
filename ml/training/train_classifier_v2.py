"""Leakage-free, resumable APTOS-to-IDRiD ordinal classifier training."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from ml.artifact_integrity import sha256
from ml.calibration import fit_temperature
from ml.datasets import ManifestFundusDataset
from ml.evaluation import best_referable_threshold, classifier_metrics
from ml.models import NominalDRClassifier, OrdinalDRClassifier, coral_loss, ordinal_probabilities
from ml.preprocessing import preprocess_fundus


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class RetinalTransform:
    def __init__(self, config: dict[str, object], training: bool) -> None:
        self.image_size = int(config["preprocessing"]["input_size"])  # type: ignore[index]
        self.enhancement = str(config["preprocessing"]["enhancement"])  # type: ignore[index]
        self.color_normalization = bool(config["preprocessing"]["color_normalization"])  # type: ignore[index]
        augmentation = config["augmentation"]  # type: ignore[index]
        operations: list[object] = []
        if training:
            operations.extend(
                [
                    transforms.RandomHorizontalFlip(float(augmentation["horizontal_flip_probability"])),
                    transforms.RandomAffine(
                        degrees=float(augmentation["rotation_degrees"]),
                        translate=(float(augmentation["translation_fraction"]),) * 2,
                        scale=(0.97, 1.03),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                        fill=0,
                    ),
                    transforms.ColorJitter(
                        brightness=float(augmentation["brightness"]),
                        contrast=float(augmentation["contrast"]),
                        saturation=float(augmentation["saturation"]),
                    ),
                ]
            )
        operations.extend([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        self.after = transforms.Compose(operations)

    def __call__(self, image):
        image = preprocess_fundus(image, self.image_size, self.enhancement, self.color_normalization)
        return self.after(image)


def device_for_training() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def hardware_report(data_root: Path) -> dict[str, object]:
    disk = shutil.disk_usage(data_root)
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "selected_device": str(device_for_training()),
        "disk_free_gib": round(disk.free / 1024**3, 2),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def make_dataset(data_root: Path, split_dir: Path, dataset: str, split: str, transform):
    if dataset == "APTOS_2019":
        manifest = split_dir / "aptos_train_validation.csv"
        images = data_root / "APTOS_2019/extracted/train_images"
    else:
        manifest = split_dir / "idrid_internal_train_validation.csv"
        images = data_root / "IDRiD/images/B. Disease Grading/1. Original Images/a. Training Set"
    return ManifestFundusDataset(manifest, images, dataset, split, transform)


def weighted_sampler(dataset: ManifestFundusDataset) -> WeightedRandomSampler:
    labels = [int(row["grade"]) for row in dataset.rows]
    counts = Counter(labels)
    weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(weights, len(weights), replacement=True)


def threshold_positive_weights(dataset: ManifestFundusDataset, device: torch.device) -> torch.Tensor:
    grades = torch.tensor([int(row["grade"]) for row in dataset.rows])
    targets = torch.stack([(grades > threshold) for threshold in range(4)], dim=1)
    positives = targets.sum(dim=0); negatives = len(targets) - positives
    return (negatives / positives.clamp_min(1)).to(device=device, dtype=torch.float32)


def class_weights(dataset: ManifestFundusDataset, device: torch.device) -> torch.Tensor:
    """Return inverse-frequency five-class weights normalized to mean one."""
    grades = torch.tensor([int(row["grade"]) for row in dataset.rows])
    counts = torch.bincount(grades, minlength=5).clamp_min(1)
    weights = counts.sum() / (len(counts) * counts)
    return weights.to(device=device, dtype=torch.float32)


def grade_loss_weights(
    dataset: ManifestFundusDataset, objective: str, device: torch.device
) -> torch.Tensor:
    return (
        threshold_positive_weights(dataset, device)
        if objective == "coral"
        else class_weights(dataset, device)
    )


def build_model(config: dict[str, object], pretrained: bool) -> nn.Module:
    experiment = config["experiment"]  # type: ignore[index]
    training = config["training"]  # type: ignore[index]
    architecture = str(experiment["architecture"])
    dme_head = float(training["dme_weight"]) > 0
    if training["objective"] == "coral":
        return OrdinalDRClassifier(architecture, pretrained=pretrained, dme_head=dme_head)
    if training["objective"] == "cross_entropy":
        return NominalDRClassifier(architecture, pretrained=pretrained, dme_head=dme_head)
    raise ValueError(f"Unsupported objective: {training['objective']}")


def load_aptos_backbone(
    model: nn.Module, checkpoint: Path, architecture: str
) -> None:
    """Load only a fixed APTOS backbone, allowing objective/head comparisons."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("phase") != "APTOS_2019":
        raise ValueError("Backbone initialization must be an APTOS checkpoint")
    checkpoint_architecture = state["config"]["experiment"]["architecture"]
    if checkpoint_architecture != architecture:
        raise ValueError(
            f"Backbone architecture mismatch: {checkpoint_architecture} != {architecture}"
        )
    backbone_state = {
        key.removeprefix("backbone."): value
        for key, value in state["model"].items()
        if key.startswith("backbone.")
    }
    model.backbone.load_state_dict(backbone_state, strict=True)


def grade_probabilities(logits: torch.Tensor, objective: str, temperature: float = 1.0) -> torch.Tensor:
    return ordinal_probabilities(logits, temperature) if objective == "coral" else torch.softmax(logits / temperature, dim=1)


def run_epoch(model, loader, device, objective, grade_weights, dme_weight, optimizer=None, accumulation=1, amp=False, scaler=None):
    training = optimizer is not None
    model.train(training); losses: list[float] = []; all_logits = []; all_truth = []
    if training: optimizer.zero_grad(set_to_none=True)
    for step, (images, grades, dme, _) in enumerate(loader, start=1):
        images, grades, dme = images.to(device), grades.to(device), dme.to(device)
        with torch.autocast(device_type=device.type, enabled=amp):
            grade_logits, dme_logits = model(images)
            grade_loss = coral_loss(grade_logits, grades, grade_weights) if objective == "coral" else nn.functional.cross_entropy(grade_logits, grades, weight=grade_weights)
            valid_dme = dme >= 0
            auxiliary = nn.functional.cross_entropy(dme_logits[valid_dme], dme[valid_dme]) if dme_logits is not None and valid_dme.any() else torch.zeros((), device=device)
            loss = (grade_loss + dme_weight * auxiliary) / accumulation
        if training:
            if scaler is not None and amp: scaler.scale(loss).backward()
            else: loss.backward()
            if step % accumulation == 0 or step == len(loader):
                if scaler is not None and amp: scaler.step(optimizer); scaler.update()
                else: optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()) * accumulation)
        all_logits.append(grade_logits.detach().cpu()); all_truth.append(grades.detach().cpu())
    logits = torch.cat(all_logits); truth = torch.cat(all_truth)
    probabilities = grade_probabilities(logits, objective).numpy()
    return float(np.mean(losses)), logits, truth, classifier_metrics(truth.numpy(), probabilities)


def loaders(data_root: Path, split_dir: Path, dataset_name: str, config: dict[str, object]):
    training = config["training"]  # type: ignore[index]
    train_data = make_dataset(data_root, split_dir, dataset_name, "train", RetinalTransform(config, True))
    validation_data = make_dataset(data_root, split_dir, dataset_name, "validation", RetinalTransform(config, False))
    sampler = weighted_sampler(train_data) if training["imbalance"] in ("weighted_sampler", "sampler_only") else None
    worker_count = int(training["num_workers"])
    common = {
        "batch_size": int(training["batch_size"]),
        "num_workers": worker_count,
        "persistent_workers": worker_count > 0,
        **({"prefetch_factor": 2} if worker_count > 0 else {}),
    }
    return (
        train_data,
        validation_data,
        DataLoader(train_data, sampler=sampler, shuffle=sampler is None, **common),
        DataLoader(validation_data, shuffle=False, **common),
    )


def train_phase(model, dataset_name, data_root, split_dir, config, run_dir, device, start_epoch=0, resume_state=None):
    training = config["training"]; objective = str(training["objective"])
    train_data, _, train_loader, validation_loader = loaders(data_root, split_dir, dataset_name, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    phase_epochs = int(training["aptos_epochs"] if dataset_name == "APTOS_2019" else training["idrid_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase_epochs)
    if resume_state:
        optimizer.load_state_dict(resume_state["optimizer"]); scheduler.load_state_dict(resume_state["scheduler"])
    grade_weights = grade_loss_weights(train_data, objective, device) if training["imbalance"] in ("weighted_sampler", "weighted_loss") else None
    best_qwk = float(resume_state["best_qwk"]) if resume_state else float("-inf"); stale = 0; history = []
    amp = bool(training["mixed_precision"] == "auto" and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    if resume_state and resume_state.get("scaler"): scaler.load_state_dict(resume_state["scaler"])
    validation_logits = validation_truth = None
    for epoch in range(start_epoch + 1, phase_epochs + 1):
        train_loss, _, _, train_metrics = run_epoch(model, train_loader, device, objective, grade_weights, float(training["dme_weight"]), optimizer, int(training["gradient_accumulation"]), amp, scaler)
        validation_loss, validation_logits, validation_truth, validation_metrics = run_epoch(model, validation_loader, device, objective, grade_weights, float(training["dme_weight"]))
        scheduler.step()
        record = {"dataset": dataset_name, "epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss, "train": train_metrics, "validation": validation_metrics, "learning_rate": scheduler.get_last_lr()[0]}
        history.append(record)
        with (run_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        is_best = float(validation_metrics["qwk"]) > best_qwk
        if is_best: best_qwk = float(validation_metrics["qwk"])
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "phase": dataset_name, "epoch": epoch, "best_qwk": best_qwk, "config": config}
        torch.save(state, run_dir / "last.pt")
        if is_best:
            stale = 0; torch.save(state, run_dir / f"best_{dataset_name.lower()}.pt")
            (run_dir / f"best_{dataset_name.lower()}_metrics.json").write_text(json.dumps(record, indent=2) + "\n")
        else:
            stale += 1
            if stale >= int(training["early_stopping_patience"]): break
    best_path = run_dir / f"best_{dataset_name.lower()}.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    # Calibration and threshold selection must use predictions from the exact
    # best checkpoint, never stale predictions from the final training epoch.
    _, validation_logits, validation_truth, _ = run_epoch(
        model,
        validation_loader,
        device,
        objective,
        grade_weights,
        float(training["dme_weight"]),
    )
    return history, validation_logits, validation_truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/classifier.yaml"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--backbone-checkpoint", type=Path)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.resume and args.backbone_checkpoint:
        raise SystemExit("Use either --resume or --backbone-checkpoint, not both")
    config = yaml.safe_load(args.config.read_text())
    data_root = args.data_root or Path(os.environ.get("RETINASATHI_DATA_ROOT", ""))
    if not data_root.is_dir(): raise SystemExit("Set RETINASATHI_DATA_ROOT or pass --data-root")
    split_dir = Path(config["data"]["split_dir"])
    if not split_dir.is_dir(): raise SystemExit("Generate splits first with scripts/create_splits.py")
    seed = int(config["experiment"]["seed"]); seed_everything(seed); device = device_for_training()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.run_dir or Path("runs") / f"{stamp}_{config['experiment']['name']}"
    run_dir.mkdir(parents=True, exist_ok=bool(args.resume))
    (run_dir / "config.yaml").write_text(args.config.read_text()); (run_dir / "hardware.json").write_text(json.dumps(hardware_report(data_root), indent=2) + "\n")
    (run_dir / "provenance.json").write_text(json.dumps({"git_commit": git_commit(), "official_test_used": False, "created_at": stamp,"backbone_checkpoint_sha256":sha256(args.backbone_checkpoint) if args.backbone_checkpoint else None}, indent=2) + "\n")
    resume_state = torch.load(args.resume, map_location="cpu") if args.resume else None
    model = build_model(config, pretrained=resume_state is None and args.backbone_checkpoint is None)
    if resume_state: model.load_state_dict(resume_state["model"])
    if args.backbone_checkpoint:
        load_aptos_backbone(
            model,
            args.backbone_checkpoint,
            str(config["experiment"]["architecture"]),
        )
    model.to(device)
    stages = []
    if config["data"]["aptos_pretrain"]: stages.append("APTOS_2019")
    if config["data"]["idrid_finetune"]: stages.append("IDRiD")
    history = []
    validation_logits = validation_truth = None
    for dataset_name in stages:
        phase_resume = resume_state if resume_state and resume_state["phase"] == dataset_name else None
        phase_history, validation_logits, validation_truth = train_phase(model, dataset_name, data_root, split_dir, config, run_dir, device, int(phase_resume["epoch"]) if phase_resume else 0, phase_resume)
        history.extend(phase_history); resume_state = None
    if validation_logits is None or validation_truth is None: raise RuntimeError("No training stage ran")
    objective = str(config["training"]["objective"])
    uncalibrated = grade_probabilities(validation_logits, objective).numpy()
    temperature = fit_temperature(validation_logits.to(device), validation_truth.to(device), objective=objective)
    calibrated = grade_probabilities(validation_logits, objective, temperature).numpy()
    threshold = best_referable_threshold(validation_truth.numpy(), calibrated)
    final_metrics = classifier_metrics(validation_truth.numpy(), calibrated, threshold)
    final = {"model_status": "candidate_not_tested", "temperature": temperature, "referable_threshold": threshold, "validation_uncalibrated": classifier_metrics(validation_truth.numpy(), uncalibrated), "validation": final_metrics, "official_test_used": False, "history": history}
    (run_dir / "final_validation.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps({"run_dir": str(run_dir), "validation_qwk": final_metrics["qwk"], "official_test_used": False}))


if __name__ == "__main__":
    main()
