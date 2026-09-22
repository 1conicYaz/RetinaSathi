"""Train the learned DeepDRiD quality gate with patient-separated validation."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from ml.artifact_integrity import sha256
from ml.datasets import DeepDRiDQualityDataset
from ml.models import QualityModel
from ml.preprocessing import preprocess_fundus
from ml.training.train_classifier_v2 import device_for_training, git_commit, hardware_report, seed_everything


class QualityTransform:
    def __init__(self, image_size: int, training: bool) -> None:
        operations = []
        if training:
            operations = [transforms.RandomHorizontalFlip(), transforms.RandomAffine(10, translate=(0.03, 0.03), scale=(0.97, 1.03)), transforms.ColorJitter(0.1, 0.1, 0.08)]
        operations += [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        self.image_size = image_size; self.after = transforms.Compose(operations)

    def __call__(self, image):
        return self.after(preprocess_fundus(image, self.image_size, enhancement="none"))


def evaluate(model, loader, device):
    model.eval(); truth = []; probability = []; attribute_truth = []; attribute_predicted = []; losses = []
    with torch.no_grad():
        for images, labels, attributes, _, _ in loader:
            images, labels = images.to(device), labels.to(device)
            attributes = torch.stack(attributes, dim=1).to(device=device, dtype=torch.float32)
            logits, predicted_attributes = model(images)
            loss = nn.functional.cross_entropy(logits, labels) + 0.25*nn.functional.smooth_l1_loss(predicted_attributes, attributes)
            losses.append(float(loss.cpu())); truth.extend(labels.cpu().tolist())
            probability.extend(torch.softmax(logits, 1)[:, 1].cpu().tolist())
            attribute_truth.append(attributes.cpu().numpy()); attribute_predicted.append(predicted_attributes.cpu().numpy())
    probability_array = np.asarray(probability); truth_array = np.asarray(truth)
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probability_array)))
    def youden(threshold: float) -> float:
        selected = probability_array >= threshold
        sensitivity = ((selected == 1) & (truth_array == 1)).sum()/max((truth_array == 1).sum(), 1)
        specificity = ((selected == 0) & (truth_array == 0)).sum()/max((truth_array == 0).sum(), 1)
        return float(sensitivity + specificity - 1)
    threshold = float(max(candidates, key=lambda value: (youden(float(value)), -abs(float(value)-0.5))))
    predicted = probability_array >= threshold
    return {
        "loss": float(np.mean(losses)), "accuracy": float(accuracy_score(truth_array, predicted)),
        "macro_f1": float(f1_score(truth_array, predicted, average="macro")),
        "auroc": float(roc_auc_score(truth_array, probability_array)), "operating_threshold": threshold,
        "good_sensitivity": float(((predicted == 1) & (truth_array == 1)).sum()/max((truth_array == 1).sum(), 1)),
        "poor_specificity": float(((predicted == 0) & (truth_array == 0)).sum()/max((truth_array == 0).sum(), 1)),
        "attribute_mae": np.abs(np.concatenate(attribute_truth)-np.concatenate(attribute_predicted)).mean(axis=0).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/quality.yaml")); parser.add_argument("--data-root", type=Path)
    parser.add_argument("--resume", type=Path); parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(); config = yaml.safe_load(args.config.read_text()); seed = int(config["experiment"]["seed"]); seed_everything(seed)
    data_root = args.data_root or Path(os.environ.get("RETINASATHI_DATA_ROOT", ""))
    if not data_root.is_dir(): raise SystemExit("Set RETINASATHI_DATA_ROOT or pass --data-root")
    release = data_root / "DeepDRiD/original_extracted/DeepDRiD-1.1"; size = int(config["preprocessing"]["input_size"])
    train = DeepDRiDQualityDataset(release, "training", QualityTransform(size, True)); validation = DeepDRiDQualityDataset(release, "validation", QualityTransform(size, False))
    labels = [int(row["Overall quality"]) for row in train.rows]; counts = np.bincount(labels, minlength=2); weights = [1/counts[label] for label in labels]
    batch = int(config["training"]["batch_size"]); workers = int(config["training"].get("num_workers", 0)); loader_options = {"num_workers": workers, "persistent_workers": workers > 0, **({"prefetch_factor": 2} if workers > 0 else {})}
    train_loader = DataLoader(train, batch_size=batch, sampler=WeightedRandomSampler(weights, len(weights)), **loader_options)
    validation_loader = DataLoader(validation, batch_size=batch, shuffle=False, **loader_options)
    device = device_for_training(); model = QualityModel(pretrained=args.resume is None).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(config["training"]["epochs"]))
    start = 0; best = float("-inf")
    if args.resume:
        state = torch.load(args.resume, map_location=device); model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"]); start = state["epoch"]; best = state["best_auroc"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"); run_dir = args.run_dir or Path("runs")/f"{stamp}_quality_v1"; run_dir.mkdir(parents=True, exist_ok=bool(args.resume))
    (run_dir/"config.yaml").write_text(args.config.read_text()); (run_dir/"hardware.json").write_text(json.dumps(hardware_report(data_root), indent=2)+"\n")
    (run_dir/"provenance.json").write_text(json.dumps({"git_commit": git_commit(), "patient_overlap": 0, "official_partitions": True}, indent=2)+"\n")
    stale = 0; history = []
    for epoch in range(start+1, int(config["training"]["epochs"])+1):
        model.train(); losses=[]
        for images, labels, attributes, _, _ in train_loader:
            images, labels = images.to(device), labels.to(device); attributes = torch.stack(attributes, dim=1).to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True); logits, predicted_attributes = model(images)
            loss = nn.functional.cross_entropy(logits, labels) + 0.25*nn.functional.smooth_l1_loss(predicted_attributes, attributes)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        scheduler.step(); metrics = evaluate(model, validation_loader, device); record={"epoch":epoch,"train_loss":float(np.mean(losses)),"validation":metrics}; history.append(record)
        with (run_dir/"history.jsonl").open("a",encoding="utf-8") as handle:handle.write(json.dumps(record)+"\n")
        print(json.dumps(record), flush=True)
        is_best=metrics["auroc"]>best
        if is_best:best=metrics["auroc"]
        state={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"epoch":epoch,"best_auroc":best,"config":config}; torch.save(state, run_dir/"last.pt")
        if is_best:
            torch.save(state, run_dir/"best.pt"); (run_dir/"best_metrics.json").write_text(json.dumps(record,indent=2)+"\n"); stale=0
        else:
            stale+=1
            if stale >= int(config["training"]["early_stopping_patience"]): break
    (run_dir/"history.json").write_text(json.dumps(history,indent=2)+"\n")
    best_path = run_dir/"best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model"])
    final_metrics = evaluate(model, validation_loader, device)
    (run_dir/"final_validation.json").write_text(json.dumps({
        "model_status":"experimental_quality_model",
        "checkpoint_sha256":sha256(best_path),
        "validation":final_metrics,
        "label_mapping":{"0":"not good enough for retinal diagnosis","1":"good enough for retinal diagnosis"},
        "attribute_order":["Artifact (higher is worse)","Clarity (higher is better)","Field definition (higher is better)"],
        "official_patient_partitions":True,
        "official_test_used":False,
    },indent=2)+"\n")


if __name__ == "__main__": main()
