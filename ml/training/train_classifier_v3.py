"""Train RetinaSathi V3-0 with source-aware dual-head screening outputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from ml.artifact_integrity import sha256
from ml.calibration import fit_binary_temperature, fit_temperature
from ml.datasets import V3ManifestDataset
from ml.evaluation import classifier_metrics, referable_metrics, select_safety_threshold
from ml.models import DualHeadDRClassifier, coral_loss, ordinal_probabilities
from ml.training.train_classifier_v2 import (
    RetinalTransform,
    device_for_training,
    git_commit,
    hardware_report,
    seed_everything,
)


def source_balanced_sampler(dataset: V3ManifestDataset) -> WeightedRandomSampler:
    counts = Counter(row["source_name"] for row in dataset.rows)
    weights = [1.0 / counts[row["source_name"]] for row in dataset.rows]
    return WeightedRandomSampler(weights, len(weights), replacement=True)


def ordinal_positive_weights(dataset: V3ManifestDataset, device: torch.device) -> torch.Tensor:
    grades = torch.tensor([int(row["mapped_icdr_grade"]) for row in dataset.rows])
    targets = torch.stack([(grades > threshold) for threshold in range(4)], dim=1)
    positive = targets.sum(dim=0)
    negative = len(targets) - positive
    return (negative / positive.clamp_min(1)).clamp(0.25, 4.0).to(device=device, dtype=torch.float32)


def referable_positive_weight(dataset: V3ManifestDataset, device: torch.device) -> torch.Tensor:
    targets = torch.tensor([int(row["referable_label"]) for row in dataset.rows])
    positive = targets.sum()
    negative = len(targets) - positive
    return (negative / positive.clamp_min(1)).clamp(0.5, 3.0).to(device=device, dtype=torch.float32)


def make_loader(
    dataset: V3ManifestDataset,
    config: dict[str, object],
    training: bool,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    options = config["training"]  # type: ignore[index]
    workers = int(options["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(options["batch_size"]),
        sampler=sampler,
        shuffle=training and sampler is None,
        num_workers=workers,
        persistent_workers=workers > 0,
        **({"prefetch_factor": 2} if workers > 0 else {}),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ordinal_weights: torch.Tensor,
    referable_weight: torch.Tensor,
    ordinal_loss_weight: float,
    referable_loss_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
    accumulation: int = 1,
    amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    gradient_clip_norm: float = 0.0,
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    total_losses: list[float] = []
    ordinal_losses: list[float] = []
    referable_losses: list[float] = []
    all_binary_logits: list[torch.Tensor] = []
    all_ordinal_logits: list[torch.Tensor] = []
    all_grades: list[torch.Tensor] = []
    all_sources: list[str] = []
    all_uids: list[str] = []

    for step, (images, grades, referable, sources, uids) in enumerate(loader, start=1):
        images = images.to(device)
        grades = grades.to(device=device, dtype=torch.long)
        referable = referable.to(device=device, dtype=torch.float32)
        with torch.autocast(device_type=device.type, enabled=amp):
            binary_logits, ordinal_logits = model(images)
            ordinal_loss = coral_loss(ordinal_logits, grades, ordinal_weights)
            binary_loss = nn.functional.binary_cross_entropy_with_logits(
                binary_logits,
                referable,
                pos_weight=referable_weight,
            )
            combined = ordinal_loss_weight * ordinal_loss + referable_loss_weight * binary_loss
            scaled_loss = combined / accumulation
        if training:
            if scaler is not None and amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            if step % accumulation == 0 or step == len(loader):
                if gradient_clip_norm > 0:
                    if scaler is not None and amp:
                        scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                if scaler is not None and amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        total_losses.append(float(combined.detach().cpu()))
        ordinal_losses.append(float(ordinal_loss.detach().cpu()))
        referable_losses.append(float(binary_loss.detach().cpu()))
        all_binary_logits.append(binary_logits.detach().cpu())
        all_ordinal_logits.append(ordinal_logits.detach().cpu())
        all_grades.append(grades.detach().cpu())
        all_sources.extend(list(sources))
        all_uids.extend(list(uids))

    binary_logits = torch.cat(all_binary_logits)
    ordinal_logits = torch.cat(all_ordinal_logits)
    grades = torch.cat(all_grades)
    grade_probabilities = ordinal_probabilities(ordinal_logits).numpy()
    binary_scores = torch.sigmoid(binary_logits).numpy()
    grade_metrics = classifier_metrics(grades.numpy(), grade_probabilities)
    binary_metrics = referable_metrics(grades.numpy() >= 2, binary_scores)
    return {
        "loss": float(np.mean(total_losses)),
        "ordinal_loss": float(np.mean(ordinal_losses)),
        "referable_loss": float(np.mean(referable_losses)),
        "binary_logits": binary_logits,
        "ordinal_logits": ordinal_logits,
        "grades": grades,
        "sources": all_sources,
        "uids": all_uids,
        "grade_metrics": grade_metrics,
        "referable_metrics": binary_metrics,
    }


def report_view(result: dict[str, object]) -> dict[str, object]:
    return {
        "loss": result["loss"],
        "ordinal_loss": result["ordinal_loss"],
        "referable_loss": result["referable_loss"],
        "grade": result["grade_metrics"],
        "referable": result["referable_metrics"],
    }


def calibrated_evaluation(
    result: dict[str, object],
    ordinal_temperature: float,
    binary_temperature: float,
    threshold: float,
) -> dict[str, object]:
    grades = result["grades"].numpy()  # type: ignore[union-attr]
    ordinal_probabilities_value = ordinal_probabilities(
        result["ordinal_logits"], ordinal_temperature  # type: ignore[arg-type]
    ).numpy()
    binary_scores = torch.sigmoid(
        result["binary_logits"] / binary_temperature  # type: ignore[operator]
    ).numpy()
    return {
        "grade": classifier_metrics(grades, ordinal_probabilities_value),
        "referable": referable_metrics(grades >= 2, binary_scores, threshold),
        "grade_probabilities": ordinal_probabilities_value,
        "referable_scores": binary_scores,
    }


def source_breakdown(
    result: dict[str, object],
    evaluation: dict[str, object],
    threshold: float,
) -> dict[str, object]:
    grades = result["grades"].numpy()  # type: ignore[union-attr]
    sources = np.asarray(result["sources"])
    grade_probabilities_value = evaluation["grade_probabilities"]
    referable_scores = evaluation["referable_scores"]
    output: dict[str, object] = {}
    for source in sorted(set(sources.tolist())):
        selected = sources == source
        output[source] = {
            "records": int(selected.sum()),
            "grade": classifier_metrics(grades[selected], grade_probabilities_value[selected]),
            "referable": referable_metrics(grades[selected] >= 2, referable_scores[selected], threshold),
        }
    return output


def write_predictions(
    path: Path,
    result: dict[str, object],
    evaluation: dict[str, object],
) -> None:
    grades = result["grades"].numpy()  # type: ignore[union-attr]
    probabilities = evaluation["grade_probabilities"]
    scores = evaluation["referable_scores"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_uid", "source", "true_grade", "referable_score", "p0", "p1", "p2", "p3", "p4"])
        for uid, source, grade, score, probability in zip(
            result["uids"], result["sources"], grades, scores, probabilities
        ):
            writer.writerow([uid, source, int(grade), float(score), *[float(value) for value in probability]])


def balanced_tiny_rows(dataset: V3ManifestDataset, per_grade: int) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    by_grade: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in dataset.rows:
        by_grade[int(row["mapped_icdr_grade"])].append(row)
    for grade in range(5):
        rows = sorted(by_grade[grade], key=lambda row: (row["source_name"], row["image_uid"]))
        if len(rows) < per_grade:
            raise ValueError(f"Grade {grade} has only {len(rows)} rows for tiny overfit")
        selected.extend(rows[:per_grade])
    return selected


def balanced_smoke_rows(dataset: V3ManifestDataset, maximum_records: int) -> list[dict[str, str]]:
    """Create a deterministic, grade-aware subset for end-to-end pipeline rehearsal."""
    if maximum_records < 5:
        raise ValueError("Smoke rehearsal needs at least five records per split")
    by_grade: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in dataset.rows:
        by_grade[int(row["mapped_icdr_grade"])].append(row)
    for rows in by_grade.values():
        rows.sort(key=lambda row: (row["source_name"], row["image_uid"]))
    selected: list[dict[str, str]] = []
    cursor = 0
    while len(selected) < maximum_records:
        added = False
        for grade in range(5):
            rows = by_grade[grade]
            if cursor < len(rows) and len(selected) < maximum_records:
                selected.append(rows[cursor])
                added = True
        if not added:
            break
        cursor += 1
    if len(selected) < 5:
        raise ValueError("Smoke rehearsal could not include all five DR grades")
    return selected


def run_tiny_overfit(
    config: dict[str, object],
    data_root: Path,
    manifest: Path,
    run_dir: Path,
    device: torch.device,
) -> None:
    training = config["training"]  # type: ignore[index]
    validation_fold = str(config["experiment"]["validation_fold"])  # type: ignore[index]
    dataset = V3ManifestDataset(
        manifest,
        data_root,
        RetinalTransform(config, False),
        roles={"development_pool"},
        folds={validation_fold},
    )
    dataset.rows = balanced_tiny_rows(dataset, int(training["tiny_overfit_per_grade"]))
    loader = make_loader(dataset, config, training=True)
    model = DualHeadDRClassifier(
        str(config["experiment"]["architecture"]),  # type: ignore[index]
        pretrained=True,
        dropout=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.0)
    ordinal_weights = ordinal_positive_weights(dataset, device)
    binary_weight = referable_positive_weight(dataset, device)
    common = {
        "model": model,
        "loader": loader,
        "device": device,
        "ordinal_weights": ordinal_weights,
        "referable_weight": binary_weight,
        "ordinal_loss_weight": float(training["ordinal_loss_weight"]),
        "referable_loss_weight": float(training["referable_loss_weight"]),
    }
    initial = run_epoch(**common)
    history = []
    for epoch in range(1, int(training["tiny_overfit_epochs"]) + 1):
        result = run_epoch(
            **common,
            optimizer=optimizer,
            accumulation=1,
            gradient_clip_norm=float(training["gradient_clip_norm"]),
        )
        history.append({"epoch": epoch, **report_view(result)})
        print(json.dumps({"tiny_overfit_epoch": epoch, "loss": result["loss"]}), flush=True)
    final = run_epoch(**common)
    passed = float(final["loss"]) < float(initial["loss"]) * 0.85
    report = {
        "status": "pass" if passed else "fail",
        "records": len(dataset),
        "initial": report_view(initial),
        "final": report_view(final),
        "history": history,
    }
    (run_dir / "tiny_overfit.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save({"model": model.state_dict(), "config": config, "report": report}, run_dir / "tiny_overfit.pt")
    print(json.dumps({"tiny_overfit": report["status"], "run_dir": str(run_dir)}))
    if not passed:
        raise SystemExit("Tiny-overfit gate failed; refusing full V3 training")


def full_training(
    config: dict[str, object],
    data_root: Path,
    manifest: Path,
    run_dir: Path,
    device: torch.device,
    smoke_records_per_split: int | None = None,
) -> None:
    experiment = config["experiment"]  # type: ignore[index]
    training = config["training"]  # type: ignore[index]
    selection = config["selection"]  # type: ignore[index]
    validation_fold = str(experiment["validation_fold"])
    calibration_fold = str(experiment["calibration_fold"])
    training_folds = {str(fold) for fold in range(5)} - {validation_fold, calibration_fold}
    train_data = V3ManifestDataset(manifest, data_root, RetinalTransform(config, True), {"development_pool"}, training_folds)
    validation_data = V3ManifestDataset(manifest, data_root, RetinalTransform(config, False), {"development_pool"}, {validation_fold})
    calibration_data = V3ManifestDataset(manifest, data_root, RetinalTransform(config, False), {"development_pool"}, {calibration_fold})
    source_validation_data = V3ManifestDataset(manifest, data_root, RetinalTransform(config, False), {"source_validation"})
    if smoke_records_per_split is not None:
        for dataset in (train_data, validation_data, calibration_data, source_validation_data):
            dataset.rows = balanced_smoke_rows(dataset, smoke_records_per_split)
    sampler = source_balanced_sampler(train_data) if training["source_balanced_sampling"] else None
    train_loader = make_loader(train_data, config, True, sampler)
    validation_loader = make_loader(validation_data, config, False)
    calibration_loader = make_loader(calibration_data, config, False)
    source_validation_loader = make_loader(source_validation_data, config, False)

    model = DualHeadDRClassifier(
        str(experiment["architecture"]),
        pretrained=True,
        dropout=float(training["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(training["epochs"]))
    ordinal_weights = ordinal_positive_weights(train_data, device)
    binary_weight = referable_positive_weight(train_data, device)
    amp = bool(training["mixed_precision"] == "auto" and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    common = {
        "device": device,
        "ordinal_weights": ordinal_weights,
        "referable_weight": binary_weight,
        "ordinal_loss_weight": float(training["ordinal_loss_weight"]),
        "referable_loss_weight": float(training["referable_loss_weight"]),
    }
    best_key = (float("-inf"), float("-inf"))
    stale = 0
    for epoch in range(1, int(training["epochs"]) + 1):
        train_result = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            accumulation=int(training["gradient_accumulation"]),
            amp=amp,
            scaler=scaler,
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            **common,
        )
        validation_result = run_epoch(model, validation_loader, **common)
        scheduler.step()
        auprc = validation_result["referable_metrics"]["auprc"]  # type: ignore[index]
        qwk = validation_result["grade_metrics"]["qwk"]  # type: ignore[index]
        key = (float(auprc if auprc is not None else -1), float(qwk))
        record = {
            "epoch": epoch,
            "learning_rate": scheduler.get_last_lr()[0],
            "train": report_view(train_result),
            "validation": report_view(validation_result),
        }
        with (run_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "selection_key": key,
            "config": config,
            "manifest_sha256": sha256(manifest),
        }
        torch.save(state, run_dir / "last.pt")
        if key > best_key:
            best_key = key
            stale = 0
            torch.save(state, run_dir / "best.pt")
        else:
            stale += 1
            if stale >= int(training["early_stopping_patience"]):
                break

    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    validation_result = run_epoch(model, validation_loader, **common)
    calibration_result = run_epoch(model, calibration_loader, **common)
    source_validation_result = run_epoch(model, source_validation_loader, **common)
    ordinal_temperature = fit_temperature(
        calibration_result["ordinal_logits"].to(device),  # type: ignore[union-attr]
        calibration_result["grades"].to(device),  # type: ignore[union-attr]
        objective="coral",
    )
    binary_temperature = fit_binary_temperature(
        calibration_result["binary_logits"].to(device),  # type: ignore[union-attr]
        (calibration_result["grades"] >= 2).to(device),  # type: ignore[operator]
    )
    calibrated_calibration = calibrated_evaluation(
        calibration_result,
        ordinal_temperature,
        binary_temperature,
        0.5,
    )
    threshold_gate = select_safety_threshold(
        calibration_result["grades"].numpy() >= 2,  # type: ignore[union-attr]
        calibrated_calibration["referable_scores"],
        float(selection["minimum_referable_sensitivity"]),
        float(selection["minimum_referable_specificity"]),
    )
    threshold = float(threshold_gate["selected"]["threshold"])  # type: ignore[index]
    validation_evaluation = calibrated_evaluation(validation_result, ordinal_temperature, binary_temperature, threshold)
    calibration_evaluation = calibrated_evaluation(calibration_result, ordinal_temperature, binary_temperature, threshold)
    source_validation_evaluation = calibrated_evaluation(source_validation_result, ordinal_temperature, binary_temperature, threshold)
    final = {
        "model_status": "candidate_source_validation_complete" if threshold_gate["passed"] else "failed_calibration_safety_gate",
        "official_test_used": False,
        "manifest_sha256": sha256(manifest),
        "checkpoint_sha256": sha256(run_dir / "best.pt"),
        "selected_epoch": checkpoint["epoch"],
        "ordinal_temperature": ordinal_temperature,
        "binary_temperature": binary_temperature,
        "threshold_gate": threshold_gate,
        "validation": {
            "grade": validation_evaluation["grade"],
            "referable": validation_evaluation["referable"],
            "by_source": source_breakdown(validation_result, validation_evaluation, threshold),
        },
        "calibration": {
            "grade": calibration_evaluation["grade"],
            "referable": calibration_evaluation["referable"],
            "by_source": source_breakdown(calibration_result, calibration_evaluation, threshold),
        },
        "source_validation": {
            "grade": source_validation_evaluation["grade"],
            "referable": source_validation_evaluation["referable"],
            "by_source": source_breakdown(source_validation_result, source_validation_evaluation, threshold),
        },
    }
    (run_dir / "final_validation.json").write_text(json.dumps(final, indent=2) + "\n")
    write_predictions(run_dir / "validation_predictions.csv", validation_result, validation_evaluation)
    write_predictions(run_dir / "calibration_predictions.csv", calibration_result, calibration_evaluation)
    write_predictions(run_dir / "source_validation_predictions.csv", source_validation_result, source_validation_evaluation)
    print(json.dumps({"run_dir": str(run_dir), "model_status": final["model_status"], "threshold_gate": threshold_gate["passed"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/classifier_v3_0.yaml"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--tiny-overfit", action="store_true")
    parser.add_argument("--input-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument(
        "--smoke-records-per-split",
        type=int,
        help="Run the complete pipeline on a small deterministic subset of each split",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.input_size:
        config["preprocessing"]["input_size"] = args.input_size
    if args.epochs:
        config["training"]["epochs"] = args.epochs
        config["training"]["early_stopping_patience"] = max(
            int(config["training"]["early_stopping_patience"]), args.epochs
        )
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers
    if args.gradient_accumulation:
        config["training"]["gradient_accumulation"] = args.gradient_accumulation
    data_root = args.data_root or Path(os.environ.get("RETINASATHI_DATA_ROOT", ""))
    if not data_root.is_dir():
        raise SystemExit("Set RETINASATHI_DATA_ROOT or pass --data-root")
    manifest = Path(config["data"]["manifest"])
    validation_report = json.loads(Path(config["data"]["validation_report"]).read_text())
    if validation_report.get("status") != "pass":
        raise SystemExit("V3 manifest validation has not passed")
    if sha256(manifest) != validation_report.get("manifest_sha256", sha256(manifest)):
        raise SystemExit("Manifest hash differs from its validation report")
    seed_everything(int(config["experiment"]["seed"]))
    device = device_for_training()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "tiny_overfit" if args.tiny_overfit else str(config["experiment"]["name"])
    run_dir = args.run_dir or Path("runs") / f"{stamp}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (run_dir / "hardware.json").write_text(json.dumps(hardware_report(data_root), indent=2) + "\n")
    (run_dir / "provenance.json").write_text(
        json.dumps(
            {
                "created_at": stamp,
                "git_commit": git_commit(),
                "manifest_sha256": sha256(manifest),
                "official_test_used": False,
                "mode": (
                    "tiny_overfit"
                    if args.tiny_overfit
                    else "smoke_rehearsal"
                    if args.smoke_records_per_split is not None
                    else "full_training"
                ),
                "smoke_records_per_split": args.smoke_records_per_split,
            },
            indent=2,
        )
        + "\n"
    )
    if args.tiny_overfit:
        run_tiny_overfit(config, data_root, manifest, run_dir, device)
    else:
        full_training(
            config,
            data_root,
            manifest,
            run_dir,
            device,
            smoke_records_per_split=args.smoke_records_per_split,
        )


if __name__ == "__main__":
    main()
