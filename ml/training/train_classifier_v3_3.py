"""Train RetinaSathi V3.3 with source-grade balancing and three screening heads."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
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
from ml.models import coral_loss, load_official_retfound_mae_cfp, ordinal_probabilities
from ml.training.v3_1_components import (
    candidate_status,
    joint_source_grade_sample_weights,
    nominal_class_weights,
    selection_key,
)
from ml.training.train_classifier_v2 import (
    RetinalTransform,
    device_for_training,
    git_commit,
    hardware_report,
    seed_everything,
)


def source_grade_balanced_sampler(
    dataset: V3ManifestDataset,
    max_ratio: float,
) -> WeightedRandomSampler:
    weights = joint_source_grade_sample_weights(dataset.rows, max_ratio=max_ratio)
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


def write_progress(path: Path, payload: dict[str, object]) -> None:
    """Atomically publish batch-level progress for the local monitor."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ordinal_weights: torch.Tensor,
    referable_weight: torch.Tensor,
    nominal_weights: torch.Tensor,
    ordinal_loss_weight: float,
    referable_loss_weight: float,
    nominal_loss_weight: float,
    nominal_label_smoothing: float,
    optimizer: torch.optim.Optimizer | None = None,
    accumulation: int = 1,
    amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    gradient_clip_norm: float = 0.0,
    progress_path: Path | None = None,
    phase: str = "evaluation",
    epoch: int | None = None,
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    total_losses: list[float] = []
    ordinal_losses: list[float] = []
    referable_losses: list[float] = []
    nominal_losses: list[float] = []
    all_binary_logits: list[torch.Tensor] = []
    all_ordinal_logits: list[torch.Tensor] = []
    all_nominal_logits: list[torch.Tensor] = []
    all_grades: list[torch.Tensor] = []
    all_sources: list[str] = []
    all_uids: list[str] = []

    for step, (images, grades, referable, sources, uids) in enumerate(loader, start=1):
        images = images.to(device)
        grades = grades.to(device=device, dtype=torch.long)
        referable = referable.to(device=device, dtype=torch.float32)
        with torch.autocast(device_type=device.type, enabled=amp):
            binary_logits, ordinal_logits, nominal_logits = model(images)
            ordinal_loss = coral_loss(ordinal_logits, grades, ordinal_weights)
            binary_loss = nn.functional.binary_cross_entropy_with_logits(
                binary_logits,
                referable,
                pos_weight=referable_weight,
            )
            nominal_loss = nn.functional.cross_entropy(
                nominal_logits,
                grades,
                weight=nominal_weights,
                label_smoothing=nominal_label_smoothing,
            )
            combined = (
                ordinal_loss_weight * ordinal_loss
                + referable_loss_weight * binary_loss
                + nominal_loss_weight * nominal_loss
            )
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
        nominal_losses.append(float(nominal_loss.detach().cpu()))
        all_binary_logits.append(binary_logits.detach().cpu())
        all_ordinal_logits.append(ordinal_logits.detach().cpu())
        all_nominal_logits.append(nominal_logits.detach().cpu())
        all_grades.append(grades.detach().cpu())
        all_sources.extend(list(sources))
        all_uids.extend(list(uids))
        if progress_path is not None and (step == 1 or step % 25 == 0 or step == len(loader)):
            write_progress(
                progress_path,
                {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "phase": phase,
                    "epoch": epoch,
                    "batch": step,
                    "batches": len(loader),
                    "running_loss": float(np.mean(total_losses)),
                },
            )

    binary_logits = torch.cat(all_binary_logits)
    ordinal_logits = torch.cat(all_ordinal_logits)
    nominal_logits = torch.cat(all_nominal_logits)
    grades = torch.cat(all_grades)
    grade_probabilities = (
        0.5 * ordinal_probabilities(ordinal_logits)
        + 0.5 * torch.softmax(nominal_logits, dim=1)
    ).numpy()
    binary_scores = torch.sigmoid(binary_logits).numpy()
    grade_metrics = classifier_metrics(grades.numpy(), grade_probabilities)
    binary_metrics = referable_metrics(grades.numpy() >= 2, binary_scores)
    return {
        "loss": float(np.mean(total_losses)),
        "ordinal_loss": float(np.mean(ordinal_losses)),
        "referable_loss": float(np.mean(referable_losses)),
        "nominal_loss": float(np.mean(nominal_losses)),
        "binary_logits": binary_logits,
        "ordinal_logits": ordinal_logits,
        "nominal_logits": nominal_logits,
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
        "nominal_loss": result["nominal_loss"],
        "grade": result["grade_metrics"],
        "referable": result["referable_metrics"],
    }


def calibrated_evaluation(
    result: dict[str, object],
    ordinal_temperature: float,
    nominal_temperature: float,
    binary_temperature: float,
    threshold: float,
) -> dict[str, object]:
    grades = result["grades"].numpy()  # type: ignore[union-attr]
    ordinal_probabilities_value = ordinal_probabilities(
        result["ordinal_logits"], ordinal_temperature  # type: ignore[arg-type]
    )
    nominal_probabilities_value = torch.softmax(
        result["nominal_logits"] / nominal_temperature, dim=1  # type: ignore[operator]
    )
    grade_probabilities_value = (0.5 * ordinal_probabilities_value + 0.5 * nominal_probabilities_value).numpy()
    binary_scores = torch.sigmoid(
        result["binary_logits"] / binary_temperature  # type: ignore[operator]
    ).numpy()
    return {
        "grade": classifier_metrics(grades, grade_probabilities_value),
        "referable": referable_metrics(grades >= 2, binary_scores, threshold),
        "grade_probabilities": grade_probabilities_value,
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
    model, foundation = load_official_retfound_mae_cfp(
        Path(str(config["experiment"]["foundation_provenance"])),  # type: ignore[index]
        dropout=0.0,
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=float(training["tiny_overfit_learning_rate"]),
        weight_decay=0.0,
    )
    ordinal_weights = ordinal_positive_weights(dataset, device)
    binary_weight = referable_positive_weight(dataset, device)
    nominal_weights = nominal_class_weights(
        dataset,
        device,
        float(training["nominal_weight_minimum"]),
        float(training["nominal_weight_maximum"]),
    )
    common = {
        "model": model,
        "loader": loader,
        "device": device,
        "ordinal_weights": ordinal_weights,
        "referable_weight": binary_weight,
        "nominal_weights": nominal_weights,
        "ordinal_loss_weight": float(training["ordinal_loss_weight"]),
        "referable_loss_weight": float(training["referable_loss_weight"]),
        "nominal_loss_weight": float(training["nominal_loss_weight"]),
        "nominal_label_smoothing": float(training["nominal_label_smoothing"]),
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
        "foundation_model": foundation,
    }
    (run_dir / "tiny_overfit.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save({"heads": model.head_state_dict(), "config": config, "report": report}, run_dir / "tiny_overfit.pt")
    print(json.dumps({"tiny_overfit": report["status"], "run_dir": str(run_dir)}))
    if not passed:
        raise SystemExit("Tiny-overfit gate failed; refusing full V3.3 training")


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
    sampler = (
        source_grade_balanced_sampler(train_data, float(training["sampler_max_weight_ratio"]))
        if training["joint_source_grade_sampling"]
        else None
    )
    train_loader = make_loader(train_data, config, True, sampler)
    validation_loader = make_loader(validation_data, config, False)
    calibration_loader = make_loader(calibration_data, config, False)
    source_validation_loader = make_loader(source_validation_data, config, False)

    model, foundation = load_official_retfound_mae_cfp(
        Path(str(experiment["foundation_provenance"])),
        dropout=float(training["dropout"]),
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(training["epochs"]))
    ordinal_weights = ordinal_positive_weights(train_data, device)
    binary_weight = referable_positive_weight(train_data, device)
    nominal_weights = nominal_class_weights(
        train_data,
        device,
        float(training["nominal_weight_minimum"]),
        float(training["nominal_weight_maximum"]),
    )
    amp = bool(training["mixed_precision"] == "auto" and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    common = {
        "device": device,
        "ordinal_weights": ordinal_weights,
        "referable_weight": binary_weight,
        "nominal_weights": nominal_weights,
        "ordinal_loss_weight": float(training["ordinal_loss_weight"]),
        "referable_loss_weight": float(training["referable_loss_weight"]),
        "nominal_loss_weight": float(training["nominal_loss_weight"]),
        "nominal_label_smoothing": float(training["nominal_label_smoothing"]),
    }
    best_key = (float("-inf"),) * 5
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
            progress_path=run_dir / "progress.json",
            phase="training",
            epoch=epoch,
            **common,
        )
        validation_result = run_epoch(
            model,
            validation_loader,
            progress_path=run_dir / "progress.json",
            phase="validation",
            epoch=epoch,
            **common,
        )
        scheduler.step()
        key = selection_key(validation_result, selection["checkpoint_weights"])
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
            "heads": model.head_state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "selection_key": key,
            "config": config,
            "manifest_sha256": sha256(manifest),
            "foundation_model": foundation,
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
    model.load_head_state_dict(checkpoint["heads"])
    validation_result = run_epoch(
        model,
        validation_loader,
        progress_path=run_dir / "progress.json",
        phase="selected_validation",
        **common,
    )
    calibration_result = run_epoch(
        model,
        calibration_loader,
        progress_path=run_dir / "progress.json",
        phase="calibration",
        **common,
    )
    source_validation_result = run_epoch(
        model,
        source_validation_loader,
        progress_path=run_dir / "progress.json",
        phase="independent_source_validation",
        **common,
    )
    ordinal_temperature = fit_temperature(
        calibration_result["ordinal_logits"].to(device),  # type: ignore[union-attr]
        calibration_result["grades"].to(device),  # type: ignore[union-attr]
        objective="coral",
    )
    nominal_temperature = fit_temperature(
        calibration_result["nominal_logits"].to(device),  # type: ignore[union-attr]
        calibration_result["grades"].to(device),  # type: ignore[union-attr]
        objective="cross_entropy",
    )
    binary_temperature = fit_binary_temperature(
        calibration_result["binary_logits"].to(device),  # type: ignore[union-attr]
        (calibration_result["grades"] >= 2).to(device),  # type: ignore[operator]
    )
    calibrated_calibration = calibrated_evaluation(
        calibration_result,
        ordinal_temperature,
        nominal_temperature,
        binary_temperature,
        0.5,
    )
    threshold_gate = select_safety_threshold(
        calibration_result["grades"].numpy() >= 2,  # type: ignore[union-attr]
        calibrated_calibration["referable_scores"],
        float(selection["minimum_calibration_sensitivity"]),
        float(selection["minimum_calibration_specificity"]),
    )
    threshold = float(threshold_gate["selected"]["threshold"])  # type: ignore[index]
    validation_evaluation = calibrated_evaluation(
        validation_result, ordinal_temperature, nominal_temperature, binary_temperature, threshold
    )
    calibration_evaluation = calibrated_evaluation(
        calibration_result, ordinal_temperature, nominal_temperature, binary_temperature, threshold
    )
    source_validation_evaluation = calibrated_evaluation(
        source_validation_result, ordinal_temperature, nominal_temperature, binary_temperature, threshold
    )
    status = candidate_status(
        bool(threshold_gate["passed"]),
        source_validation_evaluation["referable"],
        float(selection["minimum_source_sensitivity"]),
        float(selection["minimum_source_specificity"]),
    )
    final = {
        "model_status": status,
        "official_test_used": False,
        "manifest_sha256": sha256(manifest),
        "checkpoint_sha256": sha256(run_dir / "best.pt"),
        "selected_epoch": checkpoint["epoch"],
        "selection_key": checkpoint["selection_key"],
        "foundation_model": foundation,
        "ordinal_temperature": ordinal_temperature,
        "nominal_temperature": nominal_temperature,
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
    write_progress(
        run_dir / "progress.json",
        {"updated_at": datetime.now(timezone.utc).isoformat(), "phase": "complete", "epoch": checkpoint["epoch"]},
    )
    print(json.dumps({"run_dir": str(run_dir), "model_status": final["model_status"], "threshold_gate": threshold_gate["passed"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/classifier_v3_3.yaml"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--tiny-overfit", action="store_true")
    parser.add_argument("--input-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--smoke-records-per-split",
        type=int,
        help="Run the complete pipeline on a small deterministic subset of each split",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.input_size:
        config["preprocessing"]["input_size"] = args.input_size
    if int(config["preprocessing"]["input_size"]) % 16:
        raise SystemExit("V3.3 RETFound input size must be divisible by 16")
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
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
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
                "foundation_provenance": str(config["experiment"]["foundation_provenance"]),
                "foundation_provenance_sha256": sha256(Path(config["experiment"]["foundation_provenance"])),
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
