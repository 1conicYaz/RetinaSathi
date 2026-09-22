"""Controlled classifier comparisons on identical leakage-free subsets."""

from __future__ import annotations

import argparse
import copy
import json
import resource
import time
from collections import defaultdict
from itertools import product
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from ml.artifact_integrity import sha256
from ml.models import ARCHITECTURES
from ml.training.train_classifier_v2 import (
    RetinalTransform,
    build_model,
    device_for_training,
    grade_loss_weights,
    load_aptos_backbone,
    make_dataset,
    run_epoch,
    seed_everything,
)


def balanced_indices(dataset, per_class: int) -> list[int]:
    groups = defaultdict(list)
    for index, row in enumerate(dataset.rows):
        groups[int(row["grade"])].append(index)
    return [index for grade in sorted(groups) for index in groups[grade][:per_class]]


def subset_weighted_sampler(dataset, indices: list[int]) -> WeightedRandomSampler:
    labels = [int(dataset.rows[index]["grade"]) for index in indices]
    counts = {label: labels.count(label) for label in set(labels)}
    weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(weights, len(weights), replacement=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/classifier.yaml"))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/classifier_pilot_summary.json"),
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--per-class-train", type=int, default=30)
    parser.add_argument("--per-class-validation", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES)
    parser.add_argument(
        "--dataset", choices=("APTOS_2019", "IDRiD"), default="APTOS_2019"
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        choices=("coral", "cross_entropy"),
        default=["coral"],
    )
    parser.add_argument("--input-sizes", nargs="+", type=int)
    parser.add_argument("--dme-weights", nargs="+", type=float, default=[0.0])
    parser.add_argument(
        "--imbalance-strategies",
        nargs="+",
        choices=("sampler_only", "weighted_loss", "weighted_sampler", "none"),
        default=["weighted_loss"],
    )
    parser.add_argument(
        "--subset-strategy", choices=("balanced", "full"), default="balanced"
    )
    parser.add_argument("--backbone-checkpoint", type=Path)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Pilot summary already exists: {args.output}")
    base_config = yaml.safe_load(args.config.read_text())
    base_config["training"]["batch_size"] = args.batch_size
    architectures = args.architectures or list(
        base_config["experiment"]["pilot_architectures"]
    )
    input_sizes = args.input_sizes or [int(base_config["preprocessing"]["input_size"])]
    seed = int(base_config["experiment"]["seed"])
    device = device_for_training()
    results = []
    for input_size in input_sizes:
        for objective in args.objectives:
            for dme_weight, imbalance_strategy in product(
                args.dme_weights, args.imbalance_strategies
            ):
                config = copy.deepcopy(base_config)
                config["preprocessing"]["input_size"] = input_size
                config["training"]["objective"] = objective
                config["training"]["dme_weight"] = dme_weight
                config["training"]["imbalance"] = imbalance_strategy
                split_dir = Path(config["data"]["split_dir"])
                train_data = make_dataset(
                    args.data_root,
                    split_dir,
                    args.dataset,
                    "train",
                    RetinalTransform(config, True),
                )
                validation_data = make_dataset(
                    args.data_root,
                    split_dir,
                    args.dataset,
                    "validation",
                    RetinalTransform(config, False),
                )
                train_indices = (
                    balanced_indices(train_data, args.per_class_train)
                    if args.subset_strategy == "balanced"
                    else list(range(len(train_data)))
                )
                train_subset = Subset(train_data, train_indices)
                validation_subset = Subset(
                    validation_data,
                    balanced_indices(validation_data, args.per_class_validation),
                )
                grade_weights = (
                    grade_loss_weights(train_data, objective, device)
                    if imbalance_strategy in ("weighted_loss", "weighted_sampler")
                    else None
                )
                train_sampler = (
                    subset_weighted_sampler(train_data, train_indices)
                    if imbalance_strategy in ("sampler_only", "weighted_sampler")
                    else None
                )
                for architecture in architectures:
                    seed_everything(seed)
                    config["experiment"]["architecture"] = architecture
                    model = build_model(
                        config,
                        pretrained=(
                            not args.no_pretrained
                            and args.backbone_checkpoint is None
                        ),
                    ).to(device)
                    if args.backbone_checkpoint is not None:
                        load_aptos_backbone(
                            model, args.backbone_checkpoint, architecture
                        )
                    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=float(config["training"]["learning_rate"]),
                        weight_decay=float(config["training"]["weight_decay"]),
                    )
                    train_loader = DataLoader(
                        train_subset,
                        batch_size=args.batch_size,
                        sampler=train_sampler,
                        shuffle=train_sampler is None,
                        num_workers=0,
                    )
                    validation_loader = DataLoader(
                        validation_subset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=0,
                    )
                    started = time.perf_counter()
                    history = []
                    for epoch in range(1, args.epochs + 1):
                        train_loss, _, _, train_metrics = run_epoch(
                            model,
                            train_loader,
                            device,
                            objective,
                            grade_weights,
                            dme_weight,
                            optimizer,
                        )
                        validation_loss, _, _, validation_metrics = run_epoch(
                            model,
                            validation_loader,
                            device,
                            objective,
                            grade_weights,
                            dme_weight,
                        )
                        history.append(
                            {
                                "epoch": epoch,
                                "train_loss": train_loss,
                                "validation_loss": validation_loss,
                                "train": train_metrics,
                                "validation": validation_metrics,
                            }
                        )
                    results.append(
                        {
                            "architecture": architecture,
                            "objective": objective,
                            "dme_weight": dme_weight,
                            "imbalance_strategy": imbalance_strategy,
                            "dataset": args.dataset,
                            "pretrained": (
                                not args.no_pretrained
                                and args.backbone_checkpoint is None
                            ),
                            "initialization": (
                                "aptos_checkpoint"
                                if args.backbone_checkpoint is not None
                                else (
                                    "random"
                                    if args.no_pretrained
                                    else "imagenet"
                                )
                            ),
                            "input_size": input_size,
                            "epochs": args.epochs,
                            "train_samples": len(train_subset),
                            "validation_samples": len(validation_subset),
                            "elapsed_seconds": round(
                                time.perf_counter() - started, 2
                            ),
                            "peak_rss_bytes": resource.getrusage(
                                resource.RUSAGE_SELF
                            ).ru_maxrss,
                            "parameters": sum(
                                parameter.numel() for parameter in model.parameters()
                            ),
                            "final": history[-1],
                        }
                    )
                    latest = results[-1]
                    print(
                        json.dumps(
                            {
                                "comparison_complete": len(results),
                                "architecture": architecture,
                                "objective": objective,
                                "input_size": input_size,
                                "dme_weight": dme_weight,
                                "imbalance_strategy": imbalance_strategy,
                                "validation_qwk": latest["final"]["validation"][
                                    "qwk"
                                ],
                                "official_test_used": False,
                            }
                        ),
                        flush=True,
                    )
                    del model
    selected = max(
        results,
        key=lambda item: (
            item["final"]["validation"]["qwk"],
            item["final"]["validation"]["macro_f1"],
        ),
    )
    summary = {
        "status": "controlled_pilot_not_final_model",
        "seed": seed,
        "device": str(device),
        "official_test_used": False,
        "config_sha256": sha256(args.config),
        "split_sha256": sha256(
            Path(base_config["data"]["split_dir"])
            / (
                "idrid_internal_train_validation.csv"
                if args.dataset == "IDRiD"
                else "aptos_train_validation.csv"
            )
        ),
        "backbone_checkpoint_sha256": (
            sha256(args.backbone_checkpoint)
            if args.backbone_checkpoint is not None
            else None
        ),
        "subset_rule": (
            "first N records per class from the immutable split manifest"
            if args.subset_strategy == "balanced"
            else "all training records from the immutable split manifest"
        ),
        "selection_rule": "highest pilot validation QWK, then macro F1; confirm with full training",
        "provisional_selection": {
            key: selected[key]
            for key in (
                "architecture",
                "objective",
                "input_size",
                "dme_weight",
                "imbalance_strategy",
            )
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "provisional_selection": summary["provisional_selection"],
                "official_test_used": False,
            }
        )
    )


if __name__ == "__main__":
    main()
