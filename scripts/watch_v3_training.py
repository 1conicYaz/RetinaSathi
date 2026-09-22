#!/usr/bin/env python3
"""Human-readable live monitor for a RetinaSathi V3 training run."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


def percent(value: object) -> str:
    if value is None:
        return "   n/a"
    return f"{100 * float(value):6.1f}%"


def decimal(value: object) -> str:
    if value is None:
        return "   n/a"
    return f"{float(value):6.3f}"


def load_json_lines(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def render(run_dir: Path) -> str:
    history = load_json_lines(run_dir / "history.jsonl")
    final_path = run_dir / "final_validation.json"
    config_path = run_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    experiment = config.get("experiment", {})
    total_epochs = int(config.get("training", {}).get("epochs", 20))
    raw_name = str(experiment.get("name", "V3 training"))
    run_name = (
        raw_name.replace("v3_4", "V3.4")
        .replace("v3_3", "V3.3")
        .replace("v3_2", "V3.2")
        .replace("v3_1", "V3.1")
        .replace("v3_0", "V3.0")
        .replace("efficientnet_b3", "EfficientNet-B3")
        .replace("_", " ")
    )
    hardware_path = run_dir / "hardware.json"
    hardware = json.loads(hardware_path.read_text(encoding="utf-8")) if hardware_path.exists() else {}
    device = str(hardware.get("selected_device", "unknown")).upper()
    progress_path = run_dir / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    lines = [
        f"RetinaSathi {run_name} — live training monitor",
        f"Run: {run_dir.resolve()}",
        f"Updated: {datetime.now().astimezone().strftime('%d %b %Y, %I:%M:%S %p %Z')}",
        f"Compute device: {device}{' (Apple GPU)' if device == 'MPS' else ''}",
        "",
        "Ep | Train loss | Val loss |  QWK  | Macro F1 | Sens. | Spec. | AUROC | AUPRC | Grade recalls 0/1/2/3/4",
        "---+------------+----------+-------+----------+-------+-------+-------+-------+-------------------------",
    ]
    for record in history:
        validation = record["validation"]
        grade = validation["grade"]
        referral = validation["referable"]
        recalls = "/".join(f"{100 * float(value):.0f}" for value in grade["per_class_recall"])
        lines.append(
            f"{int(record['epoch']):2d} |"
            f" {float(record['train']['loss']):10.4f} |"
            f" {float(validation['loss']):8.4f} |"
            f" {decimal(grade['qwk'])} |"
            f" {decimal(grade['macro_f1'])} |"
            f" {percent(referral['sensitivity'])} |"
            f" {percent(referral['specificity'])} |"
            f" {decimal(referral['auroc'])} |"
            f" {decimal(referral['auprc'])} | {recalls}"
        )
    lines.append("")
    if final_path.exists():
        final = json.loads(final_path.read_text(encoding="utf-8"))
        source = final["source_validation"]
        minimum_sensitivity = float(final["threshold_gate"]["minimum_sensitivity"])
        minimum_specificity = float(final["threshold_gate"]["minimum_specificity"])
        source_sensitivity = float(source["referable"]["sensitivity"])
        source_specificity = float(source["referable"]["specificity"])
        source_gate_passed = (
            source_sensitivity >= minimum_sensitivity
            and source_specificity >= minimum_specificity
        )
        lines.extend(
            [
                f"TRAINING STATUS: {final['model_status']}",
                f"Selected epoch: {final['selected_epoch']}",
                f"Calibration safety gate passed: {final['threshold_gate']['passed']}",
                f"Independent source safety gate passed: {source_gate_passed}",
                f"Source-validation sensitivity: {percent(source['referable']['sensitivity'])}",
                f"Source-validation specificity: {percent(source['referable']['specificity'])}",
                f"Source-validation AUROC: {decimal(source['referable']['auroc'])}",
                f"Source-validation AUPRC: {decimal(source['referable']['auprc'])}",
                (
                    "DECISION: eligible for the next locked evaluation stage."
                    if source_gate_passed
                    else "DECISION: hold the locked test; improve the candidate first."
                ),
                f"Final report: {final_path.resolve()}",
            ]
        )
    elif history:
        age_seconds = max(0, int(time.time() - (run_dir / "history.jsonl").stat().st_mtime))
        age_minutes, age_remainder = divmod(age_seconds, 60)
        spinner = "|/-\\"[int(time.time()) % 4]
        lines.extend(
            [
                f"STATUS {spinner}: waiting for the next completed epoch",
                f"Completed epochs: {len(history)} of at most {total_epochs}",
                (
                    f"Current work: {progress.get('phase', 'training')} · epoch {progress.get('epoch', len(history) + 1)} · "
                    f"batch {progress.get('batch', '?')} of {progress.get('batches', '?')}"
                ),
                f"Current running loss: {decimal(progress.get('running_loss'))}",
                f"Time since last completed epoch: {age_minutes}m {age_remainder:02d}s",
                "A new result row appears only when the whole epoch finishes (about 13–15 minutes).",
                "Final calibration and source validation run after training stops.",
                "Do not present these intermediate values as final model accuracy.",
            ]
        )
    else:
        provenance_path = run_dir / "provenance.json"
        elapsed = "unknown"
        if provenance_path.exists():
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            created_at = provenance.get("created_at")
            if created_at:
                started = datetime.strptime(str(created_at), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
                seconds = max(0, int((datetime.now(timezone.utc) - started).total_seconds()))
                minutes, remainder = divmod(seconds, 60)
                elapsed = f"{minutes}m {remainder:02d}s"
        spinner = "|/-\\"[int(time.time()) % 4]
        lines.extend(
            [
                f"STATUS {spinner}: first epoch is currently running",
                f"Completed epochs: 0 of at most {total_epochs}",
                f"Elapsed since run initialization: {elapsed}",
                (
                    f"Current work: {progress.get('phase', 'initializing')} · epoch {progress.get('epoch', 1)} · "
                    f"batch {progress.get('batch', '?')} of {progress.get('batches', '?')}"
                ),
                f"Current running loss: {decimal(progress.get('running_loss'))}",
                "The first metrics row appears after the complete training and validation epoch.",
                "Keep this window open; it refreshes automatically.",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float, default=10.0)
    args = parser.parse_args()
    if not args.run_dir.is_dir():
        raise SystemExit(f"Run directory does not exist: {args.run_dir}")
    while True:
        if not args.once:
            os.system("clear")
        print(render(args.run_dir), flush=True)
        if args.once or (args.run_dir / "final_validation.json").exists():
            break
        time.sleep(max(args.interval, 1.0))


if __name__ == "__main__":
    main()
