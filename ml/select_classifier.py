"""Select one classifier using internal validation and immutable comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ml.artifact_integrity import require_internal_validation_binding, sha256


def factor_coverage(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Collect factor levels proven by internal-only comparison rows."""
    rows = [row for summary in summaries for row in summary.get("results", [])]
    architectures = sorted(
        {str(row["architecture"]) for row in rows if "architecture" in row}
    )
    objectives = sorted({str(row["objective"]) for row in rows if "objective" in row})
    input_sizes = sorted({int(row["input_size"]) for row in rows if "input_size" in row})
    dme_weights = sorted({float(row["dme_weight"]) for row in rows if "dme_weight" in row})
    imbalance_strategies = sorted(
        {
            str(row["imbalance_strategy"])
            for row in rows
            if "imbalance_strategy" in row
        }
    )
    return {
        "architectures": architectures,
        "objectives": objectives,
        "input_sizes": input_sizes,
        "dme_weights": dme_weights,
        "imbalance_strategies": imbalance_strategies,
        "rows": len(rows),
    }


def require_master_comparisons(coverage: dict[str, Any]) -> None:
    """Enforce the controlled comparisons required before classifier lock."""
    missing = []
    if len(coverage["architectures"]) < 2:
        missing.append("at least two architectures")
    if not {"coral", "cross_entropy"}.issubset(coverage["objectives"]):
        missing.append("CORAL and cross-entropy objectives")
    if not {384, 512}.issubset(coverage["input_sizes"]):
        missing.append("384 and 512 input sizes")
    dme_weights = coverage["dme_weights"]
    if not any(weight == 0 for weight in dme_weights) or not any(
        weight > 0 for weight in dme_weights
    ):
        missing.append("DR-only and multitask DME weights")
    if not {"sampler_only", "weighted_loss"}.issubset(
        coverage["imbalance_strategies"]
    ):
        missing.append("sampler-only and weighted-loss imbalance strategies")
    if missing:
        raise ValueError("Missing controlled comparison coverage: " + "; ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="NAME|CHECKPOINT|VALIDATION_JSON (repeatable)",
    )
    parser.add_argument(
        "--comparison-summary", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/classifier_selection.json"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Selection record already exists: {args.output}")

    candidates = []
    for specification in args.candidate:
        parts = specification.split("|")
        if len(parts) != 3:
            raise SystemExit("Each --candidate must be NAME|CHECKPOINT|VALIDATION_JSON")
        name, checkpoint_value, validation_value = parts
        checkpoint = Path(checkpoint_value)
        validation_path = Path(validation_value)
        validation = json.loads(validation_path.read_text())
        try:
            checkpoint_digest = require_internal_validation_binding(
                checkpoint, validation
            )
        except ValueError as error:
            raise SystemExit(f"Invalid provenance for candidate {name}: {error}") from error
        candidates.append(
            {
                "name": name,
                "checkpoint_sha256": checkpoint_digest,
                "validation_sha256": sha256(validation_path),
                "qwk": validation["validation"]["qwk"],
                "macro_f1": validation["validation"]["macro_f1"],
                "referable_sensitivity": validation["validation"][
                    "referable_sensitivity"
                ],
                "referable_specificity": validation["validation"][
                    "referable_specificity"
                ],
            }
        )

    summaries = []
    comparisons = []
    for path in args.comparison_summary:
        summary = json.loads(path.read_text())
        if summary.get("official_test_used") is not False:
            raise SystemExit(
                f"Comparison summary lacks internal-only provenance: {path}"
            )
        summaries.append(summary)
        comparisons.append(
            {
                "sha256": sha256(path),
                "status": summary.get("status"),
                "provisional_selection": summary.get("provisional_selection"),
            }
        )
    coverage = factor_coverage(summaries)
    try:
        require_master_comparisons(coverage)
    except ValueError as error:
        raise SystemExit(f"Refusing selection: {error}") from error

    selected = max(
        candidates,
        key=lambda item: (item["qwk"], item["macro_f1"]),
    )
    result = {
        "status": "selected_internal_validation_only",
        "selection_rule": "highest internal-validation QWK, then macro F1",
        "official_test_used": False,
        "selected": selected,
        "candidates": candidates,
        "comparison_summaries": comparisons,
        "comparison_coverage": coverage,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected": selected["name"],
                "official_test_used": False,
            }
        )
    )


if __name__ == "__main__":
    main()
