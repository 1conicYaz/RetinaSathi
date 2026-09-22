"""Create an immutable hash lock before any official-test access."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ml.artifact_integrity import sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/classifier_lock.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Lock already exists: {args.output}. Do not overwrite a model lock.")
    validation = json.loads(args.validation.read_text())
    if validation.get("official_test_used") is not False:
        raise SystemExit("Refusing to lock: validation provenance does not prove official_test_used=false")
    checkpoint_digest = sha256(args.checkpoint)
    if validation.get("checkpoint_sha256") != checkpoint_digest:
        raise SystemExit("Refusing to lock: validation artifact was not generated from this checkpoint")
    selection = json.loads(args.selection.read_text())
    selected = selection.get("selected", {})
    if selection.get("official_test_used") is not False or selected.get("checkpoint_sha256") != checkpoint_digest or selected.get("validation_sha256") != sha256(args.validation):
        raise SystemExit("Refusing to lock: selection record does not bind this checkpoint and validation")
    lock = {
        "status": "locked_pending_test",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": checkpoint_digest,
        "validation_sha256": sha256(args.validation),
        "selection_sha256": sha256(args.selection),
        "temperature": validation["temperature"],
        "referable_threshold": validation["referable_threshold"],
        "official_test_used_before_lock": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2) + "\n")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
