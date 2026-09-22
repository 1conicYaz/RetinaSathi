"""Hash and provenance gates shared by model validation and export tools."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading large artifacts at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_internal_validation_binding(
    checkpoint: Path, validation: Mapping[str, Any]
) -> str:
    """Require internal-only validation generated from the exact checkpoint."""
    checkpoint_digest = sha256(checkpoint)
    if validation.get("official_test_used") is not False:
        raise ValueError("Validation does not prove official_test_used=false")
    if validation.get("checkpoint_sha256") != checkpoint_digest:
        raise ValueError("Validation artifact does not match the checkpoint")
    return checkpoint_digest


def require_locked_evaluation_binding(
    checkpoint: Path,
    evaluation: Mapping[str, Any],
    validation: Path | None = None,
) -> str:
    """Require a completed post-lock evaluation for the exact checkpoint."""
    checkpoint_digest = sha256(checkpoint)
    if evaluation.get("status") != "locked_test_and_external_evaluation_complete":
        raise ValueError("Evaluation is not a completed locked evaluation")
    if evaluation.get("official_test_used_after_lock") is not True:
        raise ValueError("Evaluation does not prove official test use occurred after lock")
    if evaluation.get("checkpoint_sha256") != checkpoint_digest:
        raise ValueError("Locked evaluation does not match the checkpoint")
    if validation is not None and evaluation.get("validation_sha256") != sha256(
        validation
    ):
        raise ValueError("Locked evaluation does not match the validation artifact")
    return checkpoint_digest
