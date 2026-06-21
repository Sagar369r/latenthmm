import hashlib
import os
import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def load_verified_checkpoint(
    path:              str,
    expected_sha256:   Optional[str] = None,
    map_location:      str           = "cpu",
) -> dict:
    """Load state dict and optionally verify SHA-256 hash."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if expected_sha256 is not None:
        with open(path, "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        if actual != expected_sha256:
            raise ValueError(
                f"Checkpoint integrity check FAILED for {path}. "
                f"Expected {expected_sha256}, got {actual}. "
                "Model may have been tampered with."
            )
        logger.info("Checkpoint integrity verified: %s", path)

    return torch.load(path, map_location=map_location, weights_only=True)
