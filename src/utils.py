"""
utils.py
---------
Shared utility functions used across the ML pipeline.
"""

import os
import logging
import json
from datetime import datetime

log = logging.getLogger(__name__)


def ensure_dirs(*dirs):
    """Create directories if they don't exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        log.debug("Ensured directory: %s", d)


def log_metrics(metrics: dict, label: str = "Metrics") -> None:
    """Pretty-print a dict of metrics to the logger."""
    log.info("── %s ──────────────────────────", label)
    for k, v in metrics.items():
        if isinstance(v, float):
            log.info("  %-30s %.4f", k, v)
        else:
            log.info("  %-30s %s", k, v)


def save_json(data: dict, path: str) -> None:
    """Persist a dict as JSON (creates parent dirs automatically)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    log.info("Saved JSON → %s", path)


def load_json(path: str) -> dict:
    """Load a JSON file and return as dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def timestamp_str() -> str:
    """Return current timestamp as a compact string (for filenames)."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_pkr(amount: float) -> str:
    """Format a Pakistani Rupee amount with commas and two decimal places."""
    return f"PKR {amount:,.2f}"
