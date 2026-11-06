"""Frozen paper numbers used by figures (no model names from the lab trail)."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any


def load_paper_results() -> dict[str, Any]:
    pkg = resources.files("sparse_time_probes") / "assets" / "paper_results.json"
    return json.loads(pkg.read_text(encoding="utf-8"))


def write_paper_results(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
