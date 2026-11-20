#!/usr/bin/env python3
"""Recompute index-only control-task ceilings (CPU, no Chronos weights)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _path  # noqa: F401

from sparse_time_probes.experiments import run_control_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "control_task.json",
    )
    args = parser.parse_args()
    payload = run_control_task()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["medians"], indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
