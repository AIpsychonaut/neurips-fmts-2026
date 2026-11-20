"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparse_time_probes.figures import make_all


def figures_entry() -> None:
    parser = argparse.ArgumentParser(description="Render paper figures.")
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    for path in make_all(args.out):
        print(path)


def control_task_entry() -> None:
    from sparse_time_probes.experiments import run_control_task

    parser = argparse.ArgumentParser(description="Index-only control task (CPU).")
    parser.add_argument("--out", type=Path, default=Path("results/control_task.json"))
    args = parser.parse_args()
    payload = run_control_task()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.out)
