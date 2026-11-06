#!/usr/bin/env python3
"""Chronos-T5-base encoder audit (GPU). Optional; figures do not require this."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _path  # noqa: F401

from sparse_time_probes.experiments import run_chronos_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="directory for the UCR 2018 archive",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "chronos_audit.json",
    )
    args = parser.parse_args()
    payload = run_chronos_audit(args.cache)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
