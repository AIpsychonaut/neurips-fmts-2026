#!/usr/bin/env python3
"""Render paper figures to ./figures (PDF + PNG)."""

from __future__ import annotations

import argparse
from pathlib import Path

import _path  # noqa: F401

from sparse_time_probes.figures import make_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "figures")
    args = parser.parse_args()
    for path in make_all(args.out):
        print(path)


if __name__ == "__main__":
    main()
