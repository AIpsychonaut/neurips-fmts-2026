"""Apply the pre-registered series-selection rule to the local UCR 2018 archive.

Rule: every univariate dataset; first TRAIN row; drop NaN padding; truncate to
first 512 steps; keep if length >= 128 and std > 0; z-score.
Also writes the 8-series legacy roster (truncated at 400) for the replication check.

python cr/prep_series.py
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "data" / "UCRArchive_2018.zip"
OUT = Path(__file__).resolve().parent / "series.npz"
LEGACY = ["Coffee", "Wine", "Strawberry", "Meat", "Ham", "Beef", "Fish", "OliveOil"]


def first_row(text: str) -> np.ndarray:
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.replace(",", "\t").split("\t") if "\t" in line or "," in line else line.split()
        vals = []
        for p in parts[1:]:
            p = p.strip()
            if p == "" or p.lower() == "nan":
                vals.append(np.nan)
            else:
                vals.append(float(p))
        return np.asarray(vals, dtype=np.float64)
    raise ValueError("empty")


def z(x):
    s = x.std()
    return (x - x.mean()) / s


def main():
    zf = zipfile.ZipFile(ZIP)
    trains = sorted(n for n in zf.namelist() if n.endswith("_TRAIN.tsv")
                    and "Missing_value_and_variable_length_datasets_adjusted" not in n)
    names, data, skipped = [], {}, {}
    legacy = {}
    for path in trains:
        name = Path(path).parent.name
        raw = zf.read(path, pwd=b"someone").decode("utf-8", errors="ignore")
        y = first_row(raw)
        y = y[~np.isnan(y)]
        if name in LEGACY:
            legacy[name] = z(y[:400])
        y = y[:512]
        if y.size < 128:
            skipped[name] = f"len {y.size}"
            continue
        if y.std() < 1e-9:
            skipped[name] = "constant"
            continue
        names.append(name)
        data[name] = z(y)
    arrays = {f"s__{n}": data[n] for n in names}
    arrays.update({f"legacy__{n}": legacy[n] for n in LEGACY})
    np.savez_compressed(OUT, **arrays)
    meta = {"n_series": len(names), "names": names, "lengths": {n: int(data[n].size) for n in names},
            "skipped": skipped, "n_datasets_seen": len(trains)}
    (OUT.parent / "series_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"datasets {len(trains)}  kept {len(names)}  skipped {len(skipped)}")
    print("lengths: min", min(meta['lengths'].values()), "median", int(np.median(list(meta['lengths'].values()))), "max", max(meta['lengths'].values()))


if __name__ == "__main__":
    main()
