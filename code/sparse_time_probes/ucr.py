"""UCR Archive 2018 loader (first TRAIN series, truncated to the paper cap)."""

from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

from sparse_time_probes.config import T_CAP
from sparse_time_probes.geometry import zscore

UCR_URL = "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip"
# Password published by the UCR Archive maintainers.
UCR_ZIP_PASSWORD = b"someone"


def _parse_train(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            values = [float(p) for p in parts[1:]]
            if values:
                return np.asarray(values, dtype=np.float64)
    raise ValueError(f"no numeric series in {path}")


def download_archive(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "UCRArchive_2018.zip"
    if archive.is_file() and archive.stat().st_size > 1_000_000:
        return archive
    req = Request(UCR_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=600) as resp, archive.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    return archive


def load_series(name: str, cache_dir: Path) -> np.ndarray:
    extract_root = cache_dir / "UCRArchive_2018"
    train = extract_root / name / f"{name}_TRAIN.tsv"
    if not train.is_file():
        archive = download_archive(cache_dir)
        with zipfile.ZipFile(archive) as zf:
            members = [
                m
                for m in zf.namelist()
                if f"/{name}/" in m.replace("\\", "/") and not m.endswith("/")
            ]
            if not members:
                raise FileNotFoundError(name)
            for member in members:
                zf.extract(member, path=cache_dir, pwd=UCR_ZIP_PASSWORD)
    y = _parse_train(train)
    if y.size > T_CAP:
        y = y[:T_CAP]
    return zscore(y)
