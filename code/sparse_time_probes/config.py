"""Protocol constants matching the paper."""

from __future__ import annotations

MODEL_ID = "amazon/chronos-t5-base"

DATASETS = (
    "Coffee",
    "Wine",
    "Strawberry",
    "Meat",
    "Ham",
    "Beef",
    "Fish",
    "OliveOil",
)

KEEP_ALPHA = 0.35
KEEP_BETA = 4.0
P_PRIMARY = 0.5
P_GRID = (0.0, 0.3, 0.5)
LAYERS = (4, 8, 12)
PRIMARY_LAYER = 12

RIDGE_ALPHA = 1.0
N_SPLITS = 5
BASE_SEED = 42
T_CAP = 400

SPLINE_KNOTS = 8
SPLINE_DEGREE = 3
POLY_DEGREE = 3

# Illustrative series for Figure 1 (T = 400 after cap).
FIGURE1_DATASET = "Meat"
