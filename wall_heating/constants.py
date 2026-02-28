"""Common constants for the wall-heating project."""

from pathlib import Path

DEFAULT_LOAD_DIR = Path("/dtu/projects/02613_2025/data/modified_swiss_dwellings")
GRID_SIZE = 512
PADDED_SIZE = GRID_SIZE + 2
DEFAULT_MAX_ITER = 20_000
DEFAULT_ABS_TOL = 1e-4
STATS_KEYS = ("mean_temp", "std_temp", "pct_above_18", "pct_below_15")
