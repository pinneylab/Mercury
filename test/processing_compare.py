"""Reusable helpers for processing pre/post regression comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mercury.processing.chip import ChipImage
from mercury.processing.experiment import Device, make_dummy_pinlist

PathLike = Union[str, Path]

DEFAULT_IMAGE = Path(
    "/home/jzhang/Documents/scope_data/20260730/bgsub_images/"
    "2026-07-30_15-02-00_d2_aura_cyan_2_5_sensitivity_2x2_1_button_quant.tif"
)
DEFAULT_CORNERS = [(387, 610), (6616, 640), (359, 6944), (6603, 6965)]
DEVICE_DIMS = (32, 56)

BUTTON_GEOMETRY_COLS = (
    "x_button_center",
    "y_button_center",
    "radius_button_disk",
)
BUTTON_VALUE_COLS = ("summed_button_BGsub",) + BUTTON_GEOMETRY_COLS


def make_chip(
    image: PathLike,
    corners: Sequence[tuple],
    dims: tuple[int, int] = DEVICE_DIMS,
) -> ChipImage:
    """Build a stamped ChipImage with a dummy pinlist."""
    image = Path(image)
    device = Device(setup="s1", dname="d1", dims=dims)
    device.pinlist = make_dummy_pinlist({i: f"block{i}" for i in range(1, 5)})
    chip = ChipImage(device, image, list(corners))
    chip.stamp()
    return chip


def run_feature_find(
    chip: ChipImage,
    feature: str = "button",
    n_jobs: int = 1,
    coerce_chamber_center: bool = False,
) -> pd.DataFrame:
    """Run chamber and/or button finding and return summarize()."""
    if feature == "button":
        chip.findButtons(n_jobs=n_jobs)
    elif feature == "chamber":
        chip.findChambers(coerce_center=coerce_chamber_center, n_jobs=n_jobs)
    elif feature == "all":
        chip.findChambers(coerce_center=coerce_chamber_center, n_jobs=n_jobs)
        chip.findButtons(n_jobs=n_jobs)
    else:
        raise ValueError("feature must be 'button', 'chamber', or 'all'")
    return chip.summarize()


def compare_summaries(
    baseline_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    value_cols: Iterable[str],
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> pd.DataFrame:
    """Align on (x, y) and assert numeric agreement on value_cols.

    Returns the outer-joined frame with ``_baseline`` / ``_candidate`` suffixes
    for columns that appear in both inputs (useful for plotting).
    """
    value_cols = list(value_cols)
    assert baseline_df.index.equals(candidate_df.index), "summary indices differ"

    for col in value_cols:
        assert col in baseline_df.columns, f"missing baseline column: {col}"
        assert col in candidate_df.columns, f"missing candidate column: {col}"

        b = baseline_df[col]
        c = candidate_df[col]
        b_null = b.isna()
        c_null = c.isna()
        assert (b_null == c_null).all(), f"null mask mismatch for {col}"

        valid = ~b_null
        if not valid.any():
            continue
        np.testing.assert_allclose(
            c.loc[valid].to_numpy(dtype=float),
            b.loc[valid].to_numpy(dtype=float),
            atol=atol,
            rtol=rtol,
            err_msg=f"mismatch on {col}",
        )

    return baseline_df.join(candidate_df, lsuffix="_baseline", rsuffix="_candidate")


def plot_unity_scatter(
    baseline_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    col: str,
    outpath: PathLike,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> Path:
    """Scatter candidate vs baseline for ``col`` with a y=x unity line."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    x = baseline_df[col].to_numpy(dtype=float)
    y = candidate_df[col].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 2:
        r = float("nan")
    else:
        r = float(np.corrcoef(x, y)[0, 1])

    lo = float(np.nanmin([x.min(), y.min()])) if len(x) else 0.0
    hi = float(np.nanmax([x.max(), y.max()])) if len(x) else 1.0
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    lims = (lo - pad, hi + pad)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, s=8, alpha=0.5, edgecolors="none")
    ax.plot(lims, lims, "k--", lw=1, label="unity")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel or f"baseline {col}")
    ax.set_ylabel(ylabel or f"candidate {col}")
    ax.set_title(title or f"{col} (r = {r:.6f})")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath
