"""Lightweight single-raster stitching performance harness (wall time + peak RSS).

Not run in default CI; mark as slow and skip if bench data is absent.
"""

from __future__ import annotations

import os
import resource
import time
from pathlib import Path

import pytest
from skimage import io

from mercury.stitching import stitch_single_raster
from mercury.stitching import rastering

DEFAULT_BENCH_ROOT = Path("/home/jzhang/Documents/scope_data/20260730")
DEFAULT_RASTER_REL = Path(
    "raw_images/2026-07-30_10-54-29_d1_aura_cyan_2_10_dynamic_range_2x2_1_test"
)
TILE_SIZE = 1600


def _bench_root() -> Path:
    return Path(os.environ.get("MERCURY_STITCH_BENCH_DIR", DEFAULT_BENCH_ROOT))


def _read_vmhwm_kb() -> int | None:
    """Peak RSS (VmHWM) in kB from /proc/self/status, or None if unavailable."""
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return None
    for line in status_path.read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1])
    return None


def _peak_rss_kb() -> tuple[int | None, int]:
    vmhwm = _read_vmhwm_kb()
    # Linux: ru_maxrss is already in kilobytes
    ru_maxrss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return vmhwm, ru_maxrss


@pytest.mark.slow
def test_single_raster_stitch_perf(tmp_path):
    root = _bench_root()
    raster_dir = root / DEFAULT_RASTER_REL
    imaging_csv = root / "imaging.csv"

    if not imaging_csv.exists() or not raster_dir.is_dir():
        pytest.skip(f"Stitch bench data not found at {root} / {DEFAULT_RASTER_REL}")

    import pandas as pd

    df = pd.read_csv(imaging_csv)
    parent_key = str(DEFAULT_RASTER_REL).replace("\\", "/")
    mask = df["image_path_parent"].astype(str).str.replace("\\", "/") == parent_key
    group = df.loc[mask].sort_values(by=["raster_col_index", "raster_row_index"])
    if group.empty:
        pytest.skip(f"No imaging.csv rows for {parent_key}")

    overlap = float(group["raster_overlap"].iloc[0])
    width = int(group["raster_width"].iloc[0])
    height = int(group["raster_height"].iloc[0])
    refs = [(root / p).resolve() for p in group["image_path"].tolist()]

    params = rastering.RasterParams(
        overlap=overlap,
        size=TILE_SIZE,
        acqui_ori=(True, False),
        rotation=0.0,
        dims=(width, height),
        auto_ff=False,
        ff_type="BaSiC",
    )

    outpath = tmp_path / "bench_stitched.tif"
    margin = int(TILE_SIZE * overlap / 2)
    retained = TILE_SIZE - 2 * margin
    # Historical cut geometry: (rows * retained, cols * retained)
    expected_shape = (height * retained, width * retained)

    t0 = time.perf_counter()
    ok, path, err = stitch_single_raster(refs, params, outpath, method="cut")
    elapsed = time.perf_counter() - t0
    vmhwm_kb, ru_maxrss_kb = _peak_rss_kb()

    print(
        f"stitch_perf raster={DEFAULT_RASTER_REL.name} "
        f"wall_s={elapsed:.3f} "
        f"VmHWM_kB={vmhwm_kb} "
        f"ru_maxrss_kB={ru_maxrss_kb}"
    )

    assert ok, err
    assert path.exists()
    stitched = io.imread(path)
    assert stitched.shape == expected_shape
