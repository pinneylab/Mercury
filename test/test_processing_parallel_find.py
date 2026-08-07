"""Unit tests for parallel chamber/button finding across stamps."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from skimage import io

from mercury.processing.chip import ChipImage, Stamp, _resolve_n_jobs
from mercury.processing.experiment import Device, read_pinlist

from processing_compare import (
    BUTTON_VALUE_COLS,
    DEFAULT_CORNERS,
    DEFAULT_IMAGE,
    compare_summaries,
    make_chip,
    plot_unity_scatter,
    run_feature_find,
)


def _make_pinlist(nx: int, ny: int) -> pd.DataFrame:
    rows = [
        {"Indices": str((x + 1, y + 1)), "MutantID": f"m{x + 1}_{y + 1}"}
        for x in range(nx)
        for y in range(ny)
    ]
    return read_pinlist(pd.DataFrame(rows))


def _build_chip_with_stamps(tmp_path, stamp_arrays):
    """Build a ChipImage whose stamps are the given (nx, ny) stamp arrays."""
    nx, ny = stamp_arrays.shape
    h, w = stamp_arrays[0, 0].shape
    assert h == ChipImage.stampWidth and w == ChipImage.stampWidth

    tmp_path.mkdir(parents=True, exist_ok=True)

    # Minimal raster large enough for a  nx x ny grid of non-overlapping stamps
    img = np.zeros((ny * h + h, nx * w + w), dtype=np.uint16)
    for x in range(nx):
        for y in range(ny):
            cy = h // 2 + y * h
            cx = w // 2 + x * w
            img[cy - h // 2 : cy + h // 2, cx - w // 2 : cx + w // 2] = stamp_arrays[x, y]

    path = tmp_path / "synthetic_chip.tif"
    io.imsave(path, img, check_contrast=False)

    corners = (
        (w // 2, h // 2),
        (w // 2 + (nx - 1) * w, h // 2),
        (w // 2, h // 2 + (ny - 1) * h),
        (w // 2 + (nx - 1) * w, h // 2 + (ny - 1) * h),
    )
    device = Device(setup="s1", dname="d1", dims=(nx, ny))
    device.pinlist = _make_pinlist(nx, ny)
    chip = ChipImage(device, path, corners)
    chip.stamp()
    return chip


def _bright_disk_stamp(radius: int, center=None, peak: int = 40000) -> np.ndarray:
    import cv2

    size = ChipImage.stampWidth
    if center is None:
        center = (size // 2, size // 2)
    stamp = np.full((size, size), 1000, dtype=np.uint16)
    cv2.circle(stamp, (int(center[0]), int(center[1])), radius, int(peak), -1)
    return stamp


def test_resolve_n_jobs():
    assert _resolve_n_jobs(1) == 1
    assert _resolve_n_jobs(4) == 4
    assert _resolve_n_jobs(-1) >= 1


def test_find_chambers_serial_matches_parallel(tmp_path):
    nx, ny = 2, 2
    stamps = np.empty((nx, ny), dtype=object)
    for x in range(nx):
        for y in range(ny):
            # Offset slightly so Hough has a clear circle; radius in chamber search band
            stamps[x, y] = _bright_disk_stamp(
                radius=Stamp.chamberrad, center=(48 + x, 50 + y)
            )

    chip_serial = _build_chip_with_stamps(tmp_path / "serial", stamps)
    chip_parallel = _build_chip_with_stamps(tmp_path / "parallel", stamps)

    chip_serial.findChambers(n_jobs=1)
    chip_parallel.findChambers(n_jobs=2)

    for s_serial, s_parallel in zip(
        chip_serial.stamps.flatten(), chip_parallel.stamps.flatten()
    ):
        assert s_serial.chamber.blankFlag == s_parallel.chamber.blankFlag
        if s_serial.chamber.blankFlag:
            continue
        assert s_serial.chamber.center == s_parallel.chamber.center
        assert s_serial.chamber.radius == s_parallel.chamber.radius


@pytest.mark.slow
def test_find_buttons_serial_vs_parallel_real_image(tmp_path):
    """Pre (n_jobs=1) vs post (n_jobs=2) button find on a real chip image."""
    if not DEFAULT_IMAGE.exists():
        pytest.skip(f"Regression image not found at {DEFAULT_IMAGE}")

    chip_base = make_chip(DEFAULT_IMAGE, DEFAULT_CORNERS)
    df_base = run_feature_find(chip_base, feature="button", n_jobs=1)

    chip_new = make_chip(DEFAULT_IMAGE, DEFAULT_CORNERS)
    df_new = run_feature_find(chip_new, feature="button", n_jobs=2)

    compare_summaries(df_base, df_new, BUTTON_VALUE_COLS, atol=0.0, rtol=0.0)

    outpath = tmp_path / "button_find_parallel_unity.png"
    plot_unity_scatter(
        df_base,
        df_new,
        col="summed_button_BGsub",
        outpath=outpath,
        title="Button find: parallel vs serial (summed_button_BGsub)",
        xlabel="serial n_jobs=1",
        ylabel="parallel n_jobs=2",
    )
    assert outpath.exists() and outpath.stat().st_size > 0
    print(f"Unity scatter written to {outpath}")
