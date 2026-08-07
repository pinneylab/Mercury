"""Tests for RoiSet export, quantify parity, and render_summary."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from skimage import io

from mercury.processing.chip import ChipImage, Stamp
from mercury.processing.experiment import Device, read_pinlist
from mercury.processing.roi import (
    BUTTON_COLS,
    CHAMBER_COLS,
    RoiSet,
    extract_stamps,
    quantify,
    render_summary,
    stamps_from_chip,
)

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


def _bright_disk_stamp(radius: int, center=None, peak: int = 40000) -> np.ndarray:
    import cv2

    size = ChipImage.stampWidth
    if center is None:
        center = (size // 2, size // 2)
    stamp = np.full((size, size), 1000, dtype=np.uint16)
    cv2.circle(stamp, (int(center[0]), int(center[1])), radius, int(peak), -1)
    return stamp


def _build_chip_with_stamps(tmp_path, stamp_arrays):
    nx, ny = stamp_arrays.shape
    h, w = stamp_arrays[0, 0].shape
    tmp_path.mkdir(parents=True, exist_ok=True)
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


def test_roisets_from_chip_chamber_masks(tmp_path):
    nx, ny = 2, 2
    stamps = np.empty((nx, ny), dtype=object)
    for x in range(nx):
        for y in range(ny):
            stamps[x, y] = _bright_disk_stamp(
                radius=Stamp.chamberrad, center=(48 + x, 50 + y)
            )
    chip = _build_chip_with_stamps(tmp_path / "ch", stamps)
    chip.findChambers(n_jobs=1)

    roi = RoiSet.from_chip(chip, features="chamber")
    assert roi.n == 4
    assert roi.chamber_masks.shape == (4, *roi.stamp_shape)
    assert roi.button_disk_masks is None

    flat = list(chip.stamps.flatten())
    for i, s in enumerate(flat):
        if s.chamber.blankFlag:
            assert roi.chamber_blank[i]
            continue
        assert not roi.chamber_blank[i]
        expected_inside = ~np.asarray(s.chamber.disk, dtype=bool)
        np.testing.assert_array_equal(roi.chamber_masks[i], expected_inside)
        p = Stamp.circularSubsection(
            s.data, s.chamber.center, s.chamber.radius
        )
        np.testing.assert_array_equal(roi.chamber_masks[i], ~p["mask"])


def test_quantify_matches_summarize_synthetic_buttons(tmp_path):
    nx, ny = 2, 2
    stamps = np.empty((nx, ny), dtype=object)
    for x in range(nx):
        for y in range(ny):
            stamps[x, y] = _bright_disk_stamp(
                radius=12, center=(45 + x, 52 + y), peak=50000
            )
    chip = _build_chip_with_stamps(tmp_path / "btn", stamps)
    chip.findButtons(n_jobs=1)

    legacy = chip.summarize()
    roi = RoiSet.from_chip(chip, features="button")
    got = quantify(stamps_from_chip(chip), roi, features="button")

    compare_summaries(legacy, got, BUTTON_COLS, atol=1.0, rtol=0.0)


def test_quantify_matches_summarize_synthetic_chambers(tmp_path):
    nx, ny = 2, 2
    stamps = np.empty((nx, ny), dtype=object)
    for x in range(nx):
        for y in range(ny):
            stamps[x, y] = _bright_disk_stamp(
                radius=Stamp.chamberrad, center=(48 + x, 50 + y)
            )
    chip = _build_chip_with_stamps(tmp_path / "ch2", stamps)
    chip.findChambers(n_jobs=1)

    legacy = chip.summarize()
    roi = RoiSet.from_chip(chip, features="chamber")
    got = quantify(stamps_from_chip(chip), roi, features="chamber")

    compare_summaries(legacy, got, CHAMBER_COLS, atol=1.0, rtol=0.0)


def test_render_summary_all_features(tmp_path):
    nx, ny = 2, 2
    stamps_arr = np.empty((nx, ny), dtype=object)
    for x in range(nx):
        for y in range(ny):
            stamps_arr[x, y] = _bright_disk_stamp(
                radius=Stamp.chamberrad, center=(50, 50)
            )
    chip = _build_chip_with_stamps(tmp_path / "all", stamps_arr)
    chip.findChambers(coerce_center=True, n_jobs=1)
    # Define buttons manually at center for render test
    for s in chip.stamps.flatten():
        s.defineButton((50, 50), 12, (12, 24))

    roi = RoiSet.from_chip(chip, features="all")
    stamps = stamps_from_chip(chip)
    metrics = quantify(stamps, roi, features="all")
    img = render_summary(stamps, roi, features="all", metrics=metrics)
    assert img.ndim == 2
    assert img.shape[0] > ChipImage.stampWidth
    assert img.shape[1] > ChipImage.stampWidth


@pytest.mark.slow
def test_quantify_vs_mapto_real_image(tmp_path):
    """Legacy mapto+summarize vs RoiSet quantify on real chip image."""
    if not DEFAULT_IMAGE.exists():
        pytest.skip(f"Regression image not found at {DEFAULT_IMAGE}")

    chip_ref = make_chip(DEFAULT_IMAGE, DEFAULT_CORNERS)
    run_feature_find(chip_ref, feature="button", n_jobs=2)
    roi = RoiSet.from_chip(chip_ref, features="button")

    # Legacy map path onto a fresh chip of the same image
    chip_legacy = make_chip(DEFAULT_IMAGE, DEFAULT_CORNERS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        chip_ref.mapto(chip_legacy, features="button")
    df_legacy = chip_legacy.summarize()

    df_new = quantify(DEFAULT_IMAGE, roi, features="button")
    compare_summaries(df_legacy, df_new, BUTTON_VALUE_COLS, atol=1.0, rtol=0.0)

    outpath = tmp_path / "roi_quantify_unity.png"
    plot_unity_scatter(
        df_legacy,
        df_new,
        col="summed_button_BGsub",
        outpath=outpath,
        title="RoiSet quantify vs mapto (summed_button_BGsub)",
        xlabel="mapto + summarize",
        ylabel="RoiSet quantify",
    )
    assert outpath.exists() and outpath.stat().st_size > 0


def test_extract_stamps_matches_chip(tmp_path):
    nx, ny = 2, 2
    stamps = np.empty((nx, ny), dtype=object)
    for x in range(nx):
        for y in range(ny):
            stamps[x, y] = _bright_disk_stamp(12, center=(50, 50))
    chip = _build_chip_with_stamps(tmp_path / "ex", stamps)
    chip.findButtons(n_jobs=1)
    roi = RoiSet.from_chip(chip, features="button")
    extracted = extract_stamps(chip.data_ref, roi)
    stacked = stamps_from_chip(chip)
    np.testing.assert_array_equal(extracted, stacked)
