"""Tests for corner quality scoring (processing.corners)."""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from mercury.processing import corners as C
from mercury.processing.chip import ChipImage

DIMS = (10, 12)  # (nx, ny) unit cells
TRUE = np.array([(150.0, 140.0), (1780.0, 160.0), (130.0, 1590.0), (1770.0, 1600.0)])  # UL UR BL BR (x, y)
PITCH_X = (TRUE[1, 0] - TRUE[0, 0]) / (DIMS[0] - 1)
PITCH_Y = (TRUE[2, 1] - TRUE[0, 1]) / (DIMS[1] - 1)


@pytest.fixture(scope="module")
def chip_image():
    """Bright chamber-like disks (radius 35 px) on a noisy background at the TRUE lattice."""
    rng = np.random.default_rng(0)
    img = np.zeros((1800, 1950))
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    for r in C.lattice_points(TRUE, DIMS).itertuples():
        img[(xx - r.x) ** 2 + (yy - r.y) ** 2 <= 35 ** 2] = 1.0
    img = gaussian_filter(img, 1.5) * 3000 + 500 + rng.normal(0, 60, img.shape)
    return np.clip(img, 0, 65535).astype(np.uint16)


def test_lattice_points_match_quadrilateral_interp():
    dims = namedtuple("D", ["x", "y"])(*DIMS)
    prod = ChipImage.quadrilateralInterp([tuple(map(int, c)) for c in TRUE], dims)  # (nx, ny, 2), int
    mine = C.lattice_points(TRUE, DIMS)[["x", "y"]].to_numpy().reshape(DIMS[0], DIMS[1], 2)
    # production truncates to int twice; agreement within 2 px
    assert np.abs(mine - prod).max() <= 2.0


def test_bilinear_fit_recovers_known_offsets():
    pts = C.lattice_points(TRUE, DIMS)
    delta = np.array([(3.0, -2.0), (-4.0, 1.0), (2.5, 4.0), (-1.0, -3.5)])
    field = C.bilinear_weights(pts["u"], pts["v"]) @ delta
    pts["dx"], pts["dy"], pts["found"] = field[:, 0], field[:, 1], True
    pts.loc[::7, ["dx", "dy"]] += 40.0  # outliers (e.g. a wrong circle)
    fit, inlier, _ = C.fit_corner_correction(pts)
    np.testing.assert_allclose(fit, delta, atol=1e-6)
    assert not inlier[::7].any()


def test_correct_corners_pass(chip_image):
    s, offsets = C.check_corners(chip_image, TRUE, DIMS, n_jobs=1)
    assert s["status"] == "pass", s
    assert s["n_found"] == s["n_stamps"]
    assert s["max_corner_shift_px"] < 1.5
    assert s["ring"]["pass"]


def test_perturbed_corners_are_corrected(chip_image):
    guess = TRUE + np.array([(6.0, -5.0), (-9.0, 4.0), (5.0, 8.0), (-7.0, -6.0)])
    s, _ = C.check_corners(chip_image, guess, DIMS, n_jobs=1)
    assert s["status"] == "refit"
    final, _, history = C.refine_corners(chip_image, guess, DIMS, n_jobs=1)
    assert final["status"] == "pass"
    assert np.abs(np.array(final["proposed_corners"]) - TRUE).max() <= 1.5
    assert len(history) <= 3


@pytest.mark.parametrize(
    "shift", [(0, PITCH_Y), (0, -PITCH_Y), (PITCH_X, 0), (-PITCH_X, 0)],
    ids=["+row", "-row", "+col", "-col"],
)
def test_one_pitch_shift_fails_ring(chip_image, shift):
    """Offsets look perfect when the grid is off by a whole pitch; the ring check must catch it."""
    s, _ = C.check_corners(chip_image, TRUE + np.array(shift), DIMS, n_jobs=1)
    assert s["status"] == "fail", s["ring"]
    assert not s["ring"]["pass"]


def test_blank_image_is_unmeasurable():
    img = np.random.default_rng(1).normal(500, 5, (1800, 1950)).clip(0).astype(np.uint16)
    s, _ = C.check_corners(img, TRUE, DIMS, n_jobs=1)
    # Hough loosens its thresholds until it "finds" circles in noise, so offsets alone
    # can't reject this; the ring check (no unit-cell structure anywhere) must.
    assert s["status"] in ("unmeasurable", "fail")
    assert s["status"] == "unmeasurable" or not s["ring"]["pass"]
