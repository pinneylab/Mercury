"""Tests for seam-residual measurement (stitch jaggedness) and seam-based rotation/overlap detection."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from skimage import io

from mercury.stitching import rastering

ACQUI_ORIS = [(False, False), (True, False), (False, True), (True, True)]


def _make_raster(tmp_path, dims, acqui_ori, size, ov_px, theta, seed=0):
    """Tiles of a textured scene on a stage grid with step size - ov_px, camera rotated by -theta.

    Tile i is generated at canvas position grid_index(i), so the stitcher's correct
    parameters are rotation=theta and a pitch of size - ov_px.
    """
    cols, rows = dims
    step = size - ov_px
    big = int(size * 1.6)
    rng = np.random.default_rng(seed)
    scene = gaussian_filter(rng.random((rows * step + big + 50, cols * step + big + 50)), 1.5)
    scene = ((scene - scene.min()) / np.ptp(scene) * 40000).astype(np.uint16)
    off = (big - size) // 2
    refs = []
    for i in range(cols * rows):
        c, r = rastering.grid_index(i, dims, acqui_ori)
        y0, x0 = 25 + r * step, 25 + c * step
        tile = rastering.rotate_image(scene[y0:y0 + big, x0:x0 + big], -theta)
        tile = tile[off:off + size, off:off + size]
        path = tmp_path / f"tile_{i:02d}.tif"
        io.imsave(path, tile, check_contrast=False)
        refs.append(path)
    return refs


def test_seam_pairs_are_canvas_neighbours():
    dims = (3, 2)
    for acqui_ori in ACQUI_ORIS:
        pairs = rastering.seam_pairs(dims, acqui_ori)
        # (cols-1)*rows horizontal + cols*(rows-1) vertical seams
        assert len(pairs) == 2 * 2 + 3 * 1
        for direction, i_a, i_b, pa, pb in pairs:
            assert rastering.grid_index(i_a, dims, acqui_ori) == pa
            assert rastering.grid_index(i_b, dims, acqui_ori) == pb
            if direction == "H":
                assert pb == (pa[0] + 1, pa[1])
            else:
                assert pb == (pa[0], pa[1] + 1)


@pytest.mark.parametrize("acqui_ori", ACQUI_ORIS)
@pytest.mark.parametrize("theta", [-0.9, 0.9])
def test_estimate_recovers_rotation_and_overlap(tmp_path, acqui_ori, theta):
    size, ov_px, dims = 256, 30, (3, 3)
    refs = _make_raster(tmp_path, dims, acqui_ori, size, ov_px, theta)
    seams = rastering.measure_seams(refs, dims, acqui_ori, overlap0=0.1, n_seams=12)
    assert seams["inlier"].all()

    rotation, overlap = rastering.estimate_from_seams(seams)
    assert rotation == pytest.approx(theta, abs=0.05)
    assert overlap == pytest.approx(ov_px / size, abs=2 / size)

    # Seamless at the true parameters...
    good = rastering.score_seams(seams, theta, ov_px / size)
    assert good["along_median_abs"] < 0.5
    assert good["across_median_abs"] < 0.5
    # ...and a 0.5 deg rotation error shows up as ~step*sin(0.5 deg) px along the seams.
    bad = rastering.score_seams(seams, theta + 0.5, ov_px / size)
    expected = (size - ov_px) * np.sin(np.radians(0.5))
    assert bad["along_median_abs"] == pytest.approx(expected, abs=0.3)
    # A wrong overlap shows up across the seams (here: nominal 10% vs true ~11.7%).
    off = rastering.score_seams(seams, theta, 0.1)
    assert off["across_median_abs"] == pytest.approx(ov_px - 2 * int(size * 0.1 / 2), abs=0.5)


@pytest.mark.parametrize("acqui_ori", ACQUI_ORIS)
def test_detect_rotation_respects_acqui_ori(tmp_path, acqui_ori):
    """Regression: detect_rotation used to ignore acqui_ori when pairing tiles."""
    size, ov_px, dims, theta = 256, 26, (3, 3), 0.9
    refs = _make_raster(tmp_path, dims, acqui_ori, size, ov_px, theta)
    params = rastering.RasterParams(
        overlap=0.1, size=size, acqui_ori=acqui_ori, rotation=0.0, dims=dims, auto_ff=False
    )
    raster = rastering.FlatRaster(image_refs=refs, params=params)
    assert raster.detect_rotation(search_range=(-4.0, 4.0)) == pytest.approx(theta, abs=0.05)
    assert raster.detect_overlap() == pytest.approx(ov_px / size, abs=2 / size)


def test_featureless_raster_is_unmeasurable(tmp_path):
    rng = np.random.default_rng(0)
    refs = []
    for i in range(9):
        path = tmp_path / f"tile_{i:02d}.tif"
        io.imsave(path, rng.integers(100, 104, size=(128, 128), dtype=np.uint16), check_contrast=False)
        refs.append(path)
    seams = rastering.measure_seams(refs, (3, 3), (True, False), overlap0=0.1)
    score = rastering.score_seams(seams, 0.0, 0.1)
    # Pure noise: strips register somewhere, but the aligned strips don't correlate.
    assert score["n_seams"] == 12
    assert score["n_inliers"] == 0
    assert np.isnan(score["along_median_abs"])
    with pytest.raises(ValueError):
        rastering.estimate_from_seams(seams)
