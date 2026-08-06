"""Unit tests for cut-stitch paste geometry and streaming behavior."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from skimage import io

from mercury.stitching import stitch_single_raster
from mercury.stitching import rastering


def _concat_cut_reference(tiles, dims, acqui_ori, overlap, size):
    """Historical trim → reshape → flip → concatenate geometry."""
    margin = int(size * overlap / 2)
    retained = size - 2 * margin
    if margin == 0:
        border = slice(0, None)
    else:
        border = slice(margin, -margin)

    trimmed = [tile[border, border] for tile in tiles]
    tile_array = np.asarray(trimmed)
    arranged = np.reshape(tile_array, (dims[0], dims[1], retained, retained))
    if acqui_ori[0]:
        arranged = np.flip(arranged, 0)
    if acqui_ori[1]:
        arranged = np.flip(arranged, 1)
    rows = np.concatenate(arranged, axis=2)
    return np.concatenate(rows, axis=0)


def _make_synthetic_tiles(cols, rows, size, seed=0):
    rng = np.random.default_rng(seed)
    tiles = []
    for c in range(cols):
        for r in range(rows):
            # Unique base level per tile + mild gradient for seam checks
            base = ((c * rows + r + 1) * 1000) % 50000
            yy, xx = np.mgrid[0:size, 0:size]
            tile = (base + (yy // 8) + (xx // 16)).astype(np.uint16)
            # sprinkle noise so rotations aren't perfectly uniform
            tile = tile + rng.integers(0, 5, size=tile.shape, dtype=np.uint16)
            tiles.append(tile)
    return tiles


@pytest.mark.parametrize(
    "acqui_ori",
    [(False, False), (True, False), (False, True), (True, True)],
)
@pytest.mark.parametrize("rotation", [0.0, 1.5])
def test_assemble_cut_matches_concat_reference(acqui_ori, rotation, tmp_path):
    cols, rows, size, overlap = 3, 2, 64, 0.1
    tiles = _make_synthetic_tiles(cols, rows, size)

    refs = []
    for i, tile in enumerate(tiles):
        path = tmp_path / f"tile_{i:02d}.tif"
        io.imsave(path, tile, check_contrast=False)
        refs.append(path)

    # Reference uses the same rotate_image path as production loaders
    rotated = rastering.load_tiles(refs, rotation=rotation)
    expected = _concat_cut_reference(rotated, (cols, rows), acqui_ori, overlap, size)

    got = rastering.assemble_cut(
        rastering.iter_tiles(refs, rotation=rotation),
        dims=(cols, rows),
        acqui_ori=acqui_ori,
        overlap=overlap,
        size=size,
    )
    np.testing.assert_array_equal(got, expected)


def test_rotate_image_preserves_shape_and_dtype():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 40000, size=(64, 64), dtype=np.uint16)
    out = rastering.rotate_image(img, 1.5)
    assert out.shape == img.shape
    assert out.dtype == np.uint16
    # Identity angle should not allocate a distinct buffer copy of values
    same = rastering.rotate_image(img, 0.0)
    np.testing.assert_array_equal(same, img)


def test_cut_stitch_streaming_clears_images(tmp_path):
    cols, rows, size, overlap = 2, 2, 32, 0.0
    tiles = _make_synthetic_tiles(cols, rows, size)
    refs = []
    for i, tile in enumerate(tiles):
        path = tmp_path / f"tile_{i:02d}.tif"
        io.imsave(path, tile, check_contrast=False)
        refs.append(path)

    params = rastering.RasterParams(
        overlap=overlap,
        size=size,
        acqui_ori=(False, False),
        rotation=0.0,
        dims=(cols, rows),
        auto_ff=False,
    )
    raster = rastering.FlatRaster(image_refs=refs, params=params)
    out = raster.stitch(method="cut")
    assert out.shape == (rows * size, cols * size)
    assert raster._images is None


def test_parallel_stitch_single_raster_smoke(tmp_path):
    cols, rows, size, overlap = 2, 2, 32, 0.0

    def _write_raster(subdir: Path):
        tiles = _make_synthetic_tiles(cols, rows, size, seed=hash(subdir.name) % 10_000)
        refs = []
        for i, tile in enumerate(tiles):
            path = subdir / f"tile_{i:02d}.tif"
            io.imsave(path, tile, check_contrast=False)
            refs.append(str(path))
        return refs

    jobs = []
    for name in ("a", "b"):
        d = tmp_path / name
        d.mkdir()
        refs = _write_raster(d)
        params = rastering.RasterParams(
            overlap=overlap,
            size=size,
            acqui_ori=(True, False),
            rotation=0.0,
            dims=(cols, rows),
            auto_ff=False,
        )
        out = tmp_path / f"{name}_stitched.tif"
        jobs.append((refs, params, out))

    with ProcessPoolExecutor(max_workers=2) as ex:
        futs = [
            ex.submit(stitch_single_raster, refs, params, out, "cut")
            for refs, params, out in jobs
        ]
        results = [f.result() for f in futs]

    for ok, path, err in results:
        assert ok, err
        assert Path(path).exists()
        img = io.imread(path)
        assert img.shape == (rows * size, cols * size)
