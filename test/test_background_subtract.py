"""Unit tests for background subtraction (serial and parallel)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from skimage import io

from mercury.stitching import BackgroundSubtractor, backgroud_subtract, subtract_single_image


SETTINGS = ["setup", "dname", "channel", "exposure"]


def _write_tif(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    io.imsave(path, arr, check_contrast=False)


def _make_stitched_fixture(tmp_path: Path, include_bad: bool = False):
    """Build a minimal stitched_images tree + CSV for BackgroundSubtractor."""
    root = tmp_path / "exp"
    stitched = root / "stitched_images"
    stitched.mkdir(parents=True)

    bg = (np.full((32, 32), 100, dtype=np.uint16))
    t0 = (np.full((32, 32), 500, dtype=np.uint16))
    t1 = (np.full((32, 32), 250, dtype=np.uint16))

    bg_name = "background.tif"
    t0_name = "target_0.tif"
    t1_name = "target_1.tif"

    _write_tif(stitched / bg_name, bg)
    _write_tif(stitched / t0_name, t0)
    _write_tif(stitched / t1_name, t1)

    rows = [
        {"image_path": bg_name, "setup": "s1", "dname": "d1", "channel": "cyan", "exposure": 100},
        {"image_path": t0_name, "setup": "s1", "dname": "d1", "channel": "cyan", "exposure": 100},
        {"image_path": t1_name, "setup": "s1", "dname": "d1", "channel": "cyan", "exposure": 100},
    ]

    if include_bad:
        bad_name = "bad_target.tif"
        (stitched / bad_name).write_bytes(b"not a tiff")
        rows.append(
            {"image_path": bad_name, "setup": "s1", "dname": "d1", "channel": "cyan", "exposure": 100}
        )

    pd.DataFrame(rows).to_csv(stitched / "stitched_images.csv", index=False)
    return root, bg, t0, t1


@pytest.mark.parametrize("n_workers", [1, 2])
def test_subtract_serial_and_parallel(tmp_path, n_workers):
    root, bg, t0, t1 = _make_stitched_fixture(tmp_path)
    bg_path = root / "stitched_images" / "background.tif"

    subtractor = BackgroundSubtractor(root)
    subtractor.subtract(
        background_images=[bg_path],
        settings_to_match=SETTINGS,
        n_workers=n_workers,
    )

    out_dir = root / "bgsub_images"
    csv_path = out_dir / "bgsub_images.csv"
    assert csv_path.exists()

    out0 = io.imread(out_dir / "target_0.tif")
    out1 = io.imread(out_dir / "target_1.tif")
    np.testing.assert_array_equal(out0, backgroud_subtract(bg, t0))
    np.testing.assert_array_equal(out1, backgroud_subtract(bg, t1))
    assert out0.dtype == np.uint16
    expected0 = np.clip(t0.astype(float) - bg.astype(float), 0, 65535).astype(np.uint16)
    np.testing.assert_array_equal(out0, expected0)

    df = pd.read_csv(csv_path)
    assert set(df["image_path"]) == {"target_0.tif", "target_1.tif"}
    assert df["success"].all()
    assert (df["background_image"] == "background.tif").all()


def test_subtract_marks_failure(tmp_path):
    root, _, _, _ = _make_stitched_fixture(tmp_path, include_bad=True)
    bg_path = root / "stitched_images" / "background.tif"

    subtractor = BackgroundSubtractor(root)
    subtractor.subtract(
        background_images=[bg_path],
        settings_to_match=SETTINGS,
        n_workers=1,
        verbose=False,
    )

    df = pd.read_csv(root / "bgsub_images" / "bgsub_images.csv")
    bad_row = df[df["image_path"] == "bad_target.tif"].iloc[0]
    assert bad_row["success"] == False
    assert pd.isna(bad_row["background_image"])
    assert not (root / "bgsub_images" / "bad_target.tif").exists()

    ok = df[df["image_path"] != "bad_target.tif"]
    assert ok["success"].all()


def test_subtract_single_image_copy_when_no_background(tmp_path):
    src = tmp_path / "src.tif"
    dst = tmp_path / "dst.tif"
    arr = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)
    _write_tif(src, arr)

    ok, err = subtract_single_image(None, str(src), str(dst))
    assert ok and err is None
    np.testing.assert_array_equal(io.imread(dst), arr)
