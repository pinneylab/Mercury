"""Tests for the mercury-run pipeline: config, agent overrides, preflight, CLI plumbing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from skimage import io

from mercury.pipeline import cli, config as cfgmod, stages
from mercury.pipeline.record import Run


def _make_experiment(root: Path, two_backgrounds: bool = False, drop_tile: bool = False) -> None:
    """Tiny experiment: one series (2 rasters) + one background raster, 2x2 tiles of 32 px."""
    rows = []
    rasters = [
        ("raw_images/2026-01-01_00-00-00_series_A/img_0", "d1"),
        ("raw_images/2026-01-01_00-00-00_series_A/img_1", "d1"),
        ("raw_images/2026-01-01_00-00-01_d1_bg_background", "d1"),
    ]
    if two_backgrounds:
        rasters.append(("raw_images/2026-01-01_00-00-02_d1_bg_background", "d1"))
    for parent, d in rasters:
        for c in range(2):
            for r in range(2):
                p = f"{parent}/Pos-{c:03d}_{r:03d}.tif"
                rows.append({"image_path": p, "image_path_parent": parent, "setup": "S", "dname": d,
                             "raster_width": 2, "raster_height": 2, "raster_overlap": 0.1,
                             "raster_row_index": r, "raster_col_index": c, "lightsource": "uv",
                             "channel": 5, "exposure": 500, "camera_mode": "dr", "binning": "2x2",
                             "nosepiece": 4, "temp": np.nan, "hum": np.nan})
                if drop_tile and parent.endswith("img_1") and c == 1 and r == 1:
                    continue
                (root / p).parent.mkdir(parents=True, exist_ok=True)
                io.imsave(root / p, np.zeros((32, 32), np.uint16), check_contrast=False)
    pd.DataFrame(rows).to_csv(root / "imaging.csv", index=False)
    (root / "pins.csv").write_text("x\n")
    index = [{"identifier": "2026-01-01_00-00-00_series_A", "item_type": "series", "items": [
        {"identifier": "raw_images/2026-01-01_00-00-00_series_A/img_0", "item_type": "image",
         "metadata": {"concentration": 1.0}},
        {"identifier": "raw_images/2026-01-01_00-00-00_series_A/img_1", "item_type": "image",
         "metadata": {"concentration": 5.0}},
    ]}]
    (root / "series_index.json").write_text(json.dumps(index))


def _write_config(path: Path, root: Path, **extra) -> Path:
    cfg = {
        "version": 1, "root": str(root),
        "stitching": {"rotation": 0.9, "test_rasters": ["img_1"]},
        "devices": {"d1": {"setup": "s2", "dims": [4, 5], "pinlist": "pins.csv",
                           "corners": [[1, 2], [30, 2], [1, 40], [30, 40]]}},
        "series": [{"name": "nadh", "identifiers": ["2026-01-01_00-00-00_series_A"],
                    "features": "chamber", "reference": {"pick": "max", "by": "concentration"},
                    "output": "nadh.csv"}],
    }
    for k, v in extra.items():
        cfg[k] = v
    path.write_text(yaml.safe_dump(cfg))
    return path


# --------------------------------------------------------------------------- config

def test_config_defaults_and_validation(tmp_path):
    p = _write_config(tmp_path / "processing.yaml", tmp_path)
    cfg = cfgmod.load_config(p)
    assert cfg["qc"]["seam_max_residual_px"] == 1.0
    assert cfg["series"][0]["coerce_chamber_center"] is False

    bad = yaml.safe_load(p.read_text())
    bad["series"][0]["features"] = "everything"
    bad["series"][0]["reference"] = {"pick": "max"}  # max needs `by`
    bad["devices"]["d1"]["corners"] = [[1, 2]]
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(cfgmod.ConfigError) as e:
        cfgmod.load_config(p)
    msg = str(e.value)
    assert "features" in msg and "needs reference.by" in msg and "corners" in msg


def test_agent_overrides_are_allowlisted_and_reasoned(tmp_path):
    p = _write_config(tmp_path / "processing.yaml", tmp_path)
    entry = cfgmod.add_override(p, "series.nadh.reference.pick", "brightest", "max picks a saturated image")
    assert entry["reason"]
    assert cfgmod.load_config(p)["series"][0]["reference"]["pick"] == "brightest"
    # user file untouched
    assert yaml.safe_load(p.read_text())["series"][0]["reference"]["pick"] == "max"

    for key in ("qc.seam_max_residual_px", "series.nadh.features", "series.nadh.coerce_chamber_center",
                "series.nadh.identifiers", "background.images"):
        with pytest.raises(cfgmod.ConfigError):
            cfgmod.add_override(p, key, 99, "trying")
    with pytest.raises(cfgmod.ConfigError):
        cfgmod.add_override(p, "devices.d1.corners", [[1, 2], [3, 4], [5, 6], [7, 8]], "  ")
    # overrides must still produce a valid config
    with pytest.raises(cfgmod.ConfigError):
        cfgmod.add_override(p, "devices.d1.corners", [[1, 2]], "bad shape")


# --------------------------------------------------------------------------- helpers

def test_pick_image_rules(tmp_path):
    paths = []
    for i, val in enumerate([10, 50, 30]):
        f = tmp_path / f"img_{i}.tif"
        io.imsave(f, np.full((64, 64), val, np.uint16), check_contrast=False)
        paths.append(f)
    df = pd.DataFrame({"image_path": paths, "concentration": [1.0, 2.0, 8.0], "time_index": [2, 0, 1]})
    assert stages.pick_image(df, {"pick": "max", "by": "concentration"}, tmp_path)[0] == paths[2]
    assert stages.pick_image(df, {"pick": "first", "by": "time_index"}, tmp_path)[0] == paths[1]
    assert stages.pick_image(df, {"pick": "brightest"}, tmp_path)[0] == paths[1]
    assert stages.pick_image(df, {"pick": "image", "image": "img_0"}, tmp_path)[0] == paths[0]
    with pytest.raises(stages.StageError):
        stages.pick_image(df, {"pick": "max", "by": "nope"}, tmp_path)


def test_is_background():
    assert stages.is_background("x_nadh_background.tif", "*background*")
    assert stages.is_background("x_nadh_background", ["a", "x_nadh_background"])
    assert not stages.is_background("x_signal", "*background*")


# --------------------------------------------------------------------------- preflight

def test_preflight_passes_on_consistent_experiment(tmp_path):
    root = tmp_path / "exp"
    root.mkdir()
    _make_experiment(root)
    run = Run(_write_config(tmp_path / "processing.yaml", root))
    res = stages.preflight(run)
    assert res["status"] == "pass", res
    run.save()
    resolved = yaml.safe_load(run.resolved_path.read_text())
    assert resolved["stages"]["preflight"]["status"] == "pass"
    assert "mercury" in resolved["meta"]


def test_preflight_flags_ambiguous_background_and_missing_tiles(tmp_path):
    root = tmp_path / "exp"
    root.mkdir()
    _make_experiment(root, two_backgrounds=True, drop_tile=True)
    run = Run(_write_config(tmp_path / "processing.yaml", root))
    res = stages.preflight(run)
    assert res["status"] == "needs_human"
    assert any("ambiguous background" in e for e in res["summary"]["errors"])
    assert any("rasters_incomplete" in str(w) for w in res["summary"]["warnings"])
    assert run.resolved["escalations"][-1]["stage"] == "preflight"


def test_preflight_rejects_unknown_identifier_and_test_raster(tmp_path):
    root = tmp_path / "exp"
    root.mkdir()
    _make_experiment(root)
    p = _write_config(tmp_path / "processing.yaml", root,
                      stitching={"rotation": 0.9, "test_rasters": ["does_not_exist"]})
    cfg = yaml.safe_load(p.read_text())
    cfg["series"][0]["identifiers"] = ["nope"]
    p.write_text(yaml.safe_dump(cfg))
    res = stages.preflight(Run(p))
    errs = " ".join(res["summary"]["errors"])
    assert "nope" in errs and "does_not_exist" in errs


# --------------------------------------------------------------------------- CLI

def test_cli_preflight_status_and_refused_override(tmp_path, capsys):
    root = tmp_path / "exp"
    root.mkdir()
    _make_experiment(root)
    p = _write_config(tmp_path / "processing.yaml", root)

    assert cli.main(["preflight", "--config", str(p)]) == 0
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last.startswith("RESULT ") and json.loads(last[7:])["status"] == "pass"

    assert cli.main(["override", "--config", str(p), "--key", "qc.corner_max_shift_px",
                     "--value", "50", "--reason", "make it pass"]) == 2
    assert "refused" in capsys.readouterr().out

    assert cli.main(["status", "--config", str(p)]) == 0
    status = json.loads(capsys.readouterr().out.strip().splitlines()[-1][7:])
    assert status["stages"]["preflight"] == "pass" and status["next"] == "calibrate"

    log = [json.loads(l) for l in (tmp_path / "processing.log.jsonl").read_text().splitlines()]
    assert {"stage_start", "stage_end"} <= {e["event"] for e in log}


def test_stage_requires_previous_stage(tmp_path):
    root = tmp_path / "exp"
    root.mkdir()
    _make_experiment(root)
    run = Run(_write_config(tmp_path / "processing.yaml", root))
    with pytest.raises(stages.StageError):
        stages.stitch(run)
