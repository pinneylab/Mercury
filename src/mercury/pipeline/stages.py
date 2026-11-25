"""Processing stages.

Each stage takes a record.Run and returns a result dict
    {"stage": str, "status": "pass" | "warn" | "needs_human" | "fail", "summary": {...}}
and records what it did in the resolved file and the log.

Decision stages (calibrate, corners) choose parameters and may refit them within the
QC gates; when they can't, they escalate (status needs_human) rather than guess.
Execution stages (stitch, bgsub, reference, map) only consume resolved values, so a
resolved file replays without any decisions being re-made.

Every stage skips itself when its inputs (hashed) are unchanged since its last
successful run, unless redo=True.
"""

from __future__ import annotations

import contextlib
import fnmatch
import os
import pickle
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from mercury.pipeline import config as cfgmod
from mercury.pipeline.record import Run, file_sha256, now, plain

OK = ("pass", "warn")
ORDER = ["preflight", "calibrate", "stitch", "bgsub", "corners", "reference", "map"]


class StageError(RuntimeError):
    """A stage cannot run (missing prerequisite, unsafe overwrite, dirty code...)."""


def result(stage: str, status: str, **summary) -> dict:
    return {"stage": stage, "status": status, "summary": plain(summary)}


# --------------------------------------------------------------------------- helpers

@contextlib.contextmanager
def captured(run: Run, stage: str):
    """Send library stdout/stderr (incl. worker processes) to <root>/qc/logs/<stage>_<time>.log."""
    logdir = run.root / "qc" / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    path = logdir / f"{stage}_{now().replace(':', '-')}.log"
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    with open(path, "a") as f:
        os.dup2(f.fileno(), 1)
        os.dup2(f.fileno(), 2)
        try:
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                yield path
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])


def require(run: Run, *stages: str) -> None:
    for s in stages:
        st = run.stage(s).get("status")
        if st not in OK:
            raise StageError(f"Stage '{s}' has not passed (status: {st or 'not run'}). Run it first.")


def load_imaging(run: Run) -> pd.DataFrame:
    path = run.root / "imaging.csv"
    if not path.exists():
        raise StageError(f"{path} not found")
    return pd.read_csv(path)


def raster_table(imaging: pd.DataFrame) -> pd.DataFrame:
    """One row per raster (image_path_parent) with its acquisition settings."""
    t = imaging.drop_duplicates("image_path_parent").copy()
    t["name"] = t["image_path_parent"].map(lambda p: Path(p).name)
    t["series_dir"] = t["image_path_parent"].map(
        lambda p: Path(p).parts[1] if len(Path(p).parts) > 2 else None
    )
    return t.reset_index(drop=True)


def resolve_raster(rasters: pd.DataFrame, spec: str) -> str:
    """Raster dir (relative to root) for a spec: relative path, dir name, or unique suffix."""
    spec = str(spec).rstrip("/")
    for col in ("image_path_parent", "name"):
        hit = rasters[rasters[col] == spec]
        if len(hit) == 1:
            return hit["image_path_parent"].iloc[0]
    hit = rasters[rasters["image_path_parent"].str.endswith(spec)]
    if len(hit) == 1:
        return hit["image_path_parent"].iloc[0]
    raise StageError(f"Test raster {spec!r} matches {len(hit)} rasters in imaging.csv")


def series_rasters(rasters: pd.DataFrame, identifier: str) -> pd.DataFrame:
    """Rasters belonging to a series identifier (raw_images/<id>/...) or an image identifier (dir name)."""
    return rasters[(rasters["series_dir"] == identifier) | (rasters["name"] == identifier)]


def is_background(name: str, spec) -> bool:
    """background.images is a glob or a list of globs / names, matched on the raster or file name."""
    pats = [spec] if isinstance(spec, str) else list(spec)
    stem = name[:-4] if name.endswith(".tif") else name
    return any(fnmatch.fnmatch(stem, p) or fnmatch.fnmatch(stem + ".tif", p) for p in pats)


def nose_str(values: pd.Series) -> pd.Series:
    """Nosepiece as a string key ('4'), robust to float/NaN columns."""
    return pd.to_numeric(values, errors="coerce").map(lambda v: "" if v != v else str(int(v)))


def ff_mask(rasters: pd.Series, patterns: List[str]) -> pd.Series:
    names = rasters.map(lambda p: Path(p).name)
    return names.map(lambda n: any(fnmatch.fnmatch(n, pat) for pat in patterns))


def build_experiment(run: Run):
    from mercury.processing.experiment import Experiment, Device

    ex = Experiment(run.root)
    for d, dev in run.cfg["devices"].items():
        device = Device(setup=dev["setup"], dname=d, dims=tuple(dev["dims"]))
        pin = Path(dev["pinlist"])
        device.set_pinlist(pinlist_path=pin if pin.is_absolute() else run.root / pin)
        ex.add_device(device)
    return ex


def series_images(run: Run, s: dict) -> pd.DataFrame:
    """Background-subtracted images of a series (DataHandler), restricted to its devices."""
    from mercury.processing.experiment import DataHandler

    with captured(run, "datahandler"):
        dh = DataHandler(run.root)
        frames = []
        for ident in s["identifiers"]:
            try:
                df = dh.get_images([ident])
            except ValueError:  # pd.concat([]) when nothing matched
                df = pd.DataFrame()
            if df.empty:
                raise StageError(f"series {s['name']}: identifier {ident!r} matched no bgsub images")
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    return df[df["dname"].isin(cfgmod.series_devices(run.cfg, s))].reset_index(drop=True)


def pick_image(df: pd.DataFrame, ref: Optional[dict], root: Path) -> Tuple[Path, str]:
    """Choose the reference / corner-check image for one device of a series."""
    ref = ref or {"pick": "brightest"}
    pick, by = ref.get("pick", "brightest"), ref.get("by")
    paths = df["image_path"].map(Path)
    if pick == "image":
        want = str(ref["image"])
        for p in paths:
            if str(p) == want or p.stem == Path(want).stem or str(p).endswith(want):
                return p, f"image={want}"
        raise StageError(f"reference.image {want!r} is not among this series' images")
    if pick in ("first", "last"):
        order = df.sort_values(by or "image_path", kind="stable").index
        i = order[0] if pick == "first" else order[-1]
        return paths[i], f"{pick} by {by or 'image_path'}"
    if pick in ("max", "min"):
        if by not in df.columns:
            raise StageError(f"reference.by={by!r} is not a series metadata column ({list(df.columns)})")
        vals = pd.to_numeric(df[by], errors="coerce")
        i = vals.idxmax() if pick == "max" else vals.idxmin()
        return paths[i], f"{pick} {by} = {vals[i]}"
    # brightest: median of a strided subsample
    from skimage import io
    best, best_v = None, -np.inf
    for i, p in paths.items():
        v = float(np.median(io.imread(p)[::8, ::8]))
        if v > best_v:
            best, best_v = i, v
    return paths[best], f"brightest (median {best_v:.1f})"


def guard_overwrite(run: Run, stage: str, target: Path, overwrite: bool) -> None:
    """Refuse to replace outputs this runner didn't make (e.g. a hand-run stitch)."""
    if target.exists() and not run.stage(stage) and not overwrite:
        raise StageError(
            f"{target} exists but was not produced by this runner. "
            f"Re-run with --overwrite to replace it (ask the user first)."
        )


def code_id(run: Run) -> list:
    """Mercury commit + uncommitted-diff fingerprint; part of every stage's input hash."""
    return [run.provenance.get("commit"), run.provenance.get("diff_sha256")]


def unchanged(run: Run, stage: str, key: str, redo: bool) -> bool:
    rec = run.stage(stage)
    return (not redo) and rec.get("status") in OK and rec.get("inputs_sha256") == key


# --------------------------------------------------------------------------- preflight

def preflight(run: Run, **_) -> dict:
    cfg, root = run.cfg, run.root
    errors, warnings = [], []
    if not root.exists():
        errors.append(f"root {root} does not exist")
        return _finish_preflight(run, errors, warnings, {})

    imaging = load_imaging(run)
    rasters = raster_table(imaging)
    info = {"n_rasters": len(rasters), "n_tiles": len(imaging)}

    # tiles present / complete
    incomplete = []
    for parent, g in imaging.groupby("image_path_parent"):
        n_missing = sum(not (root / p).exists() for p in g["image_path"])
        expected = int(g["raster_width"].iloc[0]) * int(g["raster_height"].iloc[0])
        if n_missing or len(g) != expected:
            incomplete.append(f"{parent}: {n_missing} missing, {len(g)}/{expected} tiles listed")
    if incomplete:
        warnings.append({"rasters_incomplete (stitch will skip these)": incomplete})

    listed = set(imaging["image_path"])
    on_disk = [str(p.relative_to(root)) for p in (root / "raw_images").rglob("*.tif")]
    unlisted = sorted(p for p in on_disk if p not in listed)
    if unlisted:
        warnings.append({f"tiles_not_in_imaging_csv ({len(unlisted)})": unlisted[:10]})

    # overlap consistency per objective
    ov = rasters.groupby("nosepiece")["raster_overlap"].unique()
    for nose, vals in ov.items():
        if len(vals) > 1:
            warnings.append(f"nosepiece {nose}: raster_overlap differs across rasters {sorted(vals)}")

    # devices
    dnames = set(imaging["dname"].dropna())
    for d, dev in cfg["devices"].items():
        pin = Path(dev["pinlist"])
        pin = pin if pin.is_absolute() else root / pin
        if not pin.exists():
            errors.append(f"devices.{d}.pinlist {pin} not found")
        if d not in dnames:
            errors.append(f"device {d} not in imaging.csv dname values {sorted(dnames)}")

    # series identifiers
    used = set()
    index_ids = set()
    if (root / "series_index.json").exists():
        import json
        index_ids = {s["identifier"] for s in json.loads((root / "series_index.json").read_text())}
    for s in cfg["series"]:
        for ident in s["identifiers"]:
            hit = series_rasters(rasters, ident)
            if hit.empty:
                errors.append(f"series {s['name']}: identifier {ident!r} matches no raster")
                continue
            if (hit["series_dir"] == ident).any() and ident not in index_ids:
                errors.append(f"series {s['name']}: {ident!r} is a series but not in series_index.json")
            devs = set(cfgmod.series_devices(cfg, s))
            if not set(hit["dname"]) & devs:
                errors.append(f"series {s['name']}: {ident!r} has no images for devices {sorted(devs)}")
            used.update(hit["image_path_parent"])
        if s["reference"] is None and s["features"] != "button":
            warnings.append(f"series {s['name']}: reference is null, so features are found per image (slow)")
    if not (root / "series_index.json").exists():
        errors.append("series_index.json not found (DataHandler needs it)")

    # test rasters
    st = cfg["stitching"]
    if not st["test_rasters"]:
        errors.append("stitching.test_rasters is empty; calibrate needs at least one raster per objective")
    tested_nose = set()
    for spec in st["test_rasters"]:
        try:
            parent = resolve_raster(rasters, spec)
            tested_nose.add(nose_str(rasters.loc[rasters["image_path_parent"] == parent, "nosepiece"]).iloc[0])
        except StageError as e:
            errors.append(str(e))
    used_nose = set(nose_str(rasters.loc[rasters["image_path_parent"].isin(used), "nosepiece"]))
    if used_nose - tested_nose:
        warnings.append(f"no test raster for objective(s) {sorted(used_nose - tested_nose)} used by series")

    # background matching (on raster-level settings, as BackgroundSubtractor will see them)
    t = rasters.copy()
    t["apply_ff_correction"] = ff_mask(t["image_path_parent"], st["ff_correction"])
    keys = [k for k in cfg["background"]["settings_to_match"] if k in t.columns]
    missing_keys = [k for k in cfg["background"]["settings_to_match"] if k not in t.columns]
    if missing_keys:
        warnings.append(f"background.settings_to_match columns absent from imaging.csv: {missing_keys}")
    spec = cfg["background"]["images"]
    t["is_bg"] = t["name"].map(lambda n: is_background(n, spec))
    no_bg = []
    for _, g in t.groupby(keys, dropna=False):
        if not g["image_path_parent"].isin(used).any():
            continue
        bgs = sorted(g.loc[g["is_bg"], "name"])
        if not bgs:
            no_bg.append(", ".join(sorted(g.loc[g["image_path_parent"].isin(used), "name"]))[:300])
        elif len(bgs) > 1:
            # BackgroundSubtractor would silently take the first one.
            errors.append(f"ambiguous background: {bgs} match the same settings; "
                          "list the one to use in background.images")
    if no_bg:
        warnings.append({"series rasters with no matching background (subtracted with zeros)": no_bg})
    info["background_rasters"] = int(t["is_bg"].sum())

    # existing outputs from hand runs
    for d, stage in (("stitched_images/stitched_images.csv", "stitch"), ("bgsub_images/bgsub_images.csv", "bgsub")):
        if (root / d).exists() and not run.stage(stage):
            warnings.append(f"{d} exists from a run outside this tool; {stage} will need --overwrite")

    info["series_rasters"] = len(used)
    return _finish_preflight(run, errors, warnings, info)


def _finish_preflight(run, errors, warnings, info) -> dict:
    status = "needs_human" if errors else ("warn" if warnings else "pass")
    run.resolved["preflight"] = plain({"errors": errors, "warnings": warnings, **info})
    run.set_stage("preflight", status)
    if errors:
        run.escalate("preflight", "config", "; ".join(errors)[:500],
                     "Fix processing.yaml or the data folder, then re-run preflight.")
    else:
        run.close_escalations("preflight")
    return result("preflight", status, errors=errors, warnings=warnings, **info)


# --------------------------------------------------------------------------- calibrate

def calibrate(run: Run, redo: bool = False, **_) -> dict:
    from mercury.stitching import ImageStitcher, rastering

    require(run, "preflight")
    cfg, st, qc = run.cfg, run.cfg["stitching"], run.cfg["qc"]
    key = cfgmod.sha256_of({"stitching": st, "acqui_origin": cfg["acqui_origin"], "code": code_id(run),
                            "qc": {k: v for k, v in qc.items() if k.startswith("seam")}})
    if unchanged(run, "calibrate", key, redo):
        return result("calibrate", run.stage("calibrate")["status"], cached=True, **run.resolved.get("stitching", {}))

    thr, min_inl = qc["seam_max_residual_px"], qc["seam_min_inliers"]
    ov_user = {str(k): float(v) for k, v in (st["overlap_by_nosepiece"] or {}).items()}
    rasters = raster_table(load_imaging(run))
    rot_user = st["rotation"]

    tests = []
    with captured(run, "calibrate"):
        stitcher = ImageStitcher(run.root)
        for spec in st["test_rasters"]:
            parent = resolve_raster(rasters, spec)
            nose = nose_str(rasters.loc[rasters["image_path_parent"] == parent, "nosepiece"]).iloc[0]
            summary, seams = stitcher.check_stitching_params(
                run.root / parent, rotation=rot_user if rot_user is not None else 0.0,
                overlap=ov_user.get(nose), acqui_origin=tuple(cfg["acqui_origin"]),
                max_residual_px=thr, min_inliers=min_inl, save_seams=True,
            )
            tests.append({"spec": spec, "nosepiece": nose, "summary": summary, "seams": seams})

    measurable = [t for t in tests if t["summary"]["status"] != "unmeasurable"]
    out = {"tests": [{"spec": t["spec"], "nosepiece": t["nosepiece"], **t["summary"]} for t in tests]}
    if not measurable:
        run.resolved["stitching"] = plain(out)
        run.set_stage("calibrate", "needs_human", inputs_sha256=key)
        run.escalate("calibrate", "test_rasters", "No test raster had enough usable seams.",
                     "Choose test rasters with visible features (brightfield, button quant, high NADH) "
                     "and set stitching.test_rasters (agent may override this key).")
        return result("calibrate", "needs_human", **out)

    # rotation: keep the user's unless any test raster's along-seam residual fails
    rotation = {"value": rot_user, "source": "user", "user_value": rot_user}
    if rot_user is None or any(t["summary"]["along_median_abs"] > thr for t in measurable):
        est = float(np.median([t["summary"]["rotation_estimate"] for t in measurable]))
        rotation = {"value": round(est, 4), "source": "estimate" if rot_user is None else "refit",
                    "user_value": rot_user,
                    "reason": "no rotation given" if rot_user is None else
                    f"along-seam median residual > {thr} px at user rotation"}

    # overlap per objective: keep unless the across-seam residual fails
    overlaps = {}
    for nose in sorted({t["nosepiece"] for t in measurable}):
        group = [t for t in measurable if t["nosepiece"] == nose]
        current = group[0]["summary"]["overlap"]
        src = "user" if nose in ov_user else "imaging_csv"
        entry = {"value": current, "source": src, "user_value": current}
        if any(t["summary"]["across_median_abs"] > thr for t in group):
            est = float(np.median([t["summary"]["overlap_estimate"] for t in group]))
            entry = {"value": round(est, 5), "source": "refit", "user_value": current,
                     "reason": f"across-seam median residual > {thr} px at overlap {current}"}
        overlaps[nose] = entry

    # rescore every measurable test raster at the chosen parameters
    failures = []
    rescored = []
    for t in measurable:
        sc = rastering.score_seams(t["seams"], rotation["value"], overlaps[t["nosepiece"]]["value"])
        ok = max(sc["along_median_abs"], sc["across_median_abs"]) <= thr
        rescored.append({"spec": t["spec"], "nosepiece": t["nosepiece"], "pass": ok,
                         "along_median_abs": sc["along_median_abs"],
                         "across_median_abs": sc["across_median_abs"]})
        if not ok:
            failures.append(t["spec"])

    out.update({"rotation": rotation, "overlap_by_nosepiece": overlaps, "rescored": rescored})
    run.resolved["stitching"] = plain(out)
    unmeasurable = [t["spec"] for t in tests if t["summary"]["status"] == "unmeasurable"]
    if failures:
        status = "needs_human"
        run.escalate("calibrate", "stitching",
                     f"Seam residuals still > {thr} px after refit on: {failures}",
                     "Inspect qc/seams/*.csv and test stitches; the rasters may disagree "
                     "(different rotation per objective?) or have poor features.")
    else:
        status = "warn" if unmeasurable or rotation["source"] != "user" or \
            any(o["source"] == "refit" for o in overlaps.values()) else "pass"
        run.close_escalations("calibrate")
    run.set_stage("calibrate", status, inputs_sha256=key, unmeasurable=unmeasurable)
    run.log("calibrate", rotation=rotation, overlaps=overlaps, rescored=rescored)
    return result("calibrate", status, **{k: out[k] for k in ("rotation", "overlap_by_nosepiece", "rescored")},
                  unmeasurable=unmeasurable)


# --------------------------------------------------------------------------- stitch / bgsub

def stitch(run: Run, redo: bool = False, overwrite: bool = False, **_) -> dict:
    from mercury.stitching import ImageStitcher

    require(run, "calibrate")
    cfg, st = run.cfg, run.cfg["stitching"]
    res = run.resolved["stitching"]
    rotation = res["rotation"]["value"]
    overlaps = {k: v["value"] for k, v in res["overlap_by_nosepiece"].items()}
    key = cfgmod.sha256_of({"rotation": rotation, "overlaps": overlaps, "ff": st["ff_correction"],
                            "method": st["method"], "acqui_origin": cfg["acqui_origin"],
                            "code": code_id(run)})
    if unchanged(run, "stitch", key, redo):
        return result("stitch", run.stage("stitch")["status"], cached=True)
    guard_overwrite(run, "stitch", run.root / "stitched_images" / "stitched_images.csv", overwrite)

    with captured(run, "stitch") as logfile:
        stitcher = ImageStitcher(run.root)
        rd = stitcher.raster_data
        rd.loc[ff_mask(rd["image_path_parent"].map(str), st["ff_correction"]), "apply_ff_correction"] = True
        for nose, ov in overlaps.items():
            rd.loc[nose_str(rd["nosepiece"]) == str(nose), "raster_overlap"] = ov
        stitcher.stitch_images(rotation=rotation, acqui_origin=tuple(cfg["acqui_origin"]),
                               method=st["method"], n_workers=cfg["n_workers"])
    stitched = pd.read_csv(run.root / "stitched_images" / "stitched_images.csv")
    expected = rd["image_path_parent"].nunique()
    status = "pass" if len(stitched) == expected else "warn"
    info = {"n_stitched": len(stitched), "n_rasters": expected, "rotation": rotation,
            "overlap_by_nosepiece": overlaps, "log": str(logfile)}
    run.set_stage("stitch", status, inputs_sha256=key, **info)
    run.log("stitch", **info)
    return result("stitch", status, **info)


def bgsub(run: Run, redo: bool = False, overwrite: bool = False, **_) -> dict:
    from mercury.stitching import BackgroundSubtractor

    require(run, "stitch")
    cfg = run.cfg
    sdir = run.root / "stitched_images"
    backgrounds = sorted(str(p) for p in sdir.rglob("*.tif")
                         if is_background(p.name, cfg["background"]["images"]))
    key = cfgmod.sha256_of({"backgrounds": backgrounds, "settings": cfg["background"]["settings_to_match"],
                            "stitch": run.stage("stitch").get("finished"), "code": code_id(run)})
    if unchanged(run, "bgsub", key, redo):
        return result("bgsub", run.stage("bgsub")["status"], cached=True)
    guard_overwrite(run, "bgsub", run.root / "bgsub_images" / "bgsub_images.csv", overwrite)

    with captured(run, "bgsub") as logfile:
        BackgroundSubtractor(run.root).subtract(
            background_images=backgrounds,
            settings_to_match=cfg["background"]["settings_to_match"],
            n_workers=cfg["n_workers"],
        )
    b = pd.read_csv(run.root / "bgsub_images" / "bgsub_images.csv")
    n_fail = int((~b["success"].astype(bool)).sum())
    no_bg = b.loc[b["background_image"].isna() & b["success"].astype(bool), "image_path"].tolist()
    status = "warn" if (n_fail or no_bg) else "pass"
    info = {"n_images": len(b), "n_failed": n_fail, "n_without_background": len(no_bg),
            "without_background": no_bg[:20], "backgrounds": [Path(p).name for p in backgrounds],
            "log": str(logfile)}
    run.resolved["background"] = plain(info)
    run.set_stage("bgsub", status, inputs_sha256=key, n_images=len(b), n_failed=n_fail)
    run.log("bgsub", **info)
    return result("bgsub", status, **info)


# --------------------------------------------------------------------------- corners

def corners(run: Run, redo: bool = False, **_) -> dict:
    from mercury.processing import corners as cq

    require(run, "bgsub")
    cfg, qc = run.cfg, run.cfg["qc"]
    previous: Dict[str, Tuple[list, str]] = {}
    rows, statuses = [], []
    for s in cfg["series"]:
        srec = run.series_record(s["name"])
        dev_recs = srec.setdefault("devices", {})
        images = series_images(run, s)
        for d in cfgmod.series_devices(cfg, s):
            dev = cfg["devices"][d]
            item = f"{s['name']}/{d}"
            if s["corners"].get(d):
                guess, gsrc = s["corners"][d], "series_config"
            elif d in previous:
                guess, gsrc = previous[d][0], f"previous_series:{previous[d][1]}"
            elif dev.get("corners"):
                guess, gsrc = dev["corners"], "device_config"
            else:
                run.escalate("corners", item, "No corner guess for this device.",
                             f"Run pick-corners on a {d} image and set devices.{d}.corners.")
                statuses.append("needs_human")
                continue
            dimg = images[images["dname"] == d]
            if dimg.empty:
                raise StageError(f"series {s['name']}: no images for device {d}")

            ref_spec = s["reference"]
            key = cfgmod.sha256_of({"guess": guess, "features": s["features"], "ref": ref_spec,
                                    "dims": dev["dims"], "bgsub": run.stage("bgsub").get("finished"),
                                    "code": code_id(run),
                                    "qc": {k: v for k, v in qc.items() if k.startswith("corner")}})
            old = dev_recs.get(d, {})
            if not redo and old.get("inputs_sha256") == key and old.get("corner_qc", {}).get("status") == "pass":
                previous[d] = (old["corners"]["value"], s["name"])
                rows.append({"item": item, "status": "pass", "cached": True})
                statuses.append("pass")
                continue

            image, rule = pick_image(dimg, ref_spec, run.root)
            feats = "chamber" if s["features"] == "all" else s["features"]
            summary, offsets, history = cq.refine_corners(
                image, guess, tuple(dev["dims"]), features=feats,
                max_corner_shift_px=qc["corner_max_shift_px"],
                max_residual_px=qc["corner_max_residual_px"], max_iter=qc["corner_max_iter"],
            )
            qdir = run.root / "qc" / "corners"
            qdir.mkdir(parents=True, exist_ok=True)
            offsets_csv = qdir / f"{s['name']}_{d}.csv"
            offsets.to_csv(offsets_csv, index=False)

            qc_summary = {k: v for k, v in summary.items() if k != "history"}
            qc_summary["iterations"] = len(history)
            qc_summary["first_shift_px"] = history[0].get("max_corner_shift_px")
            rec = {"image": {"path": str(image), "rule": rule}, "corner_qc": qc_summary,
                   "offsets_csv": str(offsets_csv), "inputs_sha256": key}
            if summary["status"] == "pass":
                final = [list(c) for c in summary["proposed_corners"]]
                moved = np.abs(np.array(final) - np.array(guess)).max()
                rec["corners"] = {"value": final, "source": "refit" if moved > 0 else gsrc,
                                  "guess": guess, "guess_source": gsrc,
                                  "max_shift_from_guess_px": float(moved)}
                previous[d] = (final, s["name"])
                run.close_escalations("corners", item)
                statuses.append("pass")
            else:
                rec["corners"] = {"value": None, "guess": guess, "guess_source": gsrc}
                run.escalate(
                    "corners", item,
                    f"corner check {summary['status']}: shift {summary.get('max_corner_shift_px')}, "
                    f"residual {summary.get('residual_median_px')}, ring pass "
                    f"{(summary.get('ring') or {}).get('pass')}",
                    f"Run `pick-corners {image}` and set series.{s['name']}.corners.{d} "
                    f"(or devices.{d}.corners) in processing.yaml, then re-run corners.",
                )
                statuses.append("needs_human")
            dev_recs[d] = plain(rec)
            rows.append({"item": item, "status": summary["status"], "image": Path(image).name,
                         "shift_first_px": history[0].get("max_corner_shift_px"),
                         "iterations": len(history),
                         "corners": rec["corners"]["value"]})
            run.log("corners", item=item, **plain(rec))

    status = "needs_human" if "needs_human" in statuses else "pass"
    run.set_stage("corners", status)
    return result("corners", status, devices=rows)


# --------------------------------------------------------------------------- reference

def _roi_cache_key(run: Run, image: Path, corners_value, s: dict, d: str) -> str:
    stat = Path(image).stat()
    return cfgmod.sha256_of({"image": str(image), "size": stat.st_size, "mtime": stat.st_mtime,
                             "corners": corners_value, "features": s["features"],
                             "coerce": s["coerce_chamber_center"], "dims": run.cfg["devices"][d]["dims"],
                             "code": code_id(run)})


def reference(run: Run, redo: bool = False, **_) -> dict:
    from mercury.processing import Processor

    require(run, "corners")
    cfg = run.cfg
    experiment = None
    rows = []
    for s in cfg["series"]:
        if s["reference"] is None:
            continue
        srec = run.series_record(s["name"])
        images = None
        for d in cfgmod.series_devices(cfg, s):
            drec = srec["devices"][d]
            image = Path(drec["image"]["path"])
            corner_value = drec["corners"]["value"]
            key = _roi_cache_key(run, image, corner_value, s, d)
            cache = run.root / "qc" / "roi_cache" / f"{s['name']}_{d}_{key[:16]}.pkl"
            if cache.exists() and not redo:
                drec["roi_cache"] = {"path": str(cache), "key": key, "reused": True}
                rows.append({"item": f"{s['name']}/{d}", "cached": True})
                continue
            if experiment is None:
                experiment = build_experiment(run)
            if images is None:
                images = series_images(run, s)
            with captured(run, "reference"):
                proc = Processor(experiment, image_data=images[images["dname"] == d].copy(), features=s["features"])
                proc.set_corners(d, [tuple(int(v) for v in c) for c in corner_value])
                proc.set_reference(image=image, save_summary_images=True,
                                   coerce_chamber_center=s["coerce_chamber_center"])
                roi_set = proc.reference_rois[d]
            cache.parent.mkdir(parents=True, exist_ok=True)
            with open(cache, "wb") as f:
                pickle.dump(roi_set, f)
            drec["roi_cache"] = {"path": str(cache), "key": key, "reused": False}
            rows.append({"item": f"{s['name']}/{d}", "cached": False, "image": image.name})
            run.log("reference", item=f"{s['name']}/{d}", cache=str(cache), image=str(image))
    run.set_stage("reference", "pass")
    return result("reference", "pass", references=rows)


# --------------------------------------------------------------------------- map

def map_(run: Run, redo: bool = False, overwrite: bool = False, **_) -> dict:
    from mercury.processing import Processor

    require(run, "corners")
    cfg = run.cfg
    if any(s["reference"] is not None for s in cfg["series"]):
        require(run, "reference")
    experiment = build_experiment(run)
    outdir = run.root / cfg["output_dir"]
    rows = []
    for s in cfg["series"]:
        srec = run.series_record(s["name"])
        devs = cfgmod.series_devices(cfg, s)
        drecs = {d: srec["devices"][d] for d in devs}
        key = cfgmod.sha256_of({"series": {k: v for k, v in s.items()},
                                "corners": {d: r["corners"]["value"] for d, r in drecs.items()},
                                "roi": {d: (r.get("roi_cache") or {}).get("key") for d, r in drecs.items()},
                                "bgsub": run.stage("bgsub").get("finished"),
                                "code": code_id(run)})
        out = outdir / s["output"]
        prev = srec.get("output") or {}
        if (not redo and prev.get("inputs_sha256") == key and out.exists()
                and file_sha256(out) == prev.get("sha256")):
            rows.append({"series": s["name"], "cached": True, "path": str(out)})
            continue
        if out.exists() and file_sha256(out) != prev.get("sha256") and not overwrite:
            raise StageError(f"{out} exists and was not produced by this run record; use --overwrite (ask the user).")

        images = series_images(run, s)
        with captured(run, f"map_{s['name']}") as logfile:
            proc = Processor(experiment, image_data=images.copy(), features=s["features"])
            for d, r in drecs.items():
                proc.set_corners(d, [tuple(int(v) for v in c) for c in r["corners"]["value"]])
            if s["reference"] is not None:
                for d, r in drecs.items():
                    with open(r["roi_cache"]["path"], "rb") as f:
                        proc.reference_rois[d] = pickle.load(f)
                    proc.image_data.loc[proc.image_data["dname"] == d, "reference_image"] = r["image"]["path"]
                data = proc.process(use_reference=True, save_summary_images=s["save_summary_images"])
            else:
                data = proc.process(use_reference=False, save_summary_images=s["save_summary_images"],
                                    coerce_chamber_center=s["coerce_chamber_center"])
        out.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(out)
        info = {"path": str(out), "n_rows": int(len(data)), "n_images": int(images["image_path"].nunique()),
                "sha256": file_sha256(out), "inputs_sha256": key, "log": str(logfile), "finished": now()}
        srec["output"] = info
        rows.append({"series": s["name"], **{k: info[k] for k in ("path", "n_rows", "n_images")}})
        run.log("map", series=s["name"], **info)
    run.set_stage("map", "pass", outputs=[r.get("path") for r in rows])
    return result("map", "pass", outputs=rows)


STAGES: Dict[str, Callable[..., dict]] = {
    "preflight": preflight, "calibrate": calibrate, "stitch": stitch, "bgsub": bgsub,
    "corners": corners, "reference": reference, "map": map_,
}
DECISION_STAGES = ("calibrate", "corners")
