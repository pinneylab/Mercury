"""Corner quality: score device corners from where features land in their stamps.

Stamp centers are bilinear in the four corners (ChipImage.quadrilateralInterp):

    P(u, v) = (1-u)(1-v)·UL + u(1-v)·UR + (1-u)v·BL + uv·BR,
    u = ix / (nx - 1),  v = iy / (ny - 1).

If the corners are off by δ_UL..δ_BR, the feature offset from each stamp center
(found - predicted) is therefore the same bilinear blend of the δ's. A robust
least-squares fit of the offsets gives a corner correction (proposed = corners + δ)
and residuals that show what corners cannot explain (bad stitch, curvature).

Offsets cannot see an off-by-one-pitch error: every stamp still holds a real
unit cell. ring_check catches that by comparing stamps one pitch outside the
grid (should look like nothing) with the outermost grid row/column (should look
like the interior), using correlation against a median interior stamp.

Coordinates: corners and centers are (x, y) = (column, row) in stitched-image px,
the convention of set_corners and the pick-corners app.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from skimage import io

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from mercury.processing.chip import (
    _chamber_worker,
    _button_worker,
    _resolve_n_jobs,
)

CORNER_NAMES = ("ul", "ur", "bl", "br")
SIDES = ("top", "bottom", "left", "right")


# --------------------------------------------------------------------------- geometry

def bilinear_weights(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(N, 4) weights of (UL, UR, BL, BR) at lattice coordinates (u, v)."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    return np.stack([(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v], axis=1)


def lattice_points(
    corners: Sequence[Tuple[float, float]],
    dims: Tuple[int, int],
    indices: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Predicted stamp centers (float px) for lattice indices.

    Args:
        corners: (UL, UR, BL, BR), each (x, y).
        dims: (nx, ny) = Device.dims (columns, rows of unit cells).
        indices: (N, 2) array of (ix, iy); may lie outside [0, n) to extrapolate one
            pitch beyond the grid. Defaults to the full grid.

    Returns:
        DataFrame with ix, iy, u, v, x, y.
    """
    nx, ny = int(dims[0]), int(dims[1])
    if indices is None:
        ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        indices = np.stack([ix.ravel(), iy.ravel()], axis=1)
    indices = np.asarray(indices, int)
    u = indices[:, 0] / (nx - 1)
    v = indices[:, 1] / (ny - 1)
    xy = bilinear_weights(u, v) @ np.asarray(corners, float)
    return pd.DataFrame({"ix": indices[:, 0], "iy": indices[:, 1], "u": u, "v": v,
                         "x": xy[:, 0], "y": xy[:, 1]})


def sample_indices(dims: Tuple[int, int], interior_step: int = 4) -> np.ndarray:
    """All edge cells plus every interior_step-th interior cell (enough to fit 8 numbers)."""
    nx, ny = int(dims[0]), int(dims[1])
    out = []
    for ix in range(nx):
        for iy in range(ny):
            edge = ix in (0, nx - 1) or iy in (0, ny - 1)
            if edge or (ix % interior_step == 0 and iy % interior_step == 0):
                out.append((ix, iy))
    return np.array(out, int)


def ring_indices(dims: Tuple[int, int]) -> Dict[str, Dict[str, np.ndarray]]:
    """Per side: 'outer' = one pitch outside the grid, 'edge' = outermost grid row/column."""
    nx, ny = int(dims[0]), int(dims[1])
    xs, ys = np.arange(nx), np.arange(ny)

    def pts(a, b):
        return np.stack(np.broadcast_arrays(a, b), axis=1)

    return {
        "top": {"outer": pts(xs, -1), "edge": pts(xs, 0)},
        "bottom": {"outer": pts(xs, ny), "edge": pts(xs, ny - 1)},
        "left": {"outer": pts(-1, ys), "edge": pts(0, ys)},
        "right": {"outer": pts(nx, ys), "edge": pts(nx - 1, ys)},
    }


def _crop(image: np.ndarray, x: float, y: float, width: int) -> Optional[np.ndarray]:
    """Stamp around (x, y), same slicing as ChipImage._stamp; None if it leaves the image."""
    cx, cy, h = int(round(x)), int(round(y)), width // 2
    r0, r1, c0, c1 = cy - h, cy + h, cx - h, cx + h
    if r0 < 0 or c0 < 0 or r1 > image.shape[0] or c1 > image.shape[1]:
        return None
    return image[r0:r1, c0:c1]


def _load(image: Union[str, Path, np.ndarray]) -> np.ndarray:
    return image if isinstance(image, np.ndarray) else io.imread(image)


# --------------------------------------------------------------------------- offsets

def _quiet():
    warnings.simplefilter("ignore")  # "No chamber border found ..." for every empty stamp


def _quiet_pool(workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                               initializer=_quiet)


def measure_offsets(
    image: Union[str, Path, np.ndarray],
    corners: Sequence[Tuple[float, float]],
    dims: Tuple[int, int],
    features: str = "chamber",
    indices: Optional[np.ndarray] = None,
    interior_step: int = 4,
    stamp_width: int = 100,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Run the production feature finder on sampled stamps; report found - predicted.

    Args:
        image: Stitched (background-subtracted) image or its path.
        corners, dims: As in lattice_points.
        features: 'chamber' (Hough, Stamp.findChamber) or 'button' (Stamp.findButton).
        indices: Lattice cells to measure; default sample_indices(dims, interior_step).
        stamp_width: Stamp size in px (ChipImage.stampWidth).
        n_jobs: Worker processes (-1 = all CPUs, 1 = serial).

    Returns:
        DataFrame: ix, iy, u, v, x, y (predicted), in_bounds, found,
        fx, fy (found center, image px), radius, dx, dy (found - predicted, px).
    """
    if features not in ("chamber", "button"):
        raise ValueError("features must be 'chamber' or 'button'")
    img = _load(image)
    pts = lattice_points(corners, dims, sample_indices(dims, interior_step) if indices is None else indices)

    crops, origins = {}, {}
    for i, row in pts.iterrows():
        s = _crop(img, row.x, row.y, stamp_width)
        if s is not None:
            crops[i] = s
            origins[i] = (int(round(row.x)) - stamp_width // 2, int(round(row.y)) - stamp_width // 2)

    if features == "chamber":
        payloads, worker = [(i, s, False) for i, s in crops.items()], _chamber_worker
    else:
        payloads, worker = [(i, s) for i, s in crops.items()], _button_worker

    workers = _resolve_n_jobs(n_jobs)
    if workers <= 1 or len(payloads) < 2:
        with warnings.catch_warnings():
            _quiet()
            results = [worker(p) for p in payloads]
    else:
        chunksize = max(1, len(payloads) // (workers * 4))
        with _quiet_pool(workers) as ex:
            results = list(ex.map(worker, payloads, chunksize=chunksize))

    pts["in_bounds"] = [i in crops for i in pts.index]
    pts["found"] = False
    for col in ("fx", "fy", "radius", "dx", "dy"):
        pts[col] = np.nan
    for i, geom in results:
        if geom is None:
            continue
        if features == "chamber":
            (sx, sy), radius = geom
        else:
            (sx, sy), radius = geom[0], geom[1]
        ox, oy = origins[i]
        pts.loc[i, ["found", "fx", "fy", "radius"]] = [True, ox + sx, oy + sy, radius]
    pts["found"] = pts["found"].astype(bool)
    pts["dx"] = pts["fx"] - pts["x"]
    pts["dy"] = pts["fy"] - pts["y"]
    return pts


def fit_corner_correction(
    offsets: pd.DataFrame,
    floor_px: float = 2.0,
    k: float = 3.0,
    n_iter: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust bilinear fit of feature offsets to per-corner corrections.

    Solves offsets ≈ W·δ (W = bilinear_weights(u, v)) by least squares, iteratively
    dropping stamps whose residual exceeds max(floor_px, k·1.4826·MAD).

    Returns:
        delta: (4, 2) corrections (dx, dy) for UL, UR, BL, BR; proposed = corners + delta.
        inlier: boolean mask over offsets rows.
        residual: (N, 2) offsets minus fitted field (NaN where not found).
    """
    ok = offsets["found"].to_numpy(bool)
    W = bilinear_weights(offsets["u"], offsets["v"])
    D = offsets[["dx", "dy"]].to_numpy(float)
    inlier = ok.copy()
    delta = np.zeros((4, 2))
    for _ in range(n_iter):
        if inlier.sum() < 4:
            raise ValueError(f"Only {int(inlier.sum())} usable stamps; cannot fit corners.")
        delta, *_ = np.linalg.lstsq(W[inlier], D[inlier], rcond=None)
        res = np.linalg.norm(D - W @ delta, axis=1)
        med = np.median(res[inlier])
        mad = 1.4826 * np.median(np.abs(res[inlier] - med))
        new = ok & (res <= max(floor_px, med + k * mad))
        if np.array_equal(new, inlier):
            break
        inlier = new
    residual = D - W @ delta
    return delta, inlier, residual


# --------------------------------------------------------------------------- off-by-one guard

def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(float).ravel() - a.mean()
    b = b.astype(float).ravel() - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def ring_check(
    image: Union[str, Path, np.ndarray],
    corners: Sequence[Tuple[float, float]],
    dims: Tuple[int, int],
    stamp_width: int = 100,
    interior_step: int = 3,
    min_contrast: float = 0.2,
    min_edge_ratio: float = 0.3,
) -> Dict[str, dict]:
    """Detect a grid shifted by whole pitches.

    Builds a template (median of interior stamps) and scores stamps by their
    correlation (ncc) with it. For each side, compare the outermost grid row/column
    ('edge') with the ring one pitch outside the grid ('outer'):

        edge_ncc - outer_ncc >= min_contrast         (edge looks more like a unit cell)
        edge_ncc >= min_edge_ratio * interior_ncc    (edge isn't empty)

    A grid shifted by one pitch puts an empty row/column inside the grid on one side
    (edge ≈ outer ≈ 0) and a real one outside on the other (outer ≈ edge).
    The outer ring is compared with the edge rather than with zero because it can hit
    bright off-array structures (e.g. NADH-filled inlet channels), and the edge floor
    is loose because edge cells can be legitimately empty (blank print spots).
    Defaults come from four 20260730 images: correct corners gave contrast >= 0.25 and
    edge/interior >= 0.43; one-pitch shifts gave contrast <= 0.13 on the failing side.
    A side whose outer ring leaves the image is judged on the edge floor alone.

    Returns:
        {'interior_ncc': float, 'pass': bool,
         side: {'outer_ncc', 'edge_ncc', 'contrast', 'n_outer', 'n_edge', 'pass'}}
    """
    img = _load(image)
    nx, ny = int(dims[0]), int(dims[1])
    interior = [(ix, iy) for ix in range(1, nx - 1, interior_step) for iy in range(1, ny - 1, interior_step)]
    pts = lattice_points(corners, dims, np.array(interior, int))
    stack = [s for s in (_crop(img, r.x, r.y, stamp_width) for r in pts.itertuples()) if s is not None]
    if len(stack) < 4:
        raise ValueError("Too few interior stamps inside the image to build a template.")
    stack = np.asarray(stack, float)
    template = np.median(stack, axis=0)
    # Leave-in bias is small with hundreds of stamps; score every other stamp to be safe.
    interior_ncc = float(np.median([_ncc(s, template) for s in stack[1::2]]))

    def side_ncc(idx):
        p = lattice_points(corners, dims, idx)
        vals = [_ncc(s, template) for s in (_crop(img, r.x, r.y, stamp_width) for r in p.itertuples()) if s is not None]
        return (float(np.median(vals)) if vals else float("nan")), len(vals)

    out = {"interior_ncc": interior_ncc}
    all_pass = True
    for side, idx in ring_indices(dims).items():
        outer, n_outer = side_ncc(idx["outer"])
        edge, n_edge = side_ncc(idx["edge"])
        contrast = edge - outer
        ok_edge = edge >= min_edge_ratio * interior_ncc
        ok_contrast = np.isnan(outer) or contrast >= min_contrast
        out[side] = {"outer_ncc": outer, "edge_ncc": edge, "contrast": contrast,
                     "n_outer": n_outer, "n_edge": n_edge,
                     "pass": bool(ok_edge and ok_contrast)}
        all_pass &= out[side]["pass"]
    out["pass"] = bool(all_pass)
    return out


# --------------------------------------------------------------------------- top level

def _as_int_corners(c: np.ndarray) -> List[Tuple[int, int]]:
    return [(int(round(x)), int(round(y))) for x, y in np.asarray(c, float)]


def check_corners(
    image: Union[str, Path, np.ndarray],
    corners: Sequence[Tuple[float, float]],
    dims: Tuple[int, int],
    features: str = "chamber",
    stamp_width: int = 100,
    interior_step: int = 4,
    max_corner_shift_px: float = 3.0,
    max_residual_px: float = 10.0,
    min_inliers: int = 20,
    ring: bool = True,
    n_jobs: int = -1,
) -> Tuple[dict, pd.DataFrame]:
    """Score corners on one image; propose corrected corners.

    The per-stamp scatter of the production finders sets the residual floor: on 20260730
    the median |residual| was 3.7-5 px (chamber Hough) and 4.4-7 px (button). So
    max_residual_px is a gross-failure check (e.g. a mis-stitched tile), not a precision
    one; the fitted corrections themselves are reproducible to ~1 px.

    Status:
        'pass'          ring check passes, every |δ| <= max_corner_shift_px and the median
                        |residual| <= max_residual_px.
        'refit'         ring passes but corners should move: rerun with proposed_corners.
        'fail'          ring check fails (grid likely shifted by a whole pitch) or the
                        lattice doesn't fit (median residual too large) -> ask the user.
        'unmeasurable'  fewer than min_inliers stamps with a feature.

    Returns:
        (summary dict, per-stamp offsets DataFrame with fit residuals and inlier flags).
    """
    image = _load(image)
    corners = np.asarray(corners, float)
    offsets = measure_offsets(image, corners, dims, features=features,
                              interior_step=interior_step, stamp_width=stamp_width, n_jobs=n_jobs)
    summary = {
        "corners": _as_int_corners(corners),
        "features": features,
        "n_stamps": int(len(offsets)),
        "n_in_bounds": int(offsets["in_bounds"].sum()),
        "n_found": int(offsets["found"].sum()),
    }

    try:
        delta, inlier, residual = fit_corner_correction(offsets)
    except ValueError:
        delta, inlier, residual = None, np.zeros(len(offsets), bool), np.full((len(offsets), 2), np.nan)
    offsets["inlier"] = inlier
    offsets["res_x"], offsets["res_y"] = residual[:, 0], residual[:, 1]
    summary["n_inliers"] = int(inlier.sum())

    raw = np.hypot(offsets["dx"], offsets["dy"])[offsets["found"]]
    res = np.hypot(offsets["res_x"], offsets["res_y"])[inlier]
    summary["offset_median_px"] = float(np.median(raw)) if len(raw) else float("nan")
    summary["offset_p95_px"] = float(np.percentile(raw, 95)) if len(raw) else float("nan")
    summary["residual_median_px"] = float(np.median(res)) if len(res) else float("nan")
    summary["residual_p95_px"] = float(np.percentile(res, 95)) if len(res) else float("nan")
    if features == "chamber" and offsets["found"].any():
        summary["radius_median_px"] = float(offsets.loc[offsets["found"], "radius"].median())

    if delta is None or summary["n_inliers"] < min_inliers:
        summary.update({"delta": None, "max_corner_shift_px": float("nan"),
                        "proposed_corners": None, "ring": None, "status": "unmeasurable"})
        return summary, offsets

    shifts = np.hypot(delta[:, 0], delta[:, 1])
    summary["delta"] = {n: (float(d[0]), float(d[1])) for n, d in zip(CORNER_NAMES, delta)}
    summary["max_corner_shift_px"] = float(shifts.max())
    proposed = corners + delta
    summary["proposed_corners"] = _as_int_corners(proposed)

    # Judge the ring on the better-aligned grid.
    summary["ring"] = ring_check(image, proposed, dims, stamp_width=stamp_width) if ring else None
    ring_ok = summary["ring"]["pass"] if ring else True

    if not ring_ok or summary["residual_median_px"] > max_residual_px:
        status = "fail"
    elif summary["max_corner_shift_px"] <= max_corner_shift_px:
        status = "pass"
    else:
        status = "refit"
    summary.update({"max_corner_shift_px_allowed": max_corner_shift_px,
                    "max_residual_px_allowed": max_residual_px, "status": status})
    return summary, offsets


def refine_corners(
    image: Union[str, Path, np.ndarray],
    corners: Sequence[Tuple[float, float]],
    dims: Tuple[int, int],
    max_iter: int = 3,
    **kwargs,
) -> Tuple[dict, pd.DataFrame, List[dict]]:
    """check_corners, re-run on proposed_corners while status is 'refit'.

    Large errors are clipped by the stamp size (a chamber only fits whole within about
    ±(stamp_width/2 - radius) px), so one fit can under-correct; iterating converges.

    Returns:
        (final summary, final offsets, history of summaries).
    """
    image = _load(image)
    history = []
    current = corners
    for _ in range(max_iter):
        summary, offsets = check_corners(image, current, dims, **kwargs)
        history.append(summary)
        if summary["status"] != "refit":
            break
        current = summary["proposed_corners"]
    return summary, offsets, history
