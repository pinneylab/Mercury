"""RoiSet-based stamp quantification and summary rendering.

Masks in RoiSet use True = inside the ROI. Legacy circularSubsection /
Chamber.disk masks use numpy.ma convention (True = ignore); convert with ``~``
when exporting from a found ChipImage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import skimage.io

from mercury.processing.chip import (
    ChipImage,
    annotateStamp,
    _resolve_n_jobs,
    _stamp_process_pool,
)

PathLike = Union[str, Path]

CHAMBER_COLS = [
    "median_chamber",
    "sum_chamber",
    "std_chamber",
    "x_center_chamber",
    "y_center_chamber",
    "radius_chamber",
]

BUTTON_DISK_COLS = [
    "median_button",
    "summed_button",
    "summed_button_BGsub",
    "std_button",
    "mean_button",
    "x_button_center",
    "y_button_center",
    "radius_button_disk",
]

BUTTON_ANN_COLS = [
    "median_button_annulus",
    "summed_button_annulus_normed",
    "std_button_annulus_localBG",
    "mean_annulus",
    "inner_radius_button_annulus",
    "outer_radius_button_annulus",
]

BUTTON_COLS = BUTTON_DISK_COLS + BUTTON_ANN_COLS
STAMP_META_COLS = ["xslice", "yslice", "id"]


@dataclass
class RoiSet:
    """Geometry and inside-masks for quantifying stamps without Stamp objects."""

    features: str
    stamp_shape: Tuple[int, int]
    indices: np.ndarray  # (N, 2)
    ids: np.ndarray  # (N,)
    row0: np.ndarray
    row1: np.ndarray
    col0: np.ndarray
    col1: np.ndarray

    chamber_blank: Optional[np.ndarray] = None
    chamber_centers: Optional[np.ndarray] = None  # (N, 2)
    chamber_radii: Optional[np.ndarray] = None
    chamber_masks: Optional[np.ndarray] = None  # (N, H, W) True=inside

    button_blank: Optional[np.ndarray] = None
    button_centers: Optional[np.ndarray] = None
    button_disk_radii: Optional[np.ndarray] = None
    button_annulus_radii: Optional[np.ndarray] = None  # (N, 2) inner, outer
    button_disk_masks: Optional[np.ndarray] = None
    button_annulus_masks: Optional[np.ndarray] = None

    @property
    def n(self) -> int:
        return int(len(self.indices))

    @classmethod
    def from_chip(cls, chip: ChipImage, features: str) -> "RoiSet":
        """Build an RoiSet from a ChipImage after feature finding."""
        if features not in ("chamber", "button", "all"):
            raise ValueError("features must be 'chamber', 'button', or 'all'")
        if chip.stamps is None:
            raise ValueError("chip has no stamps; call stamp() and find* first")

        stamps = list(chip.stamps.flatten())
        n = len(stamps)
        h, w = stamps[0].data.shape
        indices = np.zeros((n, 2), dtype=int)
        ids = np.empty(n, dtype=object)
        row0 = np.zeros(n, dtype=int)
        row1 = np.zeros(n, dtype=int)
        col0 = np.zeros(n, dtype=int)
        col1 = np.zeros(n, dtype=int)

        want_chamber = features in ("chamber", "all")
        want_button = features in ("button", "all")

        chamber_blank = np.zeros(n, dtype=bool) if want_chamber else None
        chamber_centers = np.full((n, 2), np.nan) if want_chamber else None
        chamber_radii = np.full(n, np.nan) if want_chamber else None
        chamber_masks = np.zeros((n, h, w), dtype=bool) if want_chamber else None

        button_blank = np.zeros(n, dtype=bool) if want_button else None
        button_centers = np.full((n, 2), np.nan) if want_button else None
        button_disk_radii = np.full(n, np.nan) if want_button else None
        button_annulus_radii = np.full((n, 2), np.nan) if want_button else None
        button_disk_masks = np.zeros((n, h, w), dtype=bool) if want_button else None
        button_annulus_masks = np.zeros((n, h, w), dtype=bool) if want_button else None

        for i, s in enumerate(stamps):
            indices[i] = s.index
            ids[i] = s.id
            row0[i] = s.slice[0].start
            row1[i] = s.slice[0].stop
            col0[i] = s.slice[1].start
            col1[i] = s.slice[1].stop

            if want_chamber:
                ch = s.chamber
                if ch is None or ch.blankFlag:
                    chamber_blank[i] = True
                else:
                    chamber_blank[i] = False
                    chamber_centers[i] = (ch.center[0], ch.center[1])
                    chamber_radii[i] = ch.radius
                    # ch.disk is ma-style (True=outside) → True=inside
                    chamber_masks[i] = ~np.asarray(ch.disk, dtype=bool)

            if want_button:
                bt = s.button
                if bt is None or bt.blankFlag:
                    button_blank[i] = True
                else:
                    button_blank[i] = False
                    button_centers[i] = (bt.center[0], bt.center[1])
                    button_disk_radii[i] = bt.disk_radius
                    button_annulus_radii[i] = (
                        bt.annulus_radii[0],
                        bt.annulus_radii[1],
                    )
                    button_disk_masks[i] = ~np.asarray(bt.disk, dtype=bool)
                    button_annulus_masks[i] = ~np.asarray(bt.annulus, dtype=bool)

        return cls(
            features=features,
            stamp_shape=(h, w),
            indices=indices,
            ids=ids,
            row0=row0,
            row1=row1,
            col0=col0,
            col1=col1,
            chamber_blank=chamber_blank,
            chamber_centers=chamber_centers,
            chamber_radii=chamber_radii,
            chamber_masks=chamber_masks,
            button_blank=button_blank,
            button_centers=button_centers,
            button_disk_radii=button_disk_radii,
            button_annulus_radii=button_annulus_radii,
            button_disk_masks=button_disk_masks,
            button_annulus_masks=button_annulus_masks,
        )


def stamps_from_chip(chip: ChipImage) -> np.ndarray:
    """Stack stamp pixel data from a ChipImage to shape (N, H, W)."""
    stamps = list(chip.stamps.flatten())
    return np.stack([s.data for s in stamps], axis=0)


def _open_image(path: PathLike) -> np.ndarray:
    path = Path(path)
    try:
        import tifffile

        return tifffile.memmap(str(path), mode="r")
    except Exception:
        return skimage.io.imread(path)


def extract_stamps(
    image: Union[PathLike, np.ndarray],
    roi: RoiSet,
) -> np.ndarray:
    """Crop stamps to shape (N, H, W) using RoiSet slices."""
    if not isinstance(image, np.ndarray):
        img = _open_image(image)
    else:
        img = image

    h, w = roi.stamp_shape
    n = roi.n
    out = np.empty((n, h, w), dtype=img.dtype)
    for i in range(n):
        out[i] = img[roi.row0[i] : roi.row1[i], roi.col0[i] : roi.col1[i]]
    return out


def _masked_sum(stamps: np.ndarray, inside: np.ndarray) -> np.ndarray:
    return (stamps.astype(np.float64) * inside).sum(axis=(1, 2))


def _masked_count(inside: np.ndarray) -> np.ndarray:
    return inside.reshape(inside.shape[0], -1).sum(axis=1).astype(np.float64)


def _masked_mean(stamps: np.ndarray, inside: np.ndarray) -> np.ndarray:
    counts = _masked_count(inside)
    sums = _masked_sum(stamps, inside)
    out = np.full(stamps.shape[0], np.nan, dtype=np.float64)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def _masked_std(stamps: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Population std (ddof=0), matching numpy.ma.std default."""
    counts = _masked_count(inside)
    means = _masked_mean(stamps, inside)
    stamps_f = stamps.astype(np.float64)
    diff2 = (stamps_f - means[:, None, None]) ** 2
    ssq = (diff2 * inside).sum(axis=(1, 2))
    out = np.full(stamps.shape[0], np.nan, dtype=np.float64)
    np.sqrt(np.divide(ssq, counts, where=counts > 0), out=out, where=counts > 0)
    out[counts <= 0] = np.nan
    return out


def _masked_median_loop(stamps: np.ndarray, inside: np.ndarray) -> np.ndarray:
    n = stamps.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        pix = stamps[i][inside[i]]
        if pix.size:
            out[i] = np.median(pix)
    return out


def _nan_cols(cols: Sequence[str], n: int) -> Dict[str, np.ndarray]:
    return {c: np.full(n, np.nan) for c in cols}


def _quantify_chamber(stamps: np.ndarray, roi: RoiSet) -> Dict[str, np.ndarray]:
    n = roi.n
    out = _nan_cols(CHAMBER_COLS, n)
    blank = roi.chamber_blank
    valid = ~blank
    if not valid.any():
        return out

    inside = roi.chamber_masks[valid]
    sub = stamps[valid]
    med = _masked_median_loop(sub, inside)
    sm = _masked_sum(sub, inside)
    sd = _masked_std(sub, inside)

    out["median_chamber"][valid] = med.astype(int)
    out["sum_chamber"][valid] = sm.astype(int)
    out["std_chamber"][valid] = sd.astype(int)
    out["x_center_chamber"][valid] = roi.chamber_centers[valid, 0]
    out["y_center_chamber"][valid] = roi.chamber_centers[valid, 1]
    out["radius_chamber"][valid] = roi.chamber_radii[valid]
    return out


def _quantify_button(stamps: np.ndarray, roi: RoiSet) -> Dict[str, np.ndarray]:
    n = roi.n
    out = _nan_cols(BUTTON_COLS, n)
    blank = roi.button_blank
    valid = ~blank
    if not valid.any():
        return out

    disk_in = roi.button_disk_masks[valid]
    ann_in = roi.button_annulus_masks[valid]
    sub = stamps[valid]

    med_d = _masked_median_loop(sub, disk_in)
    sum_d = _masked_sum(sub, disk_in)
    std_d = _masked_std(sub, disk_in)
    mean_d = _masked_mean(sub, disk_in)

    med_a = _masked_median_loop(sub, ann_in)
    sum_a = _masked_sum(sub, ann_in)
    std_a = _masked_std(sub, ann_in)
    mean_a = _masked_mean(sub, ann_in)

    count_d = _masked_count(disk_in)
    count_a = _masked_count(ann_in)
    ratio = np.divide(count_a, count_d, out=np.full_like(count_a, np.nan), where=count_d > 0)
    sum_a_normed = np.divide(
        sum_a, ratio, out=np.full_like(sum_a, np.nan), where=np.isfinite(ratio) & (ratio > 0)
    )
    sum_bgsub = sum_d - sum_a_normed

    out["median_button"][valid] = med_d.astype(int)
    out["summed_button"][valid] = sum_d.astype(int)
    out["summed_button_BGsub"][valid] = sum_bgsub.astype(int)
    out["std_button"][valid] = std_d.astype(int)
    out["mean_button"][valid] = mean_d.astype(int)
    out["x_button_center"][valid] = roi.button_centers[valid, 0]
    out["y_button_center"][valid] = roi.button_centers[valid, 1]
    out["radius_button_disk"][valid] = roi.button_disk_radii[valid]

    out["median_button_annulus"][valid] = med_a.astype(int)
    out["summed_button_annulus_normed"][valid] = sum_a_normed.astype(int)
    out["std_button_annulus_localBG"][valid] = std_a.astype(int)
    out["mean_annulus"][valid] = mean_a.astype(int)
    out["inner_radius_button_annulus"][valid] = roi.button_annulus_radii[valid, 0]
    out["outer_radius_button_annulus"][valid] = roi.button_annulus_radii[valid, 1]
    return out


def quantify(
    stamps_or_path: Union[PathLike, np.ndarray],
    roi: RoiSet,
    features: Optional[str] = None,
) -> pd.DataFrame:
    """Quantify stamps with an RoiSet. Accepts a stamp tensor or image path."""
    features = features or roi.features
    if isinstance(stamps_or_path, np.ndarray) and stamps_or_path.ndim == 3:
        stamps = stamps_or_path
    else:
        stamps = extract_stamps(stamps_or_path, roi)

    if stamps.shape[0] != roi.n:
        raise ValueError(f"stamp count {stamps.shape[0]} != roi.n {roi.n}")

    records: Dict[str, Any] = {}
    if features in ("chamber", "all"):
        if roi.chamber_masks is None:
            raise ValueError("RoiSet has no chamber masks")
        records.update(_quantify_chamber(stamps, roi))
    if features in ("button", "all"):
        if roi.button_disk_masks is None:
            raise ValueError("RoiSet has no button masks")
        records.update(_quantify_button(stamps, roi))

    # Stamp metadata (match ChipImage.summarize / Stamp.summarize)
    xslice = [
        (int(roi.row0[i]), int(roi.row1[i])) for i in range(roi.n)
    ]
    yslice = [
        (int(roi.col0[i]), int(roi.col1[i])) for i in range(roi.n)
    ]
    records["xslice"] = xslice
    records["yslice"] = yslice
    records["id"] = list(roi.ids)

    idx = pd.MultiIndex.from_arrays(
        [roi.indices[:, 0], roi.indices[:, 1]], names=["x", "y"]
    )
    df = pd.DataFrame(records, index=idx)
    return df.sort_index()


def render_summary(
    stamps: np.ndarray,
    roi: RoiSet,
    features: Optional[str] = None,
    metrics: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """Annotate stamps from RoiSet geometry and stitch into a summary image."""
    features = features or roi.features
    if stamps.shape[0] != roi.n:
        raise ValueError(f"stamp count {stamps.shape[0]} != roi.n {roi.n}")

    # Grid layout from index extent (1-based indices as in ChipImage)
    xs = roi.indices[:, 0]
    ys = roi.indices[:, 1]
    xdim = int(xs.max())
    ydim = int(ys.max())
    h, w = roi.stamp_shape
    # annotateStamp adds 1px border → (h+2, w+2)
    tile_h, tile_w = h + 2, w + 2
    tiles = np.empty((xdim, ydim), dtype=object)

    metrics_by_index = None
    if metrics is not None:
        metrics_by_index = metrics

    for i in range(roi.n):
        ix, iy = int(roi.indices[i, 0]) - 1, int(roi.indices[i, 1]) - 1
        circles: List = []
        val = ""

        if features in ("chamber", "all") and roi.chamber_blank is not None:
            if not roi.chamber_blank[i]:
                cx, cy = roi.chamber_centers[i]
                rad = int(roi.chamber_radii[i])
                circles.append([rad, (int(cx), int(cy))])

        if features in ("button", "all") and roi.button_blank is not None:
            if not roi.button_blank[i]:
                cx, cy = roi.button_centers[i]
                c = (int(cx), int(cy))
                circles.append([int(roi.button_disk_radii[i]), c])
                circles.append([int(roi.button_annulus_radii[i, 1]), c])
                if metrics_by_index is not None:
                    try:
                        row = metrics_by_index.loc[(roi.indices[i, 0], roi.indices[i, 1])]
                        bg = row.get("summed_button_BGsub", np.nan)
                        ann = row.get("summed_button_annulus_normed", np.nan)
                        if pd.notna(bg) and pd.notna(ann):
                            val = "{}, {}".format(int(bg), int(ann))
                    except KeyError:
                        pass

        index = "{}.{} | {}".format(roi.indices[i, 0], roi.indices[i, 1], roi.ids[i])
        tiles[ix, iy] = annotateStamp(stamps[i], circles, index, val)

    # Fill any missing grid slots with zeros (should not happen for full chips)
    for ix in range(xdim):
        for iy in range(ydim):
            if tiles[ix, iy] is None:
                tiles[ix, iy] = np.zeros((tile_h, tile_w), dtype=stamps.dtype)

    arr = np.empty((xdim, ydim, tile_h, tile_w), dtype=stamps.dtype)
    for ix in range(xdim):
        for iy in range(ydim):
            arr[ix, iy] = tiles[ix, iy]
    return ChipImage.stitch2D(arr)


def _quantify_image_worker(payload):
    """Picklable worker: quantify one image path with a shared RoiSet."""
    path, roi, features = payload
    return quantify(path, roi, features=features)


def quantify_images(
    paths: Sequence[PathLike],
    roi: RoiSet,
    features: Optional[str] = None,
    n_jobs: int = 1,
) -> List[pd.DataFrame]:
    """Quantify many images with one RoiSet; optional process pool over images."""
    features = features or roi.features
    workers = _resolve_n_jobs(n_jobs)
    paths = list(paths)
    if workers <= 1 or len(paths) <= 1:
        return [quantify(p, roi, features=features) for p in paths]

    payloads = [(p, roi, features) for p in paths]
    chunksize = max(1, len(payloads) // (workers * 4))
    with _stamp_process_pool(workers) as executor:
        return list(executor.map(_quantify_image_worker, payloads, chunksize=chunksize))
