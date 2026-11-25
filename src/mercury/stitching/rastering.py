import numpy as np
import pathlib
import logging
import cv2
from skimage import io
from abc import abstractmethod, ABC
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from basicpy import BaSiC

import os
from typing import Iterable, Iterator, List, Optional, Tuple, Union


def ff_subtract(i, ffi, ff_bval, ff_scale):
    """Applies flat-field correction to an input image array.

    Args:
        i (np.ndarray): Target image array.
        ffi (np.ndarray): Flat-field reference image array.
        ff_bval (float or int): Dark field background value.
        ff_scale (float or int): Scaling factor for normalization.

    Returns:
        np.ndarray: Flat-field corrected image array scaled to uint16.
    """
    ff_result = np.subtract(i, ff_bval) / np.subtract(ffi, ff_bval) * ff_scale
    result = np.clip(ff_result, 0, 65535).astype("uint16")
    return result


def rotate_image(img, rotation_val) -> np.array:
    """Rotates an image array by a specified angle in degrees via ``cv2.warpAffine``.

    Matches the historical skimage convention: positive angles are
    counter-clockwise, output shape is unchanged, and out-of-bounds pixels
    are filled with 0. The rotation center uses skimage's ``(cols-1)/2``,
    ``(rows-1)/2`` convention for closer geometric agreement.

    Args:
        img (np.ndarray): Input image array.
        rotation_val (float): Angle of rotation in degrees.

    Returns:
        np.ndarray: Rotated uint16 image array.
    """
    if not rotation_val:
        return img.astype("uint16", copy=False)

    rows, cols = img.shape[:2]
    # skimage.transform.rotate center convention
    center = ((cols - 1) / 2.0, (rows - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, rotation_val, 1.0)
    # Float warp + truncating cast mirrors prior skimage .astype("uint16") path
    rotated = cv2.warpAffine(
        img.astype(np.float32, copy=False),
        matrix,
        (cols, rows),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return rotated.astype("uint16")


def iter_tiles(
    refs: Iterable[Union[str, pathlib.Path]], rotation: float = 0.0
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (index, tile) one at a time: read, optional rotate, then drop after yield."""
    for i, ref in enumerate(refs):
        img = io.imread(ref)
        if rotation:
            img = rotate_image(img, rotation)
        yield i, img


def load_tiles(
    refs: Iterable[Union[str, pathlib.Path]], rotation: float = 0.0
) -> List[np.ndarray]:
    """Load (and optionally rotate) all tiles into a list."""
    return [tile for _, tile in iter_tiles(refs, rotation)]


def cut_margin_retained(size: int, overlap: float) -> Tuple[int, int, slice]:
    """Return (margin, retained, border_slice) for cut-stitch trimming."""
    if overlap < 0 or overlap >= 1:
        raise ValueError("Overlap must be ≥ 0 and < 1")
    margin = int(size * overlap / 2)
    retained = size - 2 * margin
    if margin == 0:
        border = slice(0, None)
    else:
        border = slice(margin, -margin)
    return margin, retained, border


def grid_index(
    i: int, dims: Tuple[int, int], acqui_ori: Tuple[bool, bool]
) -> Tuple[int, int]:
    """Map flat tile index to (col, row) after acquisition-origin flips.

    Matches historical reshape+(flip axis 0/1)+concatenate geometry:
    dims=(cols, rows); flat order is c * rows + r.
    """
    cols, rows = int(dims[0]), int(dims[1])
    c = i // rows
    r = i % rows
    if acqui_ori[0]:
        c = cols - 1 - c
    if acqui_ori[1]:
        r = rows - 1 - r
    return c, r


def paste_cut_tile(
    canvas: np.ndarray,
    tile: np.ndarray,
    c: int,
    r: int,
    retained: int,
    border: slice,
) -> None:
    """Trim tile borders and paste into the cut-stitch canvas in place.

    Canvas layout matches historical concatenate: axis0 stacks rows, axis1 stacks cols,
    so tile (c, r) lands at canvas[r*retained:(r+1)*retained, c*retained:(c+1)*retained].
    """
    y0 = r * retained
    x0 = c * retained
    canvas[y0 : y0 + retained, x0 : x0 + retained] = tile[border, border]


def assemble_cut(
    tile_iter: Iterable[Tuple[int, np.ndarray]],
    dims: Tuple[int, int],
    acqui_ori: Tuple[bool, bool],
    overlap: float,
    size: int,
    dtype=None,
) -> np.ndarray:
    """Allocate a canvas and paste trimmed tiles from an (index, array) iterator."""
    _, retained, border = cut_margin_retained(size, overlap)
    cols, rows = int(dims[0]), int(dims[1])
    canvas = None

    for i, tile in tile_iter:
        if canvas is None:
            out_dtype = dtype if dtype is not None else tile.dtype
            # (rows*retained, cols*retained) matches np.concatenate cut geometry
            canvas = np.empty((rows * retained, cols * retained), dtype=out_dtype)
        c, r = grid_index(i, dims, acqui_ori)
        paste_cut_tile(canvas, tile, c, r, retained, border)

    if canvas is None:
        raise ValueError("assemble_cut received no tiles")
    return canvas


# ---------------------------------------------------------------------------
# Seam residuals: quantify stitch "jaggedness" in pixels.
#
# Two tiles that are neighbours on the stitched canvas image the same strip of
# the device. Registering those strips (on raw, unrotated tiles) gives the
# physical stage step between the tiles in camera coordinates:
#     along  = component parallel to the seam   (≈ D·sin θ, θ = camera/stage angle)
#     across = component perpendicular to it    (relative to the nominal step)
# where D is the true tile-to-tile step. A cut stitch with rotation θ_c and
# overlap f_c then leaves, at each seam,
#     along residual  = D·sin(θ_seam - θ_c)
#     across residual = D·cos(θ_seam - θ_c) - pitch(f_c)
# with pitch(f) = size - 2·int(size·f/2), the cut-stitch tile pitch
# (see cut_margin_retained). Residuals of 0 px mean a seamless stitch.
# Measurement is done once per raster; scoring candidate parameters is cheap.
# ---------------------------------------------------------------------------

SEAM_COLUMNS = [
    "direction", "i_a", "i_b", "c_a", "r_a", "c_b", "r_b", "tile_px",
    "contrast", "shift_along", "shift_across", "step_px", "theta_deg",
    "ncc", "inlier",
]


def seam_pairs(
    dims: Tuple[int, int], acqui_ori: Tuple[bool, bool]
) -> List[Tuple[str, int, int, Tuple[int, int], Tuple[int, int]]]:
    """Neighbouring tile pairs as placed on the stitched canvas.

    Uses grid_index (the same mapping assemble_cut uses), so the seams measured
    here are exactly the seams in the stitched image, for any acqui_ori.

    Returns:
        List of (direction, i_a, i_b, (c_a, r_a), (c_b, r_b)): tile A is left of
        (direction 'H') or above (direction 'V') tile B on the canvas, and i_*
        index into the flat image_refs list.
    """
    cols, rows = int(dims[0]), int(dims[1])
    pos_to_i = {grid_index(i, dims, acqui_ori): i for i in range(cols * rows)}
    pairs = []
    for c in range(cols):
        for r in range(rows):
            if c + 1 < cols:
                pairs.append(("H", pos_to_i[(c, r)], pos_to_i[(c + 1, r)], (c, r), (c + 1, r)))
            if r + 1 < rows:
                pairs.append(("V", pos_to_i[(c, r)], pos_to_i[(c, r + 1)], (c, r), (c, r + 1)))
    return pairs


def _ncc_after_shift(ref: np.ndarray, mov: np.ndarray, shift) -> float:
    """Pearson correlation of ref and mov over their overlap, with ref(p) ≈ mov(p - shift)."""
    sy, sx = int(round(shift[0])), int(round(shift[1]))
    h, w = ref.shape
    a = ref[max(0, sy):min(h, h + sy), max(0, sx):min(w, w + sx)].ravel()
    b = mov[max(0, -sy):min(h, h - sy), max(0, -sx):min(w, w - sx)].ravel()
    if a.size < 16 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def measure_seams(
    image_refs: List[Union[str, pathlib.Path]],
    dims: Tuple[int, int],
    acqui_ori: Tuple[bool, bool],
    overlap0: float = 0.1,
    n_seams: int = 20,
    crop_frac: float = 0.15,
    upsample_factor: int = 10,
    max_shift: float = 150.0,
    outlier_px: float = 3.0,
    min_ncc: float = 0.5,
):
    """Register the highest-contrast seams of one raster (raw, unrotated tiles).

    Args:
        image_refs: Tile paths in raster order (flat index c * rows + r, i.e.
            imaging.csv sorted by raster_col_index, raster_row_index).
        dims: (cols, rows) of the raster.
        acqui_ori: Acquisition origin flags, as passed to the stitcher.
        overlap0: Nominal overlap fraction; sets the width of the compared strips.
        n_seams: Number of highest-contrast seams to register.
        crop_frac: Fraction of the seam length trimmed from each end (avoids corners).
        upsample_factor: Sub-pixel precision of phase cross-correlation (1/upsample px).
        max_shift: Shifts larger than this (px) are rejected outright.
        outlier_px: Minimum tolerance (px) for the per-direction consensus test.
        min_ncc: Minimum correlation of the aligned strips for a seam to count. Real
            overlapping strips score ~0.8-0.98 on 20260730 data; unrelated or
            featureless strips score near 0.

    Returns:
        pd.DataFrame, one row per registered seam (columns SEAM_COLUMNS).
        shift_along / shift_across: offset of tile B's features relative to A,
        parallel / perpendicular to the seam. step_px and theta_deg: implied
        stage step and camera/stage angle (degrees, in rotate_image's convention).
        inlier: aligned strips correlate (ncc >= min_ncc) and the seam agrees with
        the per-direction consensus.
    """
    import pandas as pd
    from skimage.registration import phase_cross_correlation

    cols, rows = int(dims[0]), int(dims[1])
    if len(image_refs) != cols * rows:
        raise ValueError(f"Expected {cols * rows} tiles for dims {dims}, got {len(image_refs)}")
    if not 0 < overlap0 < 1:
        raise ValueError(f"overlap0 must be in (0, 1), got {overlap0}")

    # Load each tile once; keep only its four edge strips.
    strips = {}
    for i, ref in enumerate(image_refs):
        t = io.imread(ref).astype(np.float32)
        h, w = t.shape[:2]
        ov_x, ov_y = int(w * overlap0), int(h * overlap0)
        cm_y, cm_x = int(h * crop_frac), int(w * crop_frac)
        strips[i] = {
            "right": t[cm_y:h - cm_y, w - ov_x:],
            "left": t[cm_y:h - cm_y, :ov_x],
            "bottom": t[h - ov_y:, cm_x:w - cm_x],
            "top": t[:ov_y, cm_x:w - cm_x],
            "shape": (h, w),
        }

    candidates = []
    for direction, i_a, i_b, pa, pb in seam_pairs(dims, acqui_ori):
        if direction == "H":
            a, b = strips[i_a]["right"], strips[i_b]["left"]
            tile_px = strips[i_a]["shape"][1]
        else:
            a, b = strips[i_a]["bottom"], strips[i_b]["top"]
            tile_px = strips[i_a]["shape"][0]
        contrast = 0.5 * (float(a.std()) + float(b.std()))
        candidates.append((contrast, direction, i_a, i_b, pa, pb, a, b, tile_px))

    candidates.sort(key=lambda x: x[0], reverse=True)
    records = []
    for contrast, direction, i_a, i_b, pa, pb, a, b, tile_px in candidates[:n_seams]:
        a0, b0 = a - a.mean(), b - b.mean()
        # normalization=None (plain cross-correlation): phase normalization locks
        # onto the strip edges (zero shift) when features are smooth.
        shift, _, _ = phase_cross_correlation(
            a0, b0, upsample_factor=upsample_factor, normalization=None
        )
        dy, dx = float(shift[0]), float(shift[1])
        # Reference A, moving B: A(p) ≈ B(p - shift).
        if direction == "H":
            along, across = dy, dx
        else:
            along, across = -dx, dy
        nominal = tile_px - int(tile_px * overlap0)
        ncc = _ncc_after_shift(a0, b0, shift)
        records.append({
            "direction": direction, "i_a": i_a, "i_b": i_b,
            "c_a": pa[0], "r_a": pa[1], "c_b": pb[0], "r_b": pb[1],
            "tile_px": tile_px, "contrast": contrast,
            "shift_along": along, "shift_across": across,
            "step_px": float(np.hypot(nominal + across, along)),
            "theta_deg": float(np.degrees(np.arctan2(along, nominal + across))),
            "ncc": ncc,
            "inlier": bool(abs(along) < max_shift and abs(across) < max_shift
                           and np.isfinite(ncc) and ncc >= min_ncc),
        })

    seams = pd.DataFrame(records, columns=SEAM_COLUMNS)
    if seams.empty:
        return seams
    seams["inlier"] = seams["inlier"].astype(bool)

    # Consensus per direction: periodic device features can alias a strip onto
    # the wrong chamber, and dim seams register on noise. Both land far from
    # the per-direction median and are flagged as outliers.
    for _, grp in seams[seams["inlier"]].groupby("direction"):
        dtheta = np.radians(grp["theta_deg"] - grp["theta_deg"].median())
        along_res = grp["step_px"] * np.sin(dtheta)
        across = grp["step_px"] * np.cos(dtheta)
        for res in (along_res, across - across.median()):
            dev = np.abs(res - np.median(res))
            tol = max(outlier_px, 3.0 * 1.4826 * np.median(dev))
            seams.loc[grp.index[dev > tol], "inlier"] = False
    return seams


def cut_pitch(tile_px: int, overlap: float) -> int:
    """Tile pitch (px) on the cut-stitch canvas for a given overlap."""
    return cut_margin_retained(int(tile_px), overlap)[1]


def score_seams(seams, rotation: float, overlap: float) -> dict:
    """Residual seam offsets (px) left by cut-stitching with (rotation, overlap).

    Pure arithmetic on a measure_seams table (no image I/O); only inlier seams count.

    Returns:
        dict with n_seams, n_inliers; along_/across_ median_abs and p95_abs over all
        inlier seams; and signed medians per direction (along_H, across_H, along_V,
        across_V; NaN if a direction has no inliers). Positive across = gap
        (true step longer than the stitch pitch; overlap set too high).
    """
    out = {"n_seams": int(len(seams)),
           "n_inliers": int(seams["inlier"].sum()) if len(seams) else 0}
    keys = ["along_median_abs", "along_p95_abs", "across_median_abs", "across_p95_abs",
            "along_H", "across_H", "along_V", "across_V"]
    if out["n_inliers"] == 0:
        out.update({k: float("nan") for k in keys})
        return out
    s = seams[seams["inlier"]]
    dtheta = np.radians(s["theta_deg"] - rotation)
    along = (s["step_px"] * np.sin(dtheta)).to_numpy()
    pitch = np.array([cut_pitch(px, overlap) for px in s["tile_px"]])
    across = (s["step_px"] * np.cos(dtheta)).to_numpy() - pitch
    out["along_median_abs"] = float(np.median(np.abs(along)))
    out["along_p95_abs"] = float(np.percentile(np.abs(along), 95))
    out["across_median_abs"] = float(np.median(np.abs(across)))
    out["across_p95_abs"] = float(np.percentile(np.abs(across), 95))
    for d in ("H", "V"):
        m = (s["direction"] == d).to_numpy()
        out[f"along_{d}"] = float(np.median(along[m])) if m.any() else float("nan")
        out[f"across_{d}"] = float(np.median(across[m])) if m.any() else float("nan")
    return out


def estimate_from_seams(seams, min_inliers: int = 5) -> Tuple[float, float]:
    """Best-fit (rotation in degrees, overlap fraction) from inlier seams.

    rotation = median seam angle;
    overlap  = 1 - median(step·cos(θ_seam - rotation)) / tile_px.
    The cut stitch rounds its margin to whole pixels, so the achievable overlap
    is quantised in steps of 2/tile_px; score_seams accounts for that.

    Raises:
        ValueError: if fewer than min_inliers seams survived.
    """
    n = int(seams["inlier"].sum()) if len(seams) else 0
    if n < min_inliers:
        raise ValueError(f"Only {n} inlier seams (< {min_inliers}); raster lacks usable features.")
    s = seams[seams["inlier"]]
    rotation = float(np.median(s["theta_deg"]))
    proj = s["step_px"] * np.cos(np.radians(s["theta_deg"] - rotation))
    overlap = float(np.median(1.0 - proj / s["tile_px"]))
    return rotation, overlap


def find_imaging_csv(img_path: pathlib.Path) -> pathlib.Path:
    """Traverses parent directories to locate the imaging.csv manifest file.

    Args:
        img_path (pathlib.Path): Path to an image file or subfolder.

    Returns:
        pathlib.Path: Path to discovered imaging.csv file.

    Raises:
        FileNotFoundError: If imaging.csv is not found in parent directories.
    """
    for parent in img_path.resolve().parents:
        csv_path = parent / "imaging.csv"
        if csv_path.exists():
            return csv_path
    raise FileNotFoundError(f"imaging.csv not found in parents of {img_path}")


class RasterParams:
    """Configuration parameters describing a single multi-tile image acquisition raster.

    Args:
        overlap (float): Fractional overlap between adjacent tiles (e.g. 0.1).
        size (int): Tile width/height dimension in pixels.
        acqui_ori (tuple[int, int]): Acquisition origin coordinates.
        rotation (float): Pre-stitching rotation angle in degrees.
        dims (tuple): Grid dimensions (rows, cols).
        auto_ff (bool, optional): Flag to execute automatic flat-field correction. Default is True.
        ff_type (str, optional): Flat-field algorithm type ('BaSiC' or 'custom'). Default is 'BaSiC'.
        group_feature (int, optional): Feature value for grouping rasters. Default is 0.
    """
    def __init__(
        self,
        overlap: float,
        size: int,
        acqui_ori: Tuple[int, int],
        rotation: float,
        dims: tuple,
        auto_ff: bool = True,
        ff_type: str = "BaSiC",
        group_feature=0,
    ):
        """Initializes RasterParams instance."""
        # This can never be reached!
        # if self._root:
        #     self._parent = list(pathlib.Path(root).parents)[0]
        self._size = size
        self._overlap = overlap
        self._rotation = rotation
        self._acqui_ori = acqui_ori
        self._group_feature = group_feature
        self._auto_ff = auto_ff
        self._ff_type = ff_type

        self._exposure = None
        self._channel = None
        self._parent = None
        self._dims = dims
        self._root = None

    def update_root(self, new_root):
        self._root = new_root
        self._parent = os.path.dirname(new_root)

    @property
    def size(self):
        return self._size

    @property
    def overlap(self):
        return self._overlap

    @property
    def rotation(self):
        return self._rotation

    @property
    def acqui_ori(self):
        return self._acqui_ori

    @property
    def group_feature(self):
        return self._group_feature

    @property
    def auto_ff(self):
        return self._auto_ff

    @property
    def ff_type(self):
        return self._ff_type

    @property
    def exposure(self):
        return self._exposure

    @property
    def channel(self):
        return self._channel

    @property
    def parent(self):
        return self._parent

    @property
    def dims(self):
        return self._dims

    @property
    def root(self):
        return self._root

    @property
    def name(self):
        return f"{self.channel}_{self.exposure}"

    def update_channel(self, new_channel):
        self._channel = new_channel

    def update_exposure(self, new_exposure):
        self._exposure = new_exposure

    def update_dims(self, new_dims):
        self._dims = new_dims

    def update_group_feature(self, new_group_feature):
        self._group_feature = new_group_feature


class Raster(ABC):
    """Abstract base class representing a collection of tile image references for a single acquisition raster.

    Args:
        image_refs (list): Ordered list of tile image file paths.
        params (RasterParams): Associated raster parameters.
    """
    def __init__(self, image_refs: list, params: RasterParams):
        """Initializes Raster instance."""
        self._image_refs = image_refs
        self._params = params
        self._images = None

    @abstractmethod
    def fetch_images(self):
        raise NotImplementedError("Raster Subclass should implement this!")

    @property
    def params(self):
        return self._params

    def apply_ff(self):
        """
        Applies flat-field correction to fetched images

        Arguments:
            None

        Returns:
            None

        """
        raise NotImplementedError("Functionality not ready yet.")
        channel = self._params.channel
        exposure = self._params.exposure
        ff_image = StitchingSettings.ff_images[channel]
        ff_params = StitchingSettings.ff_params[channel][exposure]
        ff_bval = ff_params[0]
        ff_scale = ff_params[1]

        return [ff_subtract(i, ff_image, ff_bval, ff_scale) for i in self._images]

    def applyFF_BaSiC(self, plot: bool, images: Optional[List[np.ndarray]] = None):
        """
        Applies flat-field correction to fetched images using BaSiC. This technique simulates a FF and
        dark image based on common shared features across the full raster (e.g. 64 images).

        This version uses gaussian smoothing after fitting.

        Implemented in v2.1.0 by Peter Suzuki, Youngbin Lim

        Arguments:
            plot (bool): Whether to show diagnostic plots.
            images (list, optional): Explicit tile list; defaults to self._images.

        Returns:
            list[np.ndarray]: Flat-field corrected tile images.
        """
        imgarray = list(images) if images is not None else list(self._images)
        stacked = np.asarray(imgarray)

        # Initialize BaSiC and train
        basic = BaSiC(get_darkfield=False, smoothness_flatfield=1)
        basic.fit(stacked)

        # Smooth FF image and apply transform to each raw image
        ffImage = gaussian_filter(basic.flatfield, sigma=50)

        ffbval = 0
        ffscale = 1

        manual_smoothed = [ff_subtract(i, ffImage, ffbval, ffscale) for i in imgarray]

        if plot:
            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
            im = axes[0].imshow(basic.flatfield)
            fig.colorbar(im, ax=axes[0])
            axes[0].set_title("Flatfield")
            im = axes[1].imshow(ffImage)
            fig.colorbar(im, ax=axes[1])
            axes[1].set_title("Smoothed FF")
            axes[2].plot(basic.baseline)
            axes[2].set_xlabel("Frame")
            axes[2].set_ylabel("Baseline")
            fig.tight_layout()
            plt.show()

        return manual_smoothed

    def applyFF_BaSiC_masked(self):
        """
        Applies flat-field correction to fetched images using BaSiC. This technique simulates a FF and
        dark image based on common shared features across the full raster (e.g. 64 images).

        Current version fits a gaussian to the pixels and only uses pixels within 1 s.d. of the mean to
        calculate BaSiC FF image. (Using a mask to cull which pixels are seen by BaSiC during training).
        After fitting, images can be transformed using the FF image either with or w/out smoothing, with
        default setting using gaussian smoothing to eliminate small (wrong) features learned by BaSiC.

        Implemented in v2.1.0 by Peter Suzuki, Youngbin Lim

        Arguments:
            None

        Returns:
            None

        """
        
        raise NotImplementedError("Functionality not ready yet.")

        print("Running BaSiC for FF correction, masking outlier pixels...")

        # Load images
        imgarray = []
        for img in self._images:
            # print(img.shape)
            imgarray.append(img)
        images = np.asarray(imgarray)

        ## Flatten and fit gaussian to background pixels
        plt.figure(figsize=(6, 2))

        data = images.flatten()
        y, x, _ = plt.hist(data, 100, alpha=0.3, label="data")
        x = (x[1:] + x[:-1]) / 2  # for len(x)==len(y)

        def gauss(x, mu, sigma, A):
            return A * np.exp(-((x - mu) ** 2) / 2 / sigma**2)

        expected = (30000, 5000, 1e7)  # , 20000, 5000, 125)
        params, cov = curve_fit(gauss, x, y, expected)
        x_fit = np.linspace(x.min(), x.max(), 100)
        plt.plot(x_fit, gauss(x_fit, *params), color="red", lw=3, label="model")
        plt.legend()
        plt.title("Pixel intensity distribution")
        plt.show()

        # Set mask bounds to train only on background
        top = params[0] + abs(3 * params[1])  # mean + 3sd
        bottom = params[0] - abs(3 * params[1])
        mask = np.zeros(images.shape)
        mask = (images < top) & (
            images > bottom
        )  # threshold for defining bright things defined by mean + 3 s.d.
        print(params)
        print(top)
        print(bottom)
        # masked = mask*images

        # Initialize BaSiC and train
        basic_mask = BaSiC(get_darkfield=False, smoothness_flatfield=1)
        basic_mask.fit(images, fitting_weight=mask)

        # Smooth FF image and apply transform to each raw image
        ffImage = gaussian_filter(basic_mask.flatfield, sigma=50)
        # StitchingSettings.ffImages[self.params.channel] = ffImage # save ffImage to settings

        ffbval = 0
        ffscale = 1

        manual_smoothed = [ff_subtract(i, ffImage, ffbval, ffscale) for i in imgarray]

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        im = axes[0].imshow(basic_mask.flatfield)
        fig.colorbar(im, ax=axes[0])
        axes[0].set_title("Flatfield")
        im = axes[1].imshow(ffImage)
        fig.colorbar(im, ax=axes[1])
        axes[1].set_title("Smoothed FF")
        axes[2].plot(basic_mask.baseline)
        axes[2].set_xlabel("Frame")
        axes[2].set_ylabel("Baseline")
        fig.tight_layout()
        plt.show()

        return manual_smoothed

    def stitch(self, method="cut", plot: bool = False):
        """
        Wrapper for image stitching method selection.
        TODO: Implement 'overlap' method

        Arguments:
            (str) method: stitch method ('cut' | 'overlap' | 'smart' | 'rescue')

        Returns:
            (np.ndarray) A stitched image array

        """
        if method == "cut":
            # cut_stitch streams tiles (or loads for BaSiC); do not prefetch all
            return self.cut_stitch(plot)
        self.fetch_images()
        if method == "smart":
            return self.smart_stitch()
        elif method == "overlap":
            return self.overlap_stitch()
        elif method == "rescue":
            return self.coordinate_stitch(plot)
        else:
            raise ValueError(
                'Invalid stitch method. Valid methods are "cut", "smart", "overlap" and "rescue"'
            )

    def smart_stitch(self):
        """
        Returns:
            (np.ndarray) A stitched image array
        """
        raise NotImplementedError("Functionality not ready yet.")

    def cut_stitch(self, plot: bool = False):
        """
        Stitches a raster via the 'cut' method. Trims borders according to image overlap and
        pastes into a preallocated canvas. If RasterParameters.auto_ff is True, loads all
        tiles for BaSiC flat-field correction, then pastes corrected tiles.

        When auto_ff is False, tiles are streamed one at a time (read → rotate → trim → paste → drop).

        Returns:
            (np.ndarray) A stitched image array
        """
        params = self.params
        # Validate overlap early
        cut_margin_retained(params.size, params.overlap)

        if params.auto_ff and params.ff_type == "BaSiC":
            tiles = load_tiles(self._image_refs, params.rotation)
            self._images = tiles
            corrected = self.applyFF_BaSiC(plot, images=tiles)
            self.ffCorrectedImages = corrected # TODO: figure out why images are being assigned to an attribute
            result = assemble_cut(
                enumerate(corrected),
                dims=params.dims,
                acqui_ori=params.acqui_ori,
                overlap=params.overlap,
                size=params.size,
            )
            self._images = None
            self.ffCorrectedImages = None
            return result

        result = assemble_cut(
            iter_tiles(self._image_refs, params.rotation),
            dims=params.dims,
            acqui_ori=params.acqui_ori,
            overlap=params.overlap,
            size=params.size,
        )
        self._images = None
        return result

    def _seams(self, num_boundaries: int = 20):
        """measure_seams on this raster's raw tiles, using its params."""
        p = self.params
        overlap0 = p.overlap if (p.overlap and 0.0 < p.overlap < 1.0) else 0.1
        return measure_seams(
            self._image_refs, p.dims, p.acqui_ori, overlap0=overlap0, n_seams=num_boundaries
        )

    def detect_overlap(self, num_boundaries: int = 20) -> float:
        """
        Estimates the tile overlap fraction by registering neighbouring tiles'
        overlapping strips (see measure_seams / estimate_from_seams).

        Raises:
            ValueError: if too few seams could be registered.
        """
        n = int(self.params.dims[0]) * int(self.params.dims[1])
        if len(self._image_refs) != n:
            raise ValueError(f"Expected {n} tiles, got {len(self._image_refs)}")
        _, overlap = estimate_from_seams(self._seams(num_boundaries))
        return overlap

    def detect_rotation(self, search_range=None, num_boundaries=20) -> float:
        """
        Estimates the camera-to-stage rotation angle (degrees) by registering
        neighbouring raw tiles and measuring the offset along each seam
        (see measure_seams / estimate_from_seams).

        Neighbours are taken from grid_index, so acqui_ori is respected.

        Raises:
            ValueError: if too few seams could be registered, or the estimate
                falls outside search_range.
        """
        n = int(self.params.dims[0]) * int(self.params.dims[1])
        if len(self._image_refs) != n:
            return 0.0
        rotation, _ = estimate_from_seams(self._seams(num_boundaries or 20))
        if search_range is not None and not (search_range[0] <= rotation <= search_range[1]):
            raise ValueError(
                f"Detected rotation {rotation:.3f} deg outside search range {search_range}"
            )
        return rotation

    def overlap_stitch(self):
        """
        #TODO: re-implement overlap stitching method

        """
        raise NotImplementedError("Overlap Stitch not yet implemented")

    def coordinate_stitch(self, plot: bool = False):
        """
        Stitches a raster via stage coordinates from imaging.csv,
        calibrating the pixel size dynamically using a single high-contrast boundary.
        Used primarily for "rescuing" stitched images with poor overlap,
        do to settings or microscope stage errors.

        Returns:
            (np.ndarray) A stitched image array
        """
        import pandas as pd
        from skimage.registration import phase_cross_correlation

        imsize = self.params.size
        overlap = self.params.overlap
        expected_overlap = int(imsize * overlap)

        # Load images
        tiles_list = self._images
        if self.params.auto_ff and self.params.ff_type == "BaSiC":
            tiles_list = self.ffCorrectedImages = self.applyFF_BaSiC(plot)

        # Automatically locate imaging.csv
        # Pulling image.csv from parents is a little yeehaw. Adjust in the future?
        ref_path = pathlib.Path(self._image_refs[0])
        csv_path = find_imaging_csv(ref_path)
        csv_dir = csv_path.parent

        df = pd.read_csv(csv_path)

        # Map to absolute paths for robust matching
        df['abs_path'] = df['image_path'].apply(lambda p: os.path.abspath(csv_dir / p))
        ref_abs = [os.path.abspath(p) for p in self._image_refs]

        df_matched = df[df['abs_path'].isin(ref_abs)].copy()
        df_matched.set_index('abs_path', inplace=True)
        df_matched = df_matched.reindex(ref_abs).reset_index()

        cols_count = self.params.dims[0]
        rows_count = self.params.dims[1]

        # Reconstruct tiles dictionary and coordinates grid
        tiles = {}
        xs = np.zeros((cols_count, rows_count))
        ys = np.zeros((cols_count, rows_count))
        
        for idx, row in df_matched.iterrows():
            c, r = int(row['raster_col_index']), int(row['raster_row_index'])
            tiles[(c, r)] = tiles_list[idx]
            xs[c, r] = row['x']
            ys[c, r] = row['y']

        # Single boundary calibration
        best_std = -1
        best_boundary = None  # Will store (type, c, r, strip1, strip2, dx_stage)
        
        # 1. Search horizontal boundaries
        for r in range(rows_count):
            for c in range(cols_count - 1):
                t1 = tiles[(c, r)].astype(float)
                t2 = tiles[(c+1, r)].astype(float)
                
                crop_margin = int(imsize * 0.15)
                strip1 = t1[crop_margin:-crop_margin, -expected_overlap:]
                strip2 = t2[crop_margin:-crop_margin, :expected_overlap]
                
                std_val = (np.std(strip1) + np.std(strip2)) / 2
                if std_val > best_std:
                    best_std = std_val
                    dx_stage = abs(xs[c+1, r] - xs[c, r])
                    best_boundary = ('H', c, r, strip1, strip2, dx_stage)
                    
        # 2. Search vertical boundaries
        for c in range(cols_count):
            for r in range(rows_count - 1):
                t1 = tiles[(c, r)].astype(float)
                t2 = tiles[(c, r+1)].astype(float)
                
                crop_margin = int(imsize * 0.15)
                strip1 = t1[-expected_overlap:, crop_margin:-crop_margin]
                strip2 = t2[:expected_overlap, crop_margin:-crop_margin]
                
                std_val = (np.std(strip1) + np.std(strip2)) / 2
                if std_val > best_std:
                    best_std = std_val
                    dy_stage = abs(ys[c, r+1] - ys[c, r])
                    best_boundary = ('V', c, r, strip1, strip2, dy_stage)

        pixel_size = None
        if best_boundary is not None and expected_overlap > 0:
            b_type, c, r, strip1, strip2, d_stage = best_boundary
            try:
                shift, error, diffphase = phase_cross_correlation(strip1, strip2, upsample_factor=1)
                if b_type == 'H':
                    d_pixel = (imsize - expected_overlap) + shift[1]
                else:
                    d_pixel = (imsize - expected_overlap) + shift[0]
                
                cand = d_stage / abs(d_pixel)
                if 3.0 < cand < 3.5:
                    pixel_size = cand
            except Exception:
                pass

        if pixel_size is None:
            raise ValueError(f"Could not dynamically assign pixel size for {self.params.name}. ")
        else:
            print(f"Dynamic pixel size assignment for {self.params.name}: {pixel_size}")

        # Determine signs based on acquisition origin
        sign_x = -1 if self.params.acqui_ori[0] else 1
        sign_y = -1 if self.params.acqui_ori[1] else 1

        # Calculate pixel coordinates
        u = sign_x * xs / pixel_size
        v = sign_y * ys / pixel_size

        u_min = np.min(u)
        v_min = np.min(v)

        u_offsets = u - u_min
        v_offsets = v - v_min

        canvas_w = int(np.round(np.max(u_offsets))) + imsize
        canvas_h = int(np.round(np.max(v_offsets))) + imsize

        canvas = np.zeros((canvas_h, canvas_w), dtype=tiles_list[0].dtype)

        # Initialize crop arrays
        trim_left = np.zeros((cols_count, rows_count), dtype=int)
        trim_right = np.zeros((cols_count, rows_count), dtype=int)
        trim_top = np.zeros((cols_count, rows_count), dtype=int)
        trim_bottom = np.zeros((cols_count, rows_count), dtype=int)

        # Calculate horizontal trims
        for r in range(rows_count):
            for c in range(cols_count - 1):
                ol = imsize - abs(u_offsets[c+1, r] - u_offsets[c, r])
                if ol > 0:
                    trim_amount = max(0, int(np.round(ol / 2)) - 1)
                    if u_offsets[c, r] < u_offsets[c+1, r]:
                        trim_right[c, r] = trim_amount
                        trim_left[c+1, r] = trim_amount
                    else:
                        trim_left[c, r] = trim_amount
                        trim_right[c+1, r] = trim_amount

        # Calculate vertical trims
        for c in range(cols_count):
            for r in range(rows_count - 1):
                ol = imsize - abs(v_offsets[c, r+1] - v_offsets[c, r])
                if ol > 0:
                    trim_amount = max(0, int(np.round(ol / 2)) - 1)
                    if v_offsets[c, r] < v_offsets[c, r+1]:
                        trim_bottom[c, r] = trim_amount
                        trim_top[c, r+1] = trim_amount
                    else:
                        trim_top[c, r] = trim_amount
                        trim_bottom[c, r+1] = trim_amount

        # Paste tiles
        for c in range(cols_count):
            for r in range(rows_count):
                u0 = int(np.round(u_offsets[c, r]))
                v0 = int(np.round(v_offsets[c, r]))

                t_l = max(0, min(imsize // 2, trim_left[c, r]))
                t_r = max(0, min(imsize // 2, trim_right[c, r]))
                t_t = max(0, min(imsize // 2, trim_top[c, r]))
                t_b = max(0, min(imsize // 2, trim_bottom[c, r]))

                tile = tiles[(c, r)]
                cropped = tile[t_t : imsize - t_b, t_l : imsize - t_r]

                canvas[v0 + t_t : v0 + imsize - t_b, u0 + t_l : u0 + imsize - t_r] = cropped

        return canvas

    def export_stitch(
        self, method="cut", out_path_name="StitchedImages", manual_target=None
    ):
        """
        Perform stitching and export raster.

        Arguments:
            (str) method: stitch method ('cut' | 'overlap')
            (str) out_path_name: Name of folder to house stitched raster. Typically 'StitchedImages'

        Returns:
            None

        """
        stitchedRaster = self.stitch(method=method)

        features = [
            self._params.exposure,
            self._params.channel,
            self._params.group_feature,
        ]
        rasterName = "StitchedImg_{}_{}_{}.tif".format(*features)
        if manual_target:
            stitchDir = pathlib.Path(manual_target)
        else:
            stitchDir = pathlib.Path(os.path.join(self._params.parent, out_path_name))
        stitchDir.mkdir(exist_ok=True)
        outDir = os.path.join(stitchDir, rasterName)
        io.imsave(outDir, stitchedRaster, check_contrast=False)#, plugin="tifffile", check_contrast=False)
        logging.debug("Stitching Complete")

    def __lt__(self, other):
        selfstem = pathlib.Path(self.image_refs[0]).stem
        otherstem = pathlib.Path(other.image_refs[0]).stem
        return selfstem < otherstem


# TODO: deprecate in future versions if unused
class FlatRaster(Raster):
    """Raster subclass for handling flat unstacked single-file tile images.

    Args:
        image_refs (list): List of tile image file paths.
        params (RasterParams): Associated raster parameters.
    """
    def __init__(self, image_refs, params):
        """Initializes FlatRaster instance."""
        super().__init__(image_refs, params)

    def fetch_images(self):
        """
        Fetches (loads into memory) and rotates images (if indicated by raster parameters).

        Thin wrapper around load_tiles for callers that need random access (detection, rescue).
        """
        self._images = load_tiles(self._image_refs, self._params.rotation)

