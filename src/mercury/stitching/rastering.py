import numpy as np
import pathlib
import logging
from skimage import io, transform
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
    """Rotates an image array by a specified angle in degrees.

    Args:
        img (np.ndarray): Input image array.
        rotation_val (float): Angle of rotation in degrees.

    Returns:
        np.ndarray: Rotated uint16 image array.
    """
    rotation_params = {"resize": False, "clip": True, "preserve_range": True}
    return transform.rotate(img, rotation_val, **rotation_params).astype("uint16")


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

    def detect_overlap(self) -> float:
        """
        Dynamically calculates the optimal overlap fraction using phase cross-correlation
        across high-contrast boundaries of the raster images.
        """
        from skimage.registration import phase_cross_correlation
        
        imsize = self.params.size
        expected_overlap = int(imsize * 0.1)
        if expected_overlap <= 0:
            return 0.1
            
        cols_count = self.params.dims[0]
        rows_count = self.params.dims[1]
        
        if len(self._images) != cols_count * rows_count:
            return 0.1
            
        tiles = {}
        for c in range(cols_count):
            for r in range(rows_count):
                tiles[(c, r)] = self._images[c * rows_count + r]
                
        boundaries = []
        
        # 1. Search horizontal boundaries
        for r in range(rows_count):
            for c in range(cols_count - 1):
                t1 = tiles[(c, r)].astype(float)
                t2 = tiles[(c+1, r)].astype(float)
                
                crop_margin = int(imsize * 0.15)
                strip1 = t1[crop_margin:-crop_margin, -expected_overlap:]
                strip2 = t2[crop_margin:-crop_margin, :expected_overlap]
                
                std_val = (np.std(strip1) + np.std(strip2)) / 2
                boundaries.append((std_val, strip1, strip2, 'H'))
                
        # 2. Search vertical boundaries
        for c in range(cols_count):
            for r in range(rows_count - 1):
                t1 = tiles[(c, r)].astype(float)
                t2 = tiles[(c, r+1)].astype(float)
                
                crop_margin = int(imsize * 0.15)
                strip1 = t1[-expected_overlap:, crop_margin:-crop_margin]
                strip2 = t2[:expected_overlap, crop_margin:-crop_margin]
                
                std_val = (np.std(strip1) + np.std(strip2)) / 2
                boundaries.append((std_val, strip1, strip2, 'V'))
                
        if not boundaries:
            return 0.1
            
        # Sort by standard deviation (highest feature contrast first)
        boundaries.sort(key=lambda x: x[0], reverse=True)
        
        calculated_overlaps = []
        for std_val, strip1, strip2, b_type in boundaries[:5]:
            try:
                shift, error, diffphase = phase_cross_correlation(strip1, strip2, upsample_factor=1)
                if b_type == 'H':
                    shift_val = shift[1]
                else:
                    shift_val = shift[0]
                
                # Filter out shifts that are unreasonably large to prevent aligning unrelated periodic channels
                if abs(shift_val) < 50:
                    actual_overlap = expected_overlap - shift_val
                    overlap_frac = actual_overlap / imsize
                    calculated_overlaps.append(overlap_frac)
            except Exception:
                pass
                
        if not calculated_overlaps:
            # TODO: This should raise, instead of assuming 10%
            return 0.1
            
        return float(np.median(calculated_overlaps))

    def detect_rotation(self, search_range=None, num_boundaries=20) -> float:
        """
        Dynamically calculates the camera-to-stage rotation angle by registering
        unrotated neighboring tiles and measuring the diagonal drift (shear) 
        introduced by the stage translation relative to the camera sensor.
        """
        from skimage.registration import phase_cross_correlation
        
        imsize = self.params.size
        overlap = self.params.overlap
        expected_overlap = int(imsize * overlap)
        if expected_overlap <= 0:
            expected_overlap = int(imsize * 0.1) # fallback
            
        cols_count = self.params.dims[0]
        rows_count = self.params.dims[1]
        
        if len(self._image_refs) != cols_count * rows_count:
            return 0.0
            
        # Helper to load raw tiles on demand and cache them
        raw_cache = {}
        def get_raw_tile(c, r):
            coord = (c, r)
            if coord not in raw_cache:
                idx = c * rows_count + r
                raw_cache[coord] = io.imread(self._image_refs[idx]).astype(float)
            return raw_cache[coord]
            
        # Find horizontal and vertical boundaries and calculate contrast
        all_boundaries = []
        crop_margin = int(imsize * 0.15)
        
        # 1. Horizontal boundaries
        for r in range(rows_count):
            for c in range(cols_count - 1):
                try:
                    t1 = get_raw_tile(c, r)
                    t2 = get_raw_tile(c + 1, r)
                    
                    strip1 = t1[crop_margin:-crop_margin, -expected_overlap:]
                    strip2 = t2[crop_margin:-crop_margin, :expected_overlap]
                    std_val = (np.std(strip1) + np.std(strip2)) / 2
                    all_boundaries.append((std_val, c, r, c + 1, r, 'H'))
                except Exception:
                    pass
                    
        # 2. Vertical boundaries
        for c in range(cols_count):
            for r in range(rows_count - 1):
                try:
                    t1 = get_raw_tile(c, r)
                    t2 = get_raw_tile(c, r + 1)
                    
                    strip1 = t1[-expected_overlap:, crop_margin:-crop_margin]
                    strip2 = t2[:expected_overlap, crop_margin:-crop_margin]
                    std_val = (np.std(strip1) + np.std(strip2)) / 2
                    all_boundaries.append((std_val, c, r, c, r + 1, 'V'))
                except Exception:
                    pass
                    
        if not all_boundaries:
            print("Warning: No valid boundaries found for rotation detection. Defaulting to 0.0")
            return 0.0
            
        # Sort by contrast (standard deviation) descending and take the top N
        all_boundaries.sort(key=lambda x: x[0], reverse=True)
        
        n_boundaries = num_boundaries if num_boundaries is not None else 20
        selected_boundaries = all_boundaries[:min(n_boundaries, len(all_boundaries))]
        
        angles = []
        
        # For each selected boundary, evaluate the translation shift and compute angle
        for _, c1, r1, c2, r2, b_type in selected_boundaries:
            try:
                t1 = get_raw_tile(c1, r1)
                t2 = get_raw_tile(c2, r2)
                
                # Demean to improve cross correlation sensitivity to features
                t1_dm = t1 - t1.mean()
                t2_dm = t2 - t2.mean()
                
                if b_type == 'H':
                    strip1 = t1_dm[crop_margin:-crop_margin, -expected_overlap:]
                    strip2 = t2_dm[crop_margin:-crop_margin, :expected_overlap]
                    
                    shift, error, diffphase = phase_cross_correlation(strip1, strip2, upsample_factor=10)
                    dy, dx = shift[0], shift[1]
                    
                    # This should discard found deviations that are greater than 1 row or column over.
                    if abs(dx) < 150 and abs(dy) < 150:
                        Dx = imsize - expected_overlap
                        theta_val = np.degrees(np.arctan2(dy, Dx + dx))
                        angles.append(theta_val)
                else:
                    strip1 = t1_dm[-expected_overlap:, crop_margin:-crop_margin]
                    strip2 = t2_dm[:expected_overlap, crop_margin:-crop_margin]
                    
                    shift, error, diffphase = phase_cross_correlation(strip1, strip2, upsample_factor=10)
                    dy, dx = shift[0], shift[1]
                    
                    # same here, discarding too-large deviations. We want the nearest matching shift.
                    if abs(dx) < 150 and abs(dy) < 150:
                        Dy = imsize - expected_overlap
                        theta_val = np.degrees(np.arctan2(-dx, Dy + dy))
                        angles.append(theta_val)
                        
            except Exception:
                pass
                
        if not angles:
            raise ValueError("Warning: No valid rotation angles calculated.")
            #return 0.0
            
        return float(np.median(angles))

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

