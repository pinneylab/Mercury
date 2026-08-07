# -*- coding: utf-8 -*-

"""Top-level package for processing."""

from mercury.processing import chip, experiment, roi

__author__ = """Daniel Mokhtari"""
__email__ = ""
__version__ = "0.1.0"

import warnings

import skimage
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
from typing import Union, List, Tuple


class Processor:
    
    def __init__(self, experiment: experiment.Experiment, image_data: pd.DataFrame, features: str):
        self.experiment = experiment

        self.image_data = image_data
        self.image_data['corners'] = [None] * len(self.image_data)
        self.image_data['reference_image'] = [None] * len(self.image_data)

        self.features = features
        assert self.features in ('button', 'chamber', 'all'), "features argument must be 'button', 'chamber', or 'all'."

        self.reference_images = {dname: None for dname in self.experiment.devices}
        self.reference_rois = {dname: None for dname in self.experiment.devices}

        self.summary_image_dir = self.experiment.root / 'summary_images'

    def _update_reference_image(self, dname: str, chip_image: chip.ChipImage):
        self.reference_images[dname] = chip_image

    def _update_reference_roi(self, dname: str, chip_image: chip.ChipImage):
        self.reference_rois[dname] = roi.RoiSet.from_chip(chip_image, self.features)

    def _process_features(
        self,
        chip_image: chip.ChipImage,
        coerce_chamber_center: bool = False,
        n_jobs: int = -1,
    ):
        """Extract feature processing logic into a reusable method."""
        if self.features == 'chamber':
            chip_image.findChambers(coerce_center=coerce_chamber_center, n_jobs=n_jobs)
        elif self.features == 'button':
            chip_image.findButtons(n_jobs=n_jobs)
        elif self.features == 'all':
            chip_image.findChambers(coerce_center=coerce_chamber_center, n_jobs=n_jobs)
            chip_image.findButtons(n_jobs=n_jobs)

    def _summary_outpath(self, image: Union[Path, str]) -> Path:
        image = Path(image) if not isinstance(image, Path) else image
        outpath = self.summary_image_dir / image.relative_to(self.experiment.root)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        return outpath

    def _imsave_summary(
        self,
        summary_image,
        outpath: Path,
        *,
        as_ubyte: bool = False,
        compress: bool = False,
    ):
        if as_ubyte:
            summary_image = skimage.img_as_ubyte(summary_image)
        save_kwargs = {'check_contrast': False}
        if compress and outpath.suffix.lower() in {'.tif', '.tiff'}:
            save_kwargs['compression'] = 'zlib'
        skimage.io.imsave(outpath, summary_image, **save_kwargs)

    def _save_summary_image(
        self,
        chip_image: chip.ChipImage,
        image: Union[Path, str],
        *,
        as_ubyte: bool = False,
        compress: bool = False,
        roi_set: roi.RoiSet = None,
        stamps=None,
        metrics: pd.DataFrame = None,
    ):
        """Save annotated summary image for debugging/review.

        Prefers RoiSet + render_summary when ``roi_set`` (and stamps) are provided.
        """
        outpath = self._summary_outpath(image)

        if roi_set is not None:
            if stamps is None:
                stamps = roi.stamps_from_chip(chip_image)
            summary_image = roi.render_summary(
                stamps, roi_set, features=self.features, metrics=metrics
            )
        else:
            warnings.warn(
                "Saving summary via ChipImage.summary_image is deprecated; "
                "prefer RoiSet + render_summary.",
                DeprecationWarning,
                stacklevel=2,
            )
            if self.features == 'all':
                # Legacy summary_image does not support 'all'; render chamber overlay only
                summary_image = chip_image.summary_image(stamptype='chamber')
            else:
                summary_image = chip_image.summary_image(stamptype=self.features)

        self._imsave_summary(
            summary_image, outpath, as_ubyte=as_ubyte, compress=compress
        )

    def set_corners(self, dname: str, corners: List[Tuple]):
        assert dname in self.experiment.devices, '{} not found in experiment.'.format(dname)
        assert len(corners) == 4
        for item in corners:
            assert isinstance(item, tuple)
            assert len(item) == 2
            x, y = item
            assert isinstance(x, int)
            assert isinstance(y, int)

        device_mask = self.image_data['dname'] == dname
        for idx in self.image_data[device_mask].index:
            self.image_data.at[idx, 'corners'] = corners

    def set_reference(
        self, 
        image: Union[Path, str], 
        save_summary_images: bool = True,
        coerce_chamber_center: bool = False,
        as_ubyte: bool = False,
        compress: bool = False,
        n_jobs: int = -1,
    ):
        """Set reference image for a device."""

        image = Path(image) if not isinstance(image, Path) else image

        # Get device number and corners via metadata lookup 
        data = self.image_data[self.image_data['image_path'] == image]
        assert len(data) > 0, 'Image not found!'
        assert len(data) == 1, 'Duplicate images found!'
        dname = data.iloc[0]['dname']
        corners = data.iloc[0]['corners']

        # Set reference_image column
        device_mask = self.image_data['dname'] == dname
        self.image_data.loc[device_mask, 'reference_image'] = image

        # Create and process chip image
        chip_image = chip.ChipImage(self.experiment.devices[dname], image, corners)
        chip_image.stamp()
        self._process_features(chip_image, coerce_chamber_center, n_jobs=n_jobs)
        
        self._update_reference_image(dname, chip_image)
        self._update_reference_roi(dname, chip_image)

        if save_summary_images:
            roi_set = self.reference_rois[dname]
            stamps = roi.stamps_from_chip(chip_image)
            metrics = roi.quantify(stamps, roi_set, features=self.features)
            self._save_summary_image(
                chip_image,
                image,
                as_ubyte=as_ubyte,
                compress=compress,
                roi_set=roi_set,
                stamps=stamps,
                metrics=metrics,
            )

    def process(
        self, 
        *, 
        use_reference: bool = False, 
        save_summary_images: bool = True,
        coerce_chamber_center: bool = False,
        as_ubyte: bool = False,
        compress: bool = False,
        n_jobs: int = -1,
        **kwargs
    ):
        """High-level dispatcher that handles the mutual exclusivity logic."""
        
        if use_reference:
            return self._process_from_reference(
                save_summary_images=save_summary_images,
                as_ubyte=as_ubyte,
                compress=compress,
                n_jobs=n_jobs,
                **kwargs,
            )
        
        else:

            no_corners_mask = self.image_data['corners'].isna()
            cornerless_devices = set(self.image_data['dname'][no_corners_mask].to_list())
            assert len(cornerless_devices) == 0, 'ERROR: the following devices have no corners set: ' + ', '.join(cornerless_devices)

            return self._process_manually(
                coerce_chamber_center=coerce_chamber_center,
                save_summary_images=save_summary_images,
                as_ubyte=as_ubyte,
                compress=compress,
                n_jobs=n_jobs,
                **kwargs,
            )
        
    def _process_manually(
        self, 
        save_summary_images: bool = False,
        coerce_chamber_center: bool = False,
        as_ubyte: bool = False,
        compress: bool = False,
        n_jobs: int = -1,
    ):
        """Process images with manually provided corners (find per image, RoiSet quantify)."""
        data = []
        for i in tqdm(range(len(self.image_data)), desc='Processing images', leave=False):

            dname, image, c = self.image_data[['dname', 'image_path', 'corners']].iloc[i]
            chip_image = chip.ChipImage(self.experiment.devices[dname], image, c)
            chip_image.stamp()
            self._process_features(chip_image, coerce_chamber_center, n_jobs=n_jobs)

            roi_set = roi.RoiSet.from_chip(chip_image, self.features)
            stamps = roi.stamps_from_chip(chip_image)
            processed = roi.quantify(stamps, roi_set, features=self.features)
            processed_data = processed.reset_index()
            metadata = pd.DataFrame([self.image_data.iloc[i]] * len(processed_data)).reset_index(drop=True)
            merged = pd.concat([metadata, processed_data], axis=1)
            data.append(merged)
            
            if save_summary_images:
                self._save_summary_image(
                    chip_image,
                    image,
                    as_ubyte=as_ubyte,
                    compress=compress,
                    roi_set=roi_set,
                    stamps=stamps,
                    metrics=processed,
                )
            del stamps
        
        return pd.concat(data, ignore_index=False)

    def _process_from_reference(
        self, 
        save_summary_images: bool = True,
        as_ubyte: bool = False,
        compress: bool = False,
        n_jobs: int = -1,
    ):
        """Process images by mapping RoiSet geometry from reference images.

        When n_jobs > 1 and save_summary_images is False, quantifies images in a
        process pool. Summary rendering stays serial (needs stamp tensors).
        """

        # Fast parallel path when not saving summaries
        if (
            not save_summary_images
            and roi._resolve_n_jobs(n_jobs) > 1
            and len(self.image_data) > 1
        ):
            return self._process_from_reference_parallel(
                as_ubyte=as_ubyte,
                compress=compress,
                n_jobs=n_jobs,
            )

        data = []
        for i in tqdm(range(len(self.image_data)), desc='Processing images', leave=False):

            dname, image = self.image_data[['dname', 'image_path']].iloc[i]

            roi_set = self.reference_rois.get(dname)
            if roi_set is None:
                raise ValueError(
                    f"No reference RoiSet for device {dname}. Call set_reference() first."
                )

            stamps = roi.extract_stamps(image, roi_set)
            processed = roi.quantify(stamps, roi_set, features=self.features)
            processed_data = processed.reset_index()
            metadata = pd.DataFrame([self.image_data.iloc[i]] * len(processed_data)).reset_index(drop=True)
            merged = pd.concat([metadata, processed_data], axis=1)
            data.append(merged)
            
            if save_summary_images:
                # Single-crop: reuse stamps for render (no ChipImage/mapto)
                outpath = self._summary_outpath(image)
                summary_image = roi.render_summary(
                    stamps, roi_set, features=self.features, metrics=processed
                )
                self._imsave_summary(
                    summary_image, outpath, as_ubyte=as_ubyte, compress=compress
                )

            del stamps

        return pd.concat(data, ignore_index=False)

    def _process_from_reference_parallel(
        self,
        as_ubyte: bool = False,
        compress: bool = False,
        n_jobs: int = -1,
    ):
        """Parallel quantify across images (no summary images)."""
        # Group by device so each job gets the right RoiSet
        rows = []
        paths = []
        rois = []
        features_list = []
        for i in range(len(self.image_data)):
            dname, image = self.image_data[['dname', 'image_path']].iloc[i]
            roi_set = self.reference_rois.get(dname)
            if roi_set is None:
                raise ValueError(
                    f"No reference RoiSet for device {dname}. Call set_reference() first."
                )
            rows.append(i)
            paths.append(image)
            rois.append(roi_set)
            features_list.append(self.features)

        # If all same device, one RoiSet; else per-path (worker gets its roi)
        workers = roi._resolve_n_jobs(n_jobs)
        payloads = list(zip(paths, rois, features_list))
        chunksize = max(1, len(payloads) // (workers * 4))
        with roi._stamp_process_pool(workers) as executor:
            dfs = list(
                tqdm(
                    executor.map(roi._quantify_image_worker, payloads, chunksize=chunksize),
                    total=len(payloads),
                    desc='Processing images',
                    leave=False,
                )
            )

        data = []
        for i, processed in zip(rows, dfs):
            processed_data = processed.reset_index()
            metadata = pd.DataFrame([self.image_data.iloc[i]] * len(processed_data)).reset_index(drop=True)
            data.append(pd.concat([metadata, processed_data], axis=1))
        return pd.concat(data, ignore_index=False)
