# Image Stitching Guide

The `mercury.stitching` module provides tools for flat-field correcting, stitching, and background-subtracting rastered image tiles acquired via Micro-Manager or RunPack.

---

## 1. Defining Stitching Settings

Before stitching, define tile dimensions and optional flat-field correction images:

```python
from mercury.stitching import rastering

# Flat-field parameters: {channel: {exposure_ms: (dark_field, mean_intensity)}}
ff_params = {'4egfp': {500: (-150, 16665)}}
ff_paths = {'4egfp': '/path/to/FF_eGFP_500ms.tif'}

settings = rastering.StitchingSettings(
    ffPaths=ff_paths,
    ffParams=ff_params,
    setupNum=2,
    tileDim=1024
)
```

---

## 2. Walking & Stitching Directory Trees

### RunPack Raster Stitching

For flat rasters collected as single scans or kinetic series:

```python
overlap = 0.1  # Fractional tile overlap (0 to 1)
params = rastering.RasterParams(overlap=overlap, autoFF=True)

# Walk directory and stitch images
rastering.walkAndStitch('/path/to/raw_images', params, stitchtype='kinetic')
```

### Micro-Manager Stack Stitching

For `.ome.tif` Micro-Manager image stacks:

```python
channel_exposure_map = {'3-GFP-B': 500, '5------': 100}
channel_remap = {'3-GFP-B': 'eGFP', '5------': 'Cy5'}

params = rastering.RasterParams(overlap=0.1, autoFF=False)
rastering.MMStitchStacks(
    root='/path/to/stack_images',
    raster_params=params,
    channelExposureMap=channel_exposure_map,
    channelRemap=channel_remap
)
```

---

## 3. Image Background Subtraction

Background subtraction is applied to full stitched composite images:

```python
bg = rastering.BackgroundImages()

# Register reference background images
bg.add('/path/to/bg_device1.tif', device='d1', channel='eGFP', exposure=500)

# Subtract background across target directories
bg.walkAndBGSubtract(targetRoot='/path/to/stitched', device='d1', channel='eGFP')
```
