# Image Processing & Microfluidic Chip Guide

The `mercury.processing` subpackage manages microfluidic chip layouts, chamber coordinates, spot quantification, and image intensity extraction.

---

## 1. Chip Layouts & Corner Alignment

A microfluidic chip layout maps physical chamber locations on stitched images to indexed reaction micro-chambers.

### Corner Picking

Before quantification, corner coordinates are identified using the `pick-corners` CLI utility or `mercury.scripts.corner_picker`:

```bash
pick-corners --image stitched_overview.tif --output chip_corners.json
```

### Loading Chip Layouts

```python
from mercury.processing.chip import MercuryChip

# Load chip geometry using defined corner points
chip = MercuryChip(
    image_path='stitched_overview.tif',
    corners_path='chip_corners.json'
)

# Extract chamber locations and ROI bounding boxes
chambers = chip.get_chamber_rois()
```

---

## 2. Image Quantification

Quantify median or mean fluorescence intensities across channels and chambers:

```python
from mercury.processing.experiment import ProcessingExperiment

exp = ProcessingExperiment(chip=chip, image_dir='/path/to/stitched_images')

# Process button quant, kinetic series, or binding washes
quant_df = exp.quantify_chambers(channels=['eGFP', 'Cy5'])
quant_df.to_csv('quantified_chambers.csv', index=False)
```
