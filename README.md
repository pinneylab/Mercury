# Mercury
Duncan Muir
Nicholas Freitas
Jonathan Zhang

Credits: Daniel Mohktari and Scott Longwell
___
![BuildStatus](https://github.com/pinneylab/mercury/actions/workflows/ci.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)
[![Coverage Status](https://coveralls.io/repos/github/pinneylab/mercury/badge.svg?branch=main)](https://coveralls.io/github/pinneylab/mercury?branch=main)

## Overview
This is a *WORK IN PROGRESS* package for processing and analysing various assays from the Mercury/HTBAM/HTMEK platform.

## Installation

### Conda Environment Setup

(Recommended) Create a fresh conda environment with python=3.12 using:

```conda create -n mercury python=3.12```

and activate using:

```conda activate mercury```

#### Install Stable Release via Wheel File
Download the latest wheel file from the [Release Page](https://github.com/pinneylab/mercury/releases)

Then, install the package to your conda environment using:

```pip install /path/to/downloaded/wheel.file```

#### For latest code (Not recommended) clone the repo and install locally

1. Clone this repo from the pinneylab Github
2. Change directory to unzipped package path
    - `$ cd /repo-download-dir`
3. pip install the package in place and make editable
    - `$ pip install -e .`

## Processing and Analyzing Data

Our processing and analysis is done in Jupyter notebooks. To get started, download the [latest release](https://github.com/pinneylab/mercury_notebooks/releases/latest) of our notebooks repo.

Then in your conda environment, start the Jupyter server with the command `jupyter notebook`.


## Roadmap

### Stitching
Done:
1. Stream tiles: read, rotate, trim, paste into a pre-allocated canvas, and drop tiles (~one tile + one canvas peak for FF-off cut).
2. Parallelize stitching across rasters with a memory-aware process pool (`n_workers` on `ImageStitcher.stitch_images`).
3. In-place pasting during cut-stitch (shared `assemble_cut` kernel; no concat stack).
4. Faster tile rotation via `cv2.warpAffine` (`rastering.rotate_image`).

Down the line:
1. Default source of truth for overlap parameter should come from `imaging.csv`, not auto-calculated. Additionally, overlap should really be considered as a per-image parameter (or at least a shared parameter across images taken with the same settings).
2. Prefetch + aggressive release + memmap/tiled writes.

### Processing

#### Current Implementation
- Stamp: a cropped pixel tile, not a mask. The class implemented in the processing module contains the tile data, the slice (the row and column slice that produced the tile in chip coordinates), the index (grid position on the device), and the identifier.
- Feature finding code identifies buttons and/or chambers within stamps. 

1. Parallelize across stamps for chamber/button finding.

2. Vectorize stamp mapping across a batch dimension. Consider the following approach:
    - After `set_reference`, pre-build boolean masks for the chamber, button, and annulus  ([N, H, W]).
    - Load target image via memmap, get stamps [N, H, W], vectorized reductions via the masks, then calculate per-stamp median, sum, and std
        - sum/std can be vectorized, stick to loop for median
    - Free aggressively. Do not keep the full chip image after the row is appended to the `csv`. Don't keep the full stitch once stamps are cropped (or never materialized and use memmap + slice). Delete stamps to free up memory, if needed.
    - Can parallelize the above process across chip images. 

3. Optimize button finding.
    - Remove unused copy of cicularSubsection; cut deepcopy in button search.
    - Vectorize the course grid search
    - Update 110 to the actual shape (100)
