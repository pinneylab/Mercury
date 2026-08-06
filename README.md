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
2. Prefetch + aggressive release + memmap/tiled writes
