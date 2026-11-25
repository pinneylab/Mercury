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

### Command-line processing (`mercury-run`)

`mercury-run` runs the standard pipeline (stitch, background subtract, check corners, find features, quantify) from one config file and records every parameter it used, so a run can be repeated and audited. It is built to be run by a person or by an agent.

`mercury-run` is installed with the package (`pip install -e .`; reinstall after pulling if the command is missing). `python -m mercury.pipeline` is equivalent.

#### 1. Write a config

```bash
mercury-run template > processing.yaml
```

Edit it for your experiment: `root` (the `scope_data/YYYYMMDD` folder), a starting `stitching.rotation`, one or more `test_rasters` per objective (rasters with visible features, e.g. button quant or brightfield), each device's `dims`, `pinlist` and starting `corners`, and the series to quantify. Keep `processing.yaml` with your notebook entry, under git.

#### 2. Run the stages

```bash
mercury-run preflight --config processing.yaml   # check inputs before anything runs
mercury-run all --config processing.yaml         # run every stage in order
mercury-run status --config processing.yaml     # what has run, what is next, open requests
```

`all` stops at the first stage that does not pass. Stages can also be run one at a time:

| Stage | What it does |
|---|---|
| `preflight` | Checks tiles, `imaging.csv`, pinlists, series identifiers, test rasters, and that each image group has exactly one background. Writes no images. |
| `calibrate` | Measures seam mismatch on the test rasters. Refits rotation or per-objective overlap if they fail `qc.seam_max_residual_px`. |
| `stitch` | Stitches every raster with the calibrated rotation and overlap. |
| `bgsub` | Background-subtracts with the matched backgrounds. |
| `corners` | Refines each series' corners on its reference image and checks them (shift, lattice fit, ring check for a grid off by a whole row/column). |
| `reference` | Finds features on the reference image and caches the ROIs. |
| `map` | Quantifies every image in each series and writes `<root>/processed/<output>`. |

A stage skips itself if its inputs (config, upstream outputs and the Mercury code version) are unchanged. Use `--redo` to recompute anyway.

#### 3. Read the result

Each command's last line is `RESULT {json}`. Exit codes:

- **0**: pass or warn. Continue.
- **2**: needs a person. The result (and `status`) gives a specific request, e.g. pick corners for a device with `pick-corners` and add them to `processing.yaml`. Then re-run the stage.
- **1**: error (missing prerequisite, bad input, refused overwrite, or a code error).

Library output goes to `<root>/qc/logs/`. QC tables (seam residuals, per-stamp corner offsets) go to `<root>/qc/`.

#### Files written next to `processing.yaml`

| File | Written by | Contents |
|---|---|---|
| `processing.yaml` | you | What to process and the QC gates |
| `processing.agent.yaml` | `mercury-run override` | Allowlisted adjustments, each with a reason |
| `processing.resolved.yaml` | `mercury-run` | The lockfile: every value used and its source, QC results, output SHA-256s, Mercury commit |
| `processing.log.jsonl` | `mercury-run` | Append-only history of stages, decisions, escalations and notes |

Commit all four with the notebook entry. Never edit the last three by hand.

#### Overrides and notes

```bash
mercury-run override --config processing.yaml \
    --key series.nadh.reference.pick --value max --reason "standard curve: use highest concentration"
mercury-run note --config processing.yaml --reason "d3 inlet debris near column 1"
```

Only a few keys can be overridden (test rasters, reference image rules, and corners picked by a person). QC thresholds, features, identifiers and background matching can only be changed in `processing.yaml`.

#### Safeguards

- `--overwrite`: required to replace outputs that this record did not make (e.g. an earlier hand-run stitch).
- `--allow-dirty`: required to run with uncommitted changes in Mercury. It is recorded in the resolved file; use it for testing only.

#### Running with an agent

`.claude/skills/htbam-processing/SKILL.md` tells a Claude Code agent how to run the pipeline: stage order, what it may change on its own, when to stop and ask, and what to report. Copy or symlink that folder into the `.claude/skills/` of the repo you run from (e.g. your notebook repo) or into `~/.claude/skills/`. `settings.example.json` beside it has example permissions that block edits to raw data and the record files.


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

2. RoiSet-based mapping (replace mapto hot path)
   - After feature finding / set_reference, build RoiSet:
       slices, indices, ids,
       chamber/button geometry + blank flags,
       boolean masks [N,H,W] (True = inside ROI, same convention as circularSubsection today)
   - quantify(path|memmap, roi) -> DataFrame
       extract stamps [N,H,W];
       vectorized sum/mean/std; median in a loop over N;
       emit same columns as Chamber/Button.summarize (incl. BG-sub, annulus)
       for failures, NaN rows (same as BlankChamber / BlankButton)
   - render_summary(stamps, roi, metrics=None) -> image
       annotate from geometry (+ optional metric text); stitch; no Chamber/Button
       enable annotation of both chamber / button features if 'all' is passed
   - Processor._process_from_reference:
       single crop → quantify → (if save_summary_images) render → free stamps
       do not keep full ChipImage / mapto on this path
   - Keep ChipImage.mapto / summary_image for now; deprecate later
   - v1: serial over images; image-level pool as follow-up
   - Test: legacy mapto+summarize vs quantify; unity scatter (reuse processing_compare)
   - Prefer `processing/roi.py` to prevent bloating of `chip.py`.
   - Down the line: ensure that RoiSet to quantify to render summary is the standard. No need to do this immediately.

3. Enable saving of summary images during parallelized feature mapping (easy fix in theory).

4. Optimize button finding.
    - Remove unused copy of cicularSubsection; cut deepcopy in button search.
    - Vectorize the course grid search
    - Update 110 to the actual shape (100)

5. Abstract `roi.py` and cleanup deprecated functions in `chip.py`
    - Abstract classes. For each geometry, specify headers and metric calculation functions.
    - Feature finding lives in `chip.py`. Geometry transfer, quantification, and storage live in `roi.py`. `Processor` class orchestrates. 
        - Still some overlap in quantification (i.e. `.summarize()` methods and `roi._quantify_*`), summary image saving, and geometry transfer (i.e. deprecated `mapto`) 