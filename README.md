# Mercury

**A Python toolkit for processing and analyzing high-throughput microfluidic enzyme kinetics (HT-MEK) data.**

[![Tests](https://github.com/pinneylab/mercury/actions/workflows/ci.yml/badge.svg)](https://github.com/pinneylab/mercury/actions/workflows/ci.yml)
[![Coverage](https://coveralls.io/repos/github/pinneylab/mercury/badge.svg?branch=main)](https://coveralls.io/github/pinneylab/mercury?branch=main)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![Latest release](https://img.shields.io/github/v/release/pinneylab/mercury?include_prereleases)](https://github.com/pinneylab/mercury/releases)

Mercury provides a reproducible workflow for converting HT-MEK microscopy data into calibrated measurements and fitted biochemical parameters. It supports image stitching and chamber quantification, structured assay-data handling, initial-rate and model fitting, quality-control filtering, and diagnostic visualization.

The companion [Mercury notebooks](https://github.com/pinneylab/htbam_notebooks) provide the recommended, step-by-step interface for routine data processing and analysis.

> *[In Review]* Freitas, N., Zhang, J., Muir, D., et al. **Processing and analyzing high-throughput microfluidic enzymology data: a practical guide to rate fitting and quality control.** *Methods in Enzymology*.

## What Mercury does

- Stitches tiled microscopy images and corrects uneven illumination.
- Locates and quantifies microfluidic reaction chambers.
- Imports button-quantification, standard-curve, kinetic, binding, and stability measurements.
- Converts fluorescence measurements to product concentrations using per-chamber calibration curves.
- Fits initial rates, Michaelis–Menten parameters, binding isotherms, and user-supplied models.
- Applies explicit quality-control filters at the chamber, concentration, and sample levels.
- Exports fitted parameters, filtered datasets, and end-to-end diagnostic plots.

## Getting started

### 1. Create an environment

We recommend a fresh Conda environment with Python 3.12:

```bash
conda create -n mercury python=3.12 -y
conda activate mercury
```

### 2. Install Mercury

For the most recent official release:

```bash
python -m pip install mercury-kinetics
```

Or, to install the current development version instead:

```bash
git clone https://github.com/pinneylab/mercury.git
cd mercury
python -m pip install -e .
```

Verify the installation:

```bash
python -c "import mercury; print(mercury.__version__)"
```

### 3. Run the tutorial notebooks

Download the [latest notebook release](https://github.com/pinneylab/mercury_notebooks/releases/latest), activate the same environment, and start Jupyter:

```bash
jupyter notebook
```

For a first analysis, begin with the Michaelis–Menten analysis notebook. The released notebook archive includes example data and additional workflows where available.

## Documentation
> Check back soon for in-depth documentation...
- [Quickstart guide](docs/quickstart.md)
- [API reference](docs/api/analysis.md)
- [Command-line tools](docs/cli.md)

## Using Mercury beyond Michaelis–Menten kinetics

Mercury separates data ingestion, transformation, fitting, filtering, and visualization so that the workflow can be adapted to different HT-MEK assays. In addition to Michaelis–Menten analysis, the package includes support for binding and stability measurements, and its fitting routines can be extended with user-provided models and analysis logic.

## Reporting problems and contributing

To report a bug, request a feature, or ask a usage question, [open a GitHub issue](https://github.com/pinneylab/mercury/issues). When reporting a problem, please include:

- the Mercury release or commit used;
- your operating system and Python version;
- a minimal example or relevant input-file description; and
- the complete error message or unexpected output.

Contributions are welcome through pull requests. Please run the test suite before submitting changes:

```bash
pytest
```

## Citation

If Mercury contributes to published work, please cite the Methods in Enzymology chapter and the archived software release:

> *[In Review]* Freitas, N., Zhang, J., Muir, D., et al. **Processing and analyzing high-throughput microfluidic enzymology data: a practical guide to rate fitting and quality control.** *Methods in Enzymology*.


## Authors and acknowledgments

Mercury is developed by Nicholas Freitas, Duncan Muir, Jonathan Zhang, and contributors in the [Pinney Lab](https://pinneylab.com/).

Additional contributions and foundational development: Daniel Mohktari, Scott Longwell, and members of the [Fordyce Lab](https://www.fordycelab.com/).

## License

Mercury is distributed under the [MIT License](LICENSE).
