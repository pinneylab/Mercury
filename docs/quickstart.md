# Quickstart Guide

This guide will walk you through setting up `mercury` and running basic data processing and analysis.

---

## 1. Installation

### Recommended: Conda Environment Setup

We recommend installing `mercury` inside a fresh Python 3.12 environment:

```bash
conda create -n mercury python=3.12 -y
conda activate mercury
```

### Installation Options

#### Option A: Install via Pip (Editable / Local Repository)

```bash
git clone https://github.com/pinneylab/mercury.git
cd mercury
pip install -e .
```

#### Option B: Install via Wheel File

Download the latest `.whl` package release from the GitHub Releases page:

```bash
pip install mercury-2.0.0-py3-none-any.whl
```

---

## 2. Verifying Installation

Verify that `mercury` is installed correctly by checking the package version in Python:

```python
import mercury
print(mercury.__version__)
```

You can also test the command-line utility:

```bash
pick-corners --help
```

---

## 3. Basic Workflow Overview

A standard Mercury workflow involves four key steps:

1. **Stitching & Alignment**: Align raw camera images and stitch tile grids using `mercury.stitching`.
2. **Quantification**: Quantify chamber fluorescence across channels using `mercury.processing`.
3. **Database Ingestion**: Load quantified CSV files into `LocalMercuryDBAPI` (`mercury.db_api`).
4. **Analysis & Fitting**: Fit kinetic rates or binding isotherms using `MercuryExperiment` (`mercury.analysis`).

For detailed step-by-step guides on each component, refer to the [User Guides](guides/processing.md).
