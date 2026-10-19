# Quickstart Guide

This guide gets you set up with `mercury` and running your first high-throughput enzyme kinetics analysis in under 10 minutes.

---

## 1. Installation

### Recommended: Conda Environment Setup

We recommend running Mercury in a dedicated Python 3.12 environment:

```bash
conda create -n mercury python=3.12 -y
conda activate mercury
```

### Installing Mercury

Clone and install `mercury` in editable mode:

```bash
git clone https://github.com/pinneylab/mercury.git
cd mercury
pip install -e .
```

Verify that `mercury` is installed properly:

```python
import mercury
print("Mercury version:", mercury.__version__)
```

---

## 2. Getting the Template Notebooks

Mercury data analysis workflows are organized into standardized Jupyter template notebooks in the companion repository [mercury_notebooks](https://github.com/pinneylab/mercury_notebooks):

```bash
git clone https://github.com/pinneylab/mercury_notebooks.git
cd mercury_notebooks
```

The repository provides two core workflows:

| Workflow | Notebook | Purpose |
|---|---|---|
| **Kinetics Analysis & QC** | `htbam_notebooks/analysis_mm.ipynb` | Converts fluorescence tables to calibrated rates (V<sub>0</sub>), applies QC filters, fits Michaelis–Menten parameters (k<sub>cat</sub>, K<sub>M</sub>), and exports publication summaries |
| **Image Processing** | `htbam_notebooks/process.ipynb` | Stitches raw microscopy tiles, performs background subtraction, identifies reaction chambers, and quantifies fluorescence |

> **Where to start?**
> We recommend starting with **`analysis_mm.ipynb`** using the provided example dataset (`example_data/9_17_25/`). Image processing is only needed if you are starting from raw multi-gigabyte microscopy image tiles.

---

## 3. Minimal Working Example: 10-Minute Analysis

Here is a minimal script demonstrating the complete flow from quantified tables to fitted Michaelis–Menten parameters. This is for demonstration purposes - again, we recommend walking through the [template notebooks](https://github.com/pinneylab/mercury_notebooks) for actual experiments.

```python
from pathlib import Path
from mercury.db_api.mercury_db_api import LocalMercuryDBAPI
from mercury.analysis.experiment import MercuryExperiment
from mercury.db_api.units import units
from mercury.analysis.transform import transform_data
from mercury.analysis.fit import fit_luminance_vs_concentration, fit_concentration_vs_time, fit_initial_rates_vs_concentration_with_function, mm_model

# 1. Connect to data tables
data_dir = Path('./example_data/9_17_25/mm_adk_data')
db_conn = LocalMercuryDBAPI(
    standard_curve_data_path=str(data_dir / 'd1_NADPH_500ms_StandardSeries_Analysis.csv'),
    standard_name="NADPH", standard_substrate="NADPH", standard_units=units.uM, standard_concentration_col="concentration",
    kinetic_data_path=str(data_dir / 'd1_500ms_TitrationSeries_Analysis.csv'),
    kinetic_name="ADP", kinetic_substrate="ADP", kinetic_units=units.uM, kinetic_concentration_col="adp_conc",
    time_units=units.s,
    button_quant_data_path=str(data_dir / 'button_quant.csv')
)
exp = MercuryExperiment(db_conn)

# 2. Fit standard curves and convert RFU to [Product]
std_fits = fit_luminance_vs_concentration(exp.get_run('NADPH'))
exp.set_run('NADPH_standard', std_fits)
product_conc = transform_data([exp.get_run('ADP'), std_fits], '(a.luminance - b.intercept) / b.slope', output_name='concentration')
exp.set_run('kinetics_conc', product_conc)

# 3. Fit initial reaction velocities (V0)
kinetics_fits, _ = fit_concentration_vs_time(product_conc, start_timepoint=1, end_timepoint=4, max_reaction_percent=100)
exp.set_run('kinetics_fits', kinetics_fits)

# 4. Fit Michaelis-Menten parameters (Vmax, KM)
mm_fits, mm_pred = fit_initial_rates_vs_concentration_with_function(data=kinetics_fits, model_func=mm_model)
exp.set_run('mm_fits', mm_fits)

print("Fit completed successfully!")
```

---

## 4. Next Steps & Detailed Guides

To walk through real experimental analysis with step-by-step instructions, parameter tuning, diagnostic visualizations, and quality control:

- **[Michaelis–Menten Analysis & QC Guide](guides/notebook_analysis.md)**: Full walkthrough of `analysis_mm.ipynb`, including enzyme quantification, background rate subtraction, automated & manual QC masks, and diagnostic report generation.
- **[Image Processing & Quantification Guide](guides/notebook_processing.md)**: Full walkthrough of `process.ipynb`, including tile stitching, rotation alignment, corner picking, and feature quantification.
