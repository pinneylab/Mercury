# Analysis & Fitting Guide

The `mercury.analysis` subpackage provides tools for fitting kinetic rates, Michaelis-Menten constants, binding isotherms, filtering data, and generating diagnostic plots.

---

## 1. MercuryExperiment Wrapper

`MercuryExperiment` brings together database entries, curve fitting routines, and plot export methods:

```python
from mercury.analysis.experiment import MercuryExperiment
from mercury.db_api.mercury_db_api import LocalMercuryDBAPI

db = LocalMercuryDBAPI()
db.load_binding_data('data/binding.csv')

exp = MercuryExperiment(db_api=db)
```

---

## 2. Fitting Binding Isotherms

Fit binding isotherms ($K_d$ and $R_{max}$) to chamber or sample data:

```python
from mercury.analysis.fit import fit_binding_isotherm

# Fit binding curves across concentrations
fit_results = exp.fit_binding_isotherms(
    tight_binders=['sample_A', 'sample_B'],  # Fixed R_max reference samples
    fixed_rmax=None
)

print(fit_results.head())
```

---

## 3. Exporting Subplots & Summary Plots

Generate diagnostic plots per chamber and per sample:

```python
# Export binding isotherm plots for each sample
exp.export_binding_subplots_by_sample(output_dir='plots/binding_by_sample')

# Export summary data tables
exp.export_binding_sample_data('output/binding_sample_summary.csv')
exp.export_binding_chamber_data('output/binding_chamber_summary.csv')
```
