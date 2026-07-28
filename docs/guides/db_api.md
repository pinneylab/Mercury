# Database API & Data I/O Guide

The `mercury.db_api` module handles ingestion, structured storage, unit conversion, and retrieval for multi-dimensional assay data.

---

## 1. Local Mercury Database API (`LocalMercuryDBAPI`)

`LocalMercuryDBAPI` provides a flexible interface for loading combinations of:
- **Button Quant**: Initial protein / antibody quantification
- **Standard Curves**: Concentration vs. intensity calibration curves
- **Kinetic Assays**: Time series fluorescence data
- **Binding Isotherms**: Multi-concentration prey/bait ratios
- **Stability Data**: Protein stability / denaturation metrics

### Initialization & Data Ingestion

```python
from mercury.db_api.mercury_db_api import LocalMercuryDBAPI

db = LocalMercuryDBAPI()

# Ingest quantified CSV datasets
db.load_button_quant('data/button_quant.csv')
db.load_binding_data('data/binding_isotherm.csv')
db.load_standard_curves('data/standards.csv')
```

---

## 2. 2D and 3D Data Representation

Mercury encapsulates multi-dimensional experimental arrays using dedicated data structures:

- **`Data2D`**: Structured tabular data across chambers and single conditions/timepoints.
- **`Data3D`**: Multi-axis tensor data across chambers, concentrations, and time/washes.

```python
from mercury.db_api.data import Data3D

# Access binding ratios across concentrations and chambers
ratios = db.get_binding_ratios() # Returns Data3D instance
print(ratios.shape)
```
