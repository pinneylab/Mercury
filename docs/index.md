# Mercury Documentation

Mercury is a Python library and suite of tools designed for processing, quantifying, storing, and analyzing high-throughput microfluidic enzyme kinetics (HT-MEK) assay data.

---

## Key Features

- **Image Processing**: Flat-field correction, background subtraction, tile rastering, image stitching, and automated spot/chamber quantification.
- **Database & Data Management**: Structured storage and querying for multi-dimensional assay data (button quant, standard curves, kinetics, binding isotherms, stability data) via `LocalMercuryDBAPI`.
- **Assay Analysis & Model Fitting**: Kinetic rate estimation, Michaelis-Menten kinetics, binding isotherm fitting, filtering, and data visualization.
- **Interactive & CLI Utilities**: Corner picker GUI for microfluidic chip alignment and interactive visualization tools.

---

## Package Architecture

```
mercury/
├── analysis/         # Kinetic fitting, binding isotherms, filtering & plotting
├── db_api/           # Local database interface, Data structures (Data2D, Data3D), CSV import/export
├── processing/       # Chip layouts, image intensity extraction, alignment
├── stitching/        # Image rastering, stitching algorithm implementations
└── scripts/          # Command-line entrypoints (e.g., corner picker UI)
```

---

## Getting Started

Check out the [Quickstart Guide](quickstart.md) to set up your environment and run your first data analysis, or jump directly into the [API Reference](api/analysis.md) for detailed class and function documentation.
