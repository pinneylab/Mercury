"""General data export utilities for Mercury DataND objects.

Provides dimension-agnostic conversion of Data2D, Data3D, and Data4D objects
into tidy pandas DataFrames and CSV files, supporting both chamber-level
(well-level) detail and per-sample aggregation.
"""

from pathlib import Path
from typing import Optional, Sequence, Union
import os
import numpy as np
import pandas as pd
import pint

from mercury.db_api.data import Data2D, Data3D, Data4D


def _format_column_name(base_name: str, unit: Optional[pint.Unit], include_units: bool) -> str:
    """Format a column header with its Pint unit symbol if applicable.

    Args:
        base_name: Base identifier for the column.
        unit: Pint unit object, or None.
        include_units: Whether to append the unit to the header.

    Returns:
        Formatted column header string, e.g. 'v_max (RFU / s)' or 'r_squared'.
    """
    if not include_units or unit is None:
        return base_name

    try:
        unit_str = f"{unit:~}".strip()
    except Exception:
        unit_str = str(unit).strip()

    if not unit_str or unit_str == "dimensionless":
        return base_name

    return f"{base_name} ({unit_str})"


def data_to_dataframe(
    data: Union[Data2D, Data3D, Data4D],
    aggregate_by_sample: bool = False,
    agg: Union[str, Sequence[str]] = ("mean", "std"),
    include_units: bool = True,
    include_replicates: bool = True,
    include_chamber_ids: bool = False,
    chamber_col: str = "chamber",
    sample_col: str = "sample",
) -> pd.DataFrame:
    """Convert any Data2D, Data3D, or Data4D object into a tidy pandas DataFrame.

    Args:
        data: A Data2D, Data3D, or Data4D instance.
        aggregate_by_sample: If True, aggregate data per-sample across replicates.
        agg: Aggregation statistic(s) to compute ('mean', 'std', 'median', 'min', 'max', 'sum').
             Can be a string or a sequence of strings. Supports 'avg' as an alias for 'mean'.
        include_units: If True, append unit symbols to column headers (e.g. '(RFU / s)').
        include_replicates: If True and aggregating by sample, include a 'replicates' count column.
        include_chamber_ids: If True and aggregating by sample, include a semicolon-separated
                             'chamber_IDs' column listing contributing chambers.
        chamber_col: Column header for chamber identifiers (default: 'chamber').
        sample_col: Column header for sample names (default: 'sample').

    Returns:
        A pandas DataFrame containing the reshaped and optionally aggregated data.
    """
    if not isinstance(data, (Data2D, Data3D, Data4D)):
        raise TypeError(
            f"Expected Data2D, Data3D, or Data4D, got {type(data).__name__}"
        )

    dep_var = data.dep_var
    indep = data.indep_vars
    n_chamb = indep.chamber_IDs.shape[0]

    # Resolve dependent variable column headers
    dep_col_names = []
    dep_units = getattr(data, "dep_var_units", []) or []
    for idx, var_name in enumerate(data.dep_var_type):
        unit = dep_units[idx] if idx < len(dep_units) else None
        dep_col_names.append(_format_column_name(var_name, unit, include_units))

    # --- Reshape to Tidy Chamber-Level DataFrame ---
    if isinstance(data, Data2D):
        data_dict = {
            chamber_col: indep.chamber_IDs,
            sample_col: indep.sample_IDs,
        }
        for idx, col_name in enumerate(dep_col_names):
            data_dict[col_name] = dep_var[:, idx]

        df = pd.DataFrame(data_dict)
        group_axes = [sample_col]

    elif isinstance(data, Data3D):
        n_conc = dep_var.shape[0]
        conc_unit = getattr(indep.concentration, "units", None)
        conc_col = _format_column_name("concentration", conc_unit, include_units)
        conc_vals = (
            indep.concentration.magnitude
            if hasattr(indep.concentration, "magnitude")
            else np.asarray(indep.concentration)
        )

        data_dict = {
            conc_col: np.repeat(conc_vals, n_chamb),
            chamber_col: np.tile(indep.chamber_IDs, n_conc),
            sample_col: np.tile(indep.sample_IDs, n_conc),
        }
        for idx, col_name in enumerate(dep_col_names):
            data_dict[col_name] = dep_var[:, :, idx].reshape(-1)

        df = pd.DataFrame(data_dict)
        group_axes = [conc_col, sample_col]

    elif isinstance(data, Data4D):
        n_conc = dep_var.shape[0]
        n_time = dep_var.shape[1]

        conc_unit = getattr(indep.concentration, "units", None)
        conc_col = _format_column_name("concentration", conc_unit, include_units)
        conc_vals = (
            indep.concentration.magnitude
            if hasattr(indep.concentration, "magnitude")
            else np.asarray(indep.concentration)
        )

        time_unit = getattr(indep.time, "units", None)
        time_col = _format_column_name("time", time_unit, include_units)
        time_raw = (
            indep.time.magnitude
            if hasattr(indep.time, "magnitude")
            else np.asarray(indep.time)
        )

        # Handle 2D vs 1D time arrays
        if time_raw.ndim == 2:
            time_expanded = np.repeat(time_raw[:, :, np.newaxis], n_chamb, axis=2).reshape(-1)
        elif time_raw.ndim == 1:
            time_tile = np.tile(time_raw, n_conc)
            time_expanded = np.repeat(time_tile[:, np.newaxis], n_chamb, axis=1).reshape(-1)
        else:
            time_expanded = np.tile(time_raw.flatten(), (n_conc * n_time * n_chamb) // len(time_raw.flatten()))

        data_dict = {
            conc_col: np.repeat(conc_vals, n_time * n_chamb),
            time_col: time_expanded,
            chamber_col: np.tile(indep.chamber_IDs, n_conc * n_time),
            sample_col: np.tile(indep.sample_IDs, n_conc * n_time),
        }
        for idx, col_name in enumerate(dep_col_names):
            data_dict[col_name] = dep_var[:, :, :, idx].reshape(-1)

        df = pd.DataFrame(data_dict)
        group_axes = [conc_col, time_col, sample_col]

    if not aggregate_by_sample:
        return df

    # --- Per-Sample Aggregation ---
    if isinstance(agg, str):
        agg = [agg]

    agg_func_map = {
        "avg": "mean",
        "mean": "mean",
        "std": "std",
        "median": "median",
        "min": "min",
        "max": "max",
        "sum": "sum",
        "count": "count",
    }

    grouped = df.groupby(group_axes, sort=False)

    agg_results = []
    # Count valid replicates (non-null observations)
    if include_replicates:
        counts = grouped[dep_col_names[0]].count()
        agg_results.append(counts.rename("replicates"))

    # Compute requested statistics
    for func in agg:
        pandas_func = agg_func_map.get(func, func)
        prefix = "avg" if func == "avg" else ("mean" if func == "mean" else func)
        for col in dep_col_names:
            series = grouped[col].agg(pandas_func)
            new_col_name = f"{prefix}_{col}"
            agg_results.append(series.rename(new_col_name))

    # Optional chamber IDs listing
    if include_chamber_ids:
        def _get_valid_chambers(group_s):
            mask = group_s[dep_col_names[0]].notna()
            valid_chambs = group_s.loc[mask, chamber_col]
            return "; ".join(valid_chambs.astype(str))

        chamb_series = grouped.apply(_get_valid_chambers, include_groups=False)
        agg_results.append(chamb_series.rename("chamber_IDs"))

    res_df = pd.concat(agg_results, axis=1).reset_index()
    return res_df


def export_data_to_csv(
    data: Union[Data2D, Data3D, Data4D],
    file_path: Union[str, Path],
    aggregate_by_sample: bool = False,
    agg: Union[str, Sequence[str]] = ("mean", "std"),
    include_units: bool = True,
    include_replicates: bool = True,
    include_chamber_ids: bool = False,
    chamber_col: str = "chamber",
    sample_col: str = "sample",
) -> pd.DataFrame:
    """Convert a DataND object and export it directly to a CSV file.

    Args:
        data: A Data2D, Data3D, or Data4D instance.
        file_path: Path where the CSV file will be written.
        aggregate_by_sample: If True, aggregate data per-sample across replicates.
        agg: Aggregation statistic(s) to compute.
        include_units: If True, include unit symbols in column headers.
        include_replicates: If True and aggregating, include 'replicates' count.
        include_chamber_ids: If True and aggregating, include 'chamber_IDs' string.
        chamber_col: Column header for chamber identifiers.
        sample_col: Column header for sample names.

    Returns:
        The generated pandas DataFrame.
    """
    df = data_to_dataframe(
        data=data,
        aggregate_by_sample=aggregate_by_sample,
        agg=agg,
        include_units=include_units,
        include_replicates=include_replicates,
        include_chamber_ids=include_chamber_ids,
        chamber_col=chamber_col,
        sample_col=sample_col,
    )

    out_path = Path(file_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out_path), index=False)

    mode_desc = "Sample-aggregated" if aggregate_by_sample else "Chamber-level"
    print(f"{mode_desc} data exported to {out_path}")

    return df
