from dataclasses import dataclass, field
import numpy as np
from copy import deepcopy
import pint

@dataclass
class IndepVars:
    """Stores independent variables shared across multi-dimensional assay data objects.

    Attributes:
        concentration (pint.Quantity): 1D array of substrate or ligand concentrations (n_conc,).
        chamber_IDs (np.ndarray): 1D array of chamber identifiers/indices (n_chamb,).
        sample_IDs (np.ndarray): 1D array of sample names or identifiers per chamber (n_chamb,).
        button_quant_sum (pint.Quantity): 1D array of integrated button quantification signals (n_chamb,).
        time (pint.Quantity): 2D array of reaction timepoints per concentration (n_conc, n_time).
    """
    concentration: pint.Quantity     # (n_conc,)
    chamber_IDs: np.ndarray       # (n_chamb,)   # These two are dimensionless
    sample_IDs: np.ndarray        # (n_chamb,)   # since they're just labels
    button_quant_sum: pint.Quantity  # (n_chamb,)
    time: pint.Quantity              # (n_conc, n_time)

    def __post_init__(self):
        for var in self.concentration, self.chamber_IDs, self.sample_IDs, self.button_quant_sum:
            if var.ndim != 1:
                raise ValueError(f"Expected 1D array, got {var.shape} for {var}")
        if self.time.ndim != 2:
            raise ValueError(f"time must be 2D, got {self.time.shape}")

@dataclass
class Meta:
    """Metadata container tracking dataset provenance, applied masks, and model fits.

    Attributes:
        based_on (list[str]): List of previous run or dataset identifiers used to generate this dataset.
        description (str): Human-readable summary of dataset contents or processing history.
        applied_masks (list[str]): Names of filter or quality masks applied to the dataset.
        fit_type (str): Type of model fit performed (e.g., 'linear', 'MM', 'binding_isotherm').
        mask_type (str): Name of mask criteria (e.g., 'r_squared', 'positive_slope').
        mask_cutoff (float): Threshold value associated with the applied mask.
    """
    based_on: list[str] = field(default_factory=list)  # e.g. ['previous_run_1', previous_run_2'] if fit/masked from previus data
    description: str = field(default='')
    applied_masks: list[str] = field(default_factory=list)  # e.g. ['saved_mask_1', 'saved_mask_2'] if applied to this data
    # If it's a curve fit:
    fit_type: str = field(default='')  # e.g. 'linear', 'MM', etc.
    # If it's a mask:
    mask_type: str = field(default='')  # e.g. 'r_squared', 'positive_slope', etc.
    mask_cutoff: float = field(default=0.0)  # e.g. 0.9 for R2 cutoff

@dataclass
class Data4D:
    """Four-dimensional dataset structure containing kinetic time-series observations.

    Data dimensions correspond to: (concentrations, timepoints, chambers, values).

    Attributes:
        indep_vars (IndepVars): Shared independent variables.
        dep_var (np.ndarray): 4D numpy array storing dependent variable values.
        dep_var_type (list[str]): Descriptors for dependent variable values (e.g., ['luminance']).
        dep_var_units (list[pint.Unit]): Pint units corresponding to each dependent variable type.
        meta (Meta): Provenance and fitting metadata.
    """
    indep_vars: IndepVars

    dep_var: np.ndarray           # (n_conc, n_time, n_chamb, n_values)
    dep_var_type: list[str]       # e.g. ['luminance'] or ['slopes', 'intercepts']
    dep_var_units: list[pint.Unit]  # e.g. [units.RFU] or [units.RFU / units.s, units.RFU]
    
    meta: Meta = field(default_factory=Meta)

    def __post_init__(self):
        # make a full copy so original IndepVars isn’t shared
        self.indep_vars = deepcopy(self.indep_vars)
        self.indep_vars.__post_init__()  # validate copied indep vars
        if self.dep_var.ndim != 4:
            raise ValueError(f"dep_var must be 4D, got {self.dep_var.shape}")

@dataclass
class Data3D:
    """Three-dimensional dataset structure containing summary assay observations.

    Data dimensions correspond to: (concentrations, chambers, values).

    Attributes:
        indep_vars (IndepVars): Shared independent variables.
        dep_var (np.ndarray): 3D numpy array storing dependent variable values.
        dep_var_type (list[str]): Descriptors for dependent variable values (e.g., ['slopes', 'r_squared']).
        dep_var_units (list[pint.Unit]): Pint units corresponding to each dependent variable type.
        meta (Meta): Provenance and fitting metadata.
    """
    indep_vars: IndepVars

    dep_var: np.ndarray           # (n_conc, n_chamb, n_values)
    dep_var_type: list[str]       # e.g. ['luminance'] or ['slopes', 'intercepts']
    dep_var_units: list[pint.Unit]  # e.g. [units.RFU] or [units.RFU / units.s, units.RFU]

    meta: Meta = field(default_factory=Meta)

    def __post_init__(self):
        self.indep_vars = deepcopy(self.indep_vars)
        self.indep_vars.__post_init__()
        if self.dep_var.ndim != 3:
            raise ValueError(f"dep_var must be 3D, got {self.dep_var.shape}")

@dataclass
class Data2D:
    """Two-dimensional dataset structure containing single-timepoint or fitted summary values per chamber.

    Data dimensions correspond to: (chambers, values).

    Attributes:
        indep_vars (IndepVars): Shared independent variables.
        dep_var (np.ndarray): 2D numpy array storing dependent variable values.
        dep_var_type (list[str]): Descriptors for dependent variable values (e.g., ['Kd', 'r_max']).
        dep_var_units (list[pint.Unit]): Pint units corresponding to each dependent variable type.
        meta (Meta): Provenance and fitting metadata.
    """
    indep_vars: IndepVars
    
    dep_var: np.ndarray           # (n_chambers, n_values)
    dep_var_type: list[str]       # e.g. ['luminance'] or ['slopes', 'intercepts']
    dep_var_units: list[pint.Unit]  # e.g. [units.RFU] or [units.RFU / units.s, units.RFU]
    
    meta: Meta = field(default_factory=Meta)

    def __post_init__(self):
        self.indep_vars = deepcopy(self.indep_vars)
        self.indep_vars.__post_init__()
        if self.dep_var.ndim != 2:
            raise ValueError(f"dep_var must be 2D, got {self.dep_var.shape}")
