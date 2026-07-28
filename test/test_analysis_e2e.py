import pytest
import numpy as np
from pathlib import Path

from mercury.db_api.mercury_db_api import LocalMercuryDBAPI
from mercury.db_api.units import units
from mercury.analysis.experiment import MercuryExperiment
from mercury.analysis.transform import transform_data
from mercury.analysis.fit import (
    fit_luminance_vs_concentration,
    fit_concentration_vs_time,
    fit_initial_rates_vs_concentration_with_function,
    mm_model,
)
from mercury.analysis.filter import (
    filter_by_sample_id,
    filter_expression_cutoff,
    filter_initial_rates_positive_cutoff,
    filter_initial_rates_r2_cutoff,
    filter_standard_curve_r2_cutoff,
    filter_number_concentrations,
    filter_r2_cutoff,
    filter_number_replicates,
)


@pytest.mark.e2e
def test_analysis_e2e_pipeline(reference_csv_dir):
    """
    End-to-end integration test for the Mercury analysis pipeline using
    pre-computed reference CSV data (Hayden's AdkSubset_DnaK dataset).
    """
    # 1. Verify reference CSV files exist
    button_quant_path = reference_csv_dir / "button_quant.csv"
    standard_data_path = reference_csv_dir / "standard_data.csv.bz2"
    kinetics_data_path = reference_csv_dir / "kinetics_data.csv.bz2"

    assert button_quant_path.exists(), f"Missing {button_quant_path}"
    assert standard_data_path.exists(), f"Missing {standard_data_path}"
    assert kinetics_data_path.exists(), f"Missing {kinetics_data_path}"

    # 2. Database API Connection
    EGFP_SLOPE = (91900.03 / 9) * (units.RFU / units.nM)

    db_conn = LocalMercuryDBAPI(
        standard_curve_data_path=str(standard_data_path),
        standard_name="NADPH",
        standard_substrate="NADPH",
        standard_units=units.uM,
        standard_concentration_col="concentration",
        kinetic_data_path=str(kinetics_data_path),
        kinetic_name="ADP",
        kinetic_substrate="ADP",
        kinetic_units=units.uM,
        kinetic_concentration_col="adp_conc",
        time_units=units.s,
        button_quant_data_path=str(button_quant_path),
    )

    mercury_experiment = MercuryExperiment(db_conn)
    run_names = set(db_conn.get_run_names())
    assert {"NADPH", "ADP", "button_quant"}.issubset(run_names)

    # 3. Enzyme Quant Transformation
    button_concentrations = transform_data(
        data_objs=[mercury_experiment.get_run("button_quant")],
        expr="(a_luminance / slope)",
        expression_vars={"slope": EGFP_SLOPE},
        output_name="concentration",
    )
    mercury_experiment.set_run("enzyme_concentrations", button_concentrations)

    assert button_concentrations is not None
    assert button_concentrations.dep_var.shape[0] == 1792
    assert np.all(np.isfinite(button_concentrations.dep_var))

    # 4. Product Standards Fitting
    standard_experiment_data = mercury_experiment.get_run("NADPH")
    standard_fits = fit_luminance_vs_concentration(standard_experiment_data)
    mercury_experiment.set_run("NADPH_standard", standard_fits)

    assert standard_fits is not None
    assert "slope" in standard_fits.dep_var_type
    assert "r_squared" in standard_fits.dep_var_type

    # 5. Kinetics Product Concentration Transformation
    product_concentrations = transform_data(
        data_objs=[
            mercury_experiment.get_run("ADP"),
            mercury_experiment.get_run("NADPH_standard"),
        ],
        expr="(a.luminance - b.intercept) / b.slope",
        output_name="concentration",
    )
    mercury_experiment.set_run("kinetics_ADP_conc", product_concentrations)

    assert product_concentrations is not None

    # 6. Fit Initial Rates vs Time
    kinetics_concentrations = mercury_experiment.get_run("kinetics_ADP_conc")
    kinetics_fits, fit_points_mask = fit_concentration_vs_time(
        kinetics_concentrations,
        start_timepoint=1,
        end_timepoint=4,
        max_reaction_percent=100,
    )
    mercury_experiment.set_run("kinetics_ADP_conc_fits", kinetics_fits)
    mercury_experiment.set_run("fit_points_mask", fit_points_mask)

    assert kinetics_fits is not None
    assert "slope" in kinetics_fits.dep_var_type

    # 7. Background Rate Subtraction using meGFP control wells
    buffer_mask = filter_by_sample_id(kinetics_fits, sample_ids=["meGFP"])
    mercury_experiment.set_run("buffer_mask", buffer_mask)

    mercury_experiment.apply_mask(
        run_name="kinetics_ADP_conc_fits",
        dep_variables=["slope", "intercept", "r_squared"],
        save_as="buffer_wells",
        mask_names=["buffer_mask"],
    )

    buffer_wells = mercury_experiment.get_run("buffer_wells")
    buffer_wells_lower_quartile = transform_data(
        data_objs=[buffer_wells],
        expr="np.nanpercentile(a.device.slope, 25, axis=1)",
        output_name="slope_lower_quartile",
    )
    mercury_experiment.set_run(
        "buffer_wells_lower_quartile", buffer_wells_lower_quartile
    )

    corrected_initial_rates = transform_data(
        data_objs=[
            mercury_experiment.get_run("kinetics_ADP_conc_fits"),
            mercury_experiment.get_run("buffer_wells_lower_quartile"),
        ],
        expr="(a.chamber.slope - b.chamber.slope_lower_quartile)",
        output_name="slope",
        keep_existing=True,
    )
    mercury_experiment.set_run(
        "kinetics_ADP_conc_fits_bgsub", corrected_initial_rates
    )

    assert corrected_initial_rates is not None

    # 8. Filter Initial Rates
    kinetics_fits_bgsub = mercury_experiment.get_run("kinetics_ADP_conc_fits_bgsub")
    initial_rates_r2_mask = filter_initial_rates_r2_cutoff(
        kinetics_fits_bgsub, r2_cutoff=0.9
    )
    initial_rates_positive_mask = filter_initial_rates_positive_cutoff(
        kinetics_fits_bgsub
    )
    standard_curve_r2_mask = filter_standard_curve_r2_cutoff(
        standard_fits, kinetics_fits_bgsub, r2_cutoff=0.9
    )
    expression_mask = filter_expression_cutoff(
        button_concentrations, kinetics_fits_bgsub, expression_cutoff=1
    )

    mercury_experiment.set_run("initial_rates_r2_mask", initial_rates_r2_mask)
    mercury_experiment.set_run(
        "initial_rates_positive_mask", initial_rates_positive_mask
    )
    mercury_experiment.set_run("standard_curve_r2_mask", standard_curve_r2_mask)
    mercury_experiment.set_run("expression_mask", expression_mask)

    mercury_experiment.apply_mask(
        run_name="kinetics_ADP_conc_fits_bgsub",
        dep_variables=["slope", "intercept"],
        save_as="kinetics_ADP_conc_fits_masked",
        mask_names=[
            "initial_rates_r2_mask",
            "initial_rates_positive_mask",
            "standard_curve_r2_mask",
            "expression_mask",
        ],
    )

    kinetics_fits_masked = mercury_experiment.get_run(
        "kinetics_ADP_conc_fits_masked"
    )
    number_initial_rates_mask = filter_number_concentrations(
        kinetics_fits_masked, min_concentrations=5, var_to_check="slope"
    )
    mercury_experiment.set_run(
        "number_initial_rates_mask", number_initial_rates_mask
    )
    mercury_experiment.apply_mask(
        run_name="kinetics_ADP_conc_fits_masked",
        dep_variables=["slope", "intercept", "r_squared"],
        mask_names=["number_initial_rates_mask"],
        save_as="kinetics_ADP_conc_fits_masked",
    )

    # 9. Fit Michaelis-Menten Model
    kinetics_fits_final = mercury_experiment.get_run(
        "kinetics_ADP_conc_fits_masked"
    )
    MM_fits, MM_pred_data = fit_initial_rates_vs_concentration_with_function(
        data=kinetics_fits_final, model_func=mm_model
    )
    mercury_experiment.set_run("kinetics_ADP_MM_fits", MM_fits)
    mercury_experiment.set_run("kinetics_ADP_MM_pred_data", MM_pred_data)

    assert MM_fits is not None
    assert "v_max" in MM_fits.dep_var_type
    assert "K_m" in MM_fits.dep_var_type
    assert "r_squared" in MM_fits.dep_var_type

    # 10. Filter MM fits by R2 and Replicate Count
    MM_fits_mask = filter_r2_cutoff(MM_fits, r2_cutoff=0.8)
    MM_fits_replicate_mask = filter_number_replicates(
        MM_fits, min_replicates=3, var_to_check="K_m"
    )

    mercury_experiment.set_run("MM_R2_mask", MM_fits_mask)
    mercury_experiment.set_run("MM_replicate_mask", MM_fits_replicate_mask)

    mercury_experiment.apply_mask(
        run_name="kinetics_ADP_MM_fits",
        dep_variables=["v_max", "K_m", "r_squared"],
        save_as="kinetics_ADP_MM_fits_masked",
        mask_names=["MM_R2_mask", "MM_replicate_mask"],
    )

    final_mm_fits = mercury_experiment.get_run("kinetics_ADP_MM_fits_masked")
    assert final_mm_fits is not None
    # Ensure some valid fitted chambers survived masking
    v_max_vals = final_mm_fits.dep_var[..., 0]
    valid_fits_count = np.count_nonzero(~np.isnan(v_max_vals))
    assert valid_fits_count > 0, "Expected at least one valid MM fit after filtering"
