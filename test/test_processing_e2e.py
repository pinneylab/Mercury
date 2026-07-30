import pytest
from pathlib import Path
import pandas as pd

from mercury.stitching import ImageStitcher, BackgroundSubtractor
from mercury.processing import Processor
from mercury.processing.experiment import Experiment, Device, DataHandler


@pytest.mark.e2e
@pytest.mark.slow
def test_processing_e2e_pipeline(raw_example_data_dir, tmp_path):
    """
    End-to-end integration test for raw image stitching, background subtraction,
    and feature processing on Hayden's AdkSubset_DnaK dataset.
    """
    if not raw_example_data_dir.exists():
        pytest.skip(
            f"Raw image example data directory not found at {raw_example_data_dir}"
        )

    # 1. Create an isolated work directory with symlinks to raw input data
    work_dir = tmp_path / "processing_run"
    work_dir.mkdir()

    for item in raw_example_data_dir.iterdir():
        (work_dir / item.name).symlink_to(item)

    # 2. Image Stitching
    stitcher = ImageStitcher(work_dir)
    stitcher.stitch_images(
        rotation=1.240,
        get_rotation_from="*brightfield*",
        overlap=None,
        rotation_method="overlaps",
        get_overlap_from="*brightfield*",
        acqui_origin=(True, False),
    )

    stitched_csv = work_dir / "stitched_images" / "stitched_images.csv"
    assert stitched_csv.exists(), "stitched_images.csv was not generated"

    # 3. Background Subtraction
    background_images = [
        work_dir
        / "stitched_images"
        / "2026-06-18_14-20-10_d3_spectra_cyan_2_5_sensitivity_2x2_1_button_quant_background.tif",
        work_dir
        / "stitched_images"
        / "2026-06-18_17-42-48_d3_retra_cyan_1_500_dynamic_range_2x2_4_nadh_background.tif",
        work_dir
        / "stitched_images"
        / "2026-06-18_19-21-01_d3_retra_cyan_1_500_dynamic_range_2x2_4_nadh_background.tif",
    ]
    settings_to_match = [
        "temp",
        "hum",
        "setup",
        "dname",
        "lightsource",
        "channel",
        "exposure",
        "camera_mode",
        "binning",
        "nosepiece",
        "apply_ff_correction",
    ]
    subtractor = BackgroundSubtractor(work_dir)
    subtractor.subtract(
        background_images=background_images,
        settings_to_match=settings_to_match,
    )

    bgsub_csv = work_dir / "bgsub_images" / "bgsub_images.csv"
    assert bgsub_csv.exists(), "bgsub_images.csv was not generated"

    # 4. Experiment Initialization and Device Registration
    experiment = Experiment(work_dir)
    d1 = Device(setup="s1", dname="d3", dims=(32, 56))
    d1.set_pinlist(
        pinlist_path=work_dir / "20260601_Adk_Subset_output_pinlist.csv"
    )
    experiment.add_device(d1)
    data_handler = DataHandler(work_dir)

    # 5. Button Quant Processing
    button_quant_identifiers = [
        "2026-06-18_16-51-57_d3_spectra_cyan_2_5_sensitivity_2x2_1_button_quant"
    ]
    button_quant_image_data = data_handler.get_images(
        identifiers=button_quant_identifiers
    )
    button_quant_processor = Processor(
        experiment=experiment,
        image_data=button_quant_image_data,
        features="button",
    )
    button_quant_processor.set_corners(
        "d3", [(407, 476), (6617, 409), (418, 6776), (6641, 6766)]
    )
    button_quant_df = button_quant_processor.process()
    button_quant_csv = work_dir / "button_quant.csv"
    button_quant_df.to_csv(button_quant_csv)

    assert button_quant_csv.exists()
    assert not button_quant_df.empty
    assert "mean_button" in button_quant_df.columns or "summed_button" in button_quant_df.columns


    # 6. Standard Curve Processing
    standard_images = data_handler.get_images(
        identifiers="2026-06-18_17-44-23_standard_curve_NADPH"
    )
    standard_processor = Processor(
        experiment, image_data=standard_images, features="chamber"
    )
    standard_processor.set_corners(
        "d3", [(408, 467), (6613, 400), (418, 6769), (6639, 6759)]
    )
    standard_processor.set_reference(
        image=work_dir
        / "bgsub_images/2026-06-18_17-44-23_standard_curve_NADPH/2026-06-18_18-16-02_d3_retra_cyan_1_500_dynamic_range_2x2_4_standard_curve_NADPH_7.tif",
        coerce_chamber_center=False,
    )
    standard_df = standard_processor.process(use_reference=True)
    standard_csv = work_dir / "standard_data.csv.bz2"
    standard_df.to_csv(standard_csv, compression="bz2")

    assert standard_csv.exists()
    assert not standard_df.empty

    # 7. Kinetics Processing
    kinetics_images = data_handler.get_images(
        identifiers="2026-06-18_19-24-15_kinetic_series_adk_adp"
    )
    kinetics_processor = Processor(
        experiment, image_data=kinetics_images, features="chamber"
    )
    kinetics_processor.set_corners(
        "d3", [(406, 465), (6615, 401), (412, 6770), (6635, 6763)]
    )
    kinetics_processor.set_reference(
        image=work_dir
        / "bgsub_images/2026-06-18_19-24-15_kinetic_series_adk_adp/2026-06-19_00-24-46_timecourse_7/2026-06-19_01-07-13_d3_retra_cyan_1_500_dynamic_range_2x2_4_kinetic_series_adk_adp_timecourse_7_19.tif",
        coerce_chamber_center=False,
    )
    kinetics_df = kinetics_processor.process(use_reference=True)
    kinetics_csv = work_dir / "kinetics_data.csv.bz2"
    kinetics_df.to_csv(kinetics_csv, compression="bz2")

    assert kinetics_csv.exists()
    assert not kinetics_df.empty
