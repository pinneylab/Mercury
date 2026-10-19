import unittest
import tempfile
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

from mercury.db_api.data import Data2D, Data3D, Data4D, IndepVars, Meta
from mercury.db_api.units.units import units
from mercury.analysis.export import data_to_dataframe, export_data_to_csv
from mercury.analysis.experiment import MercuryExperiment
from mercury.db_api.mercury_db_api import AbstractMercuryDBAPI


class DummyDBAPI(AbstractMercuryDBAPI):
    def __init__(self, runs_dict=None):
        self._runs = runs_dict or {}

    def get_run_names(self):
        return list(self._runs.keys())

    def get_run(self, run_name: str):
        if run_name not in self._runs:
            raise KeyError(f"Run {run_name} not found")
        return self._runs[run_name]


class TestGeneralExport(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        # 4 chambers across 2 samples: sample_A has 3 replicates, sample_B has 1
        self.chambers = np.array(["1,1", "1,2", "1,3", "1,4"])
        self.samples = np.array(["sample_A", "sample_A", "sample_A", "sample_B"])
        self.concs = np.array([1.0, 5.0]) * units.uM
        self.times = np.array([[0.0, 10.0], [0.0, 10.0]]) * units.s

        self.indep_vars = IndepVars(
            concentration=self.concs,
            chamber_IDs=self.chambers,
            sample_IDs=self.samples,
            button_quant_sum=np.zeros(4) * units.RFU,
            time=self.times,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_data2d_well_level(self):
        # 4 chambers, 2 variables: luminance (RFU) and r_squared (dimensionless)
        dep_var = np.array([
            [100.0, 0.95],
            [120.0, 0.98],
            [np.nan, np.nan],
            [200.0, 0.99],
        ])
        data = Data2D(
            indep_vars=self.indep_vars,
            dep_var=dep_var,
            dep_var_type=["luminance", "r_squared"],
            dep_var_units=[units.RFU, units.dimensionless],
            meta=Meta(),
        )

        df = data_to_dataframe(data, aggregate_by_sample=False)

        self.assertEqual(len(df), 4)
        # Unit included for RFU, omitted for dimensionless
        self.assertIn("luminance (RFU)", df.columns)
        self.assertIn("r_squared", df.columns)
        self.assertNotIn("r_squared (dimensionless)", df.columns)
        self.assertListEqual(list(df["chamber"]), ["1,1", "1,2", "1,3", "1,4"])
        self.assertListEqual(list(df["sample"]), ["sample_A", "sample_A", "sample_A", "sample_B"])
        self.assertEqual(df.loc[0, "luminance (RFU)"], 100.0)
        self.assertTrue(np.isnan(df.loc[2, "luminance (RFU)"]))

    def test_data2d_sample_level_aggregation(self):
        # sample_A has values 100.0, 120.0, and NaN
        # sample_B has value 200.0
        dep_var = np.array([
            [100.0, 0.90],
            [120.0, 0.92],
            [np.nan, np.nan],
            [200.0, 0.99],
        ])
        data = Data2D(
            indep_vars=self.indep_vars,
            dep_var=dep_var,
            dep_var_type=["luminance", "r_squared"],
            dep_var_units=[units.RFU, units.dimensionless],
            meta=Meta(),
        )

        df = data_to_dataframe(
            data,
            aggregate_by_sample=True,
            include_chamber_ids=True,
            agg=("mean", "std"),
        )

        self.assertEqual(len(df), 2)
        self.assertListEqual(list(df["sample"]), ["sample_A", "sample_B"])
        self.assertIn("replicates", df.columns)
        self.assertIn("mean_luminance (RFU)", df.columns)
        self.assertIn("std_luminance (RFU)", df.columns)
        self.assertIn("mean_r_squared", df.columns)
        self.assertIn("chamber_IDs", df.columns)

        # sample_A: 2 valid replicates (100 and 120), mean = 110.0
        sample_a_row = df[df["sample"] == "sample_A"].iloc[0]
        self.assertEqual(sample_a_row["replicates"], 2)
        self.assertAlmostEqual(sample_a_row["mean_luminance (RFU)"], 110.0)
        self.assertAlmostEqual(sample_a_row["mean_r_squared"], 0.91)
        self.assertEqual(sample_a_row["chamber_IDs"], "1,1; 1,2")

        # sample_B: 1 replicate
        sample_b_row = df[df["sample"] == "sample_B"].iloc[0]
        self.assertEqual(sample_b_row["replicates"], 1)
        self.assertEqual(sample_b_row["mean_luminance (RFU)"], 200.0)
        self.assertTrue(np.isnan(sample_b_row["std_luminance (RFU)"]))
        self.assertEqual(sample_b_row["chamber_IDs"], "1,4")

    def test_data3d_well_and_sample_level(self):
        # 2 concentrations, 4 chambers, 1 dep var
        dep_var = np.ones((2, 4, 1))
        dep_var[0, :, 0] = [10.0, 20.0, 30.0, 40.0]  # conc 1.0 uM
        dep_var[1, :, 0] = [50.0, 60.0, 70.0, 80.0]  # conc 5.0 uM

        data = Data3D(
            indep_vars=self.indep_vars,
            dep_var=dep_var,
            dep_var_type=["slope"],
            dep_var_units=[units.RFU / units.s],
            meta=Meta(),
        )

        # Chamber level
        df_well = data_to_dataframe(data, aggregate_by_sample=False)
        self.assertEqual(len(df_well), 8)
        self.assertIn("concentration (μM)", df_well.columns)
        self.assertIn("chamber", df_well.columns)
        self.assertIn("sample", df_well.columns)
        self.assertIn("slope (RFU / s)", df_well.columns)

        # Sample level
        df_sample = data_to_dataframe(data, aggregate_by_sample=True, agg=["mean"])
        self.assertEqual(len(df_sample), 4)  # 2 concs x 2 samples
        self.assertIn("concentration (μM)", df_sample.columns)
        self.assertIn("sample", df_sample.columns)
        self.assertIn("mean_slope (RFU / s)", df_sample.columns)

        # Check sample_A at conc 1.0 uM: (10 + 20 + 30) / 3 = 20.0
        row_c1_a = df_sample[(df_sample["concentration (μM)"] == 1.0) & (df_sample["sample"] == "sample_A")].iloc[0]
        self.assertAlmostEqual(row_c1_a["mean_slope (RFU / s)"], 20.0)
        self.assertEqual(row_c1_a["replicates"], 3)

    def test_data4d_well_and_sample_level(self):
        # 2 concentrations, 2 timepoints, 4 chambers, 1 dep var
        dep_var = np.arange(16.0).reshape(2, 2, 4, 1)

        data = Data4D(
            indep_vars=self.indep_vars,
            dep_var=dep_var,
            dep_var_type=["luminance"],
            dep_var_units=[units.RFU],
            meta=Meta(),
        )

        df_well = data_to_dataframe(data, aggregate_by_sample=False)
        self.assertEqual(len(df_well), 16)
        self.assertIn("concentration (μM)", df_well.columns)
        self.assertIn("time (s)", df_well.columns)
        self.assertIn("chamber", df_well.columns)
        self.assertIn("sample", df_well.columns)
        self.assertIn("luminance (RFU)", df_well.columns)

        df_sample = data_to_dataframe(data, aggregate_by_sample=True)
        # 2 concs x 2 timepoints x 2 samples = 8 rows
        self.assertEqual(len(df_sample), 8)
        self.assertIn("mean_luminance (RFU)", df_sample.columns)
        self.assertIn("std_luminance (RFU)", df_sample.columns)

    def test_export_options_and_formatting(self):
        dep_var = np.array([[100.0], [200.0], [300.0], [400.0]])
        data = Data2D(
            indep_vars=self.indep_vars,
            dep_var=dep_var,
            dep_var_type=["v_max"],
            dep_var_units=[units.uM / units.s],
            meta=Meta(),
        )

        # include_units=False
        df_no_units = data_to_dataframe(data, include_units=False)
        self.assertIn("v_max", df_no_units.columns)
        self.assertNotIn("v_max (μM / s)", df_no_units.columns)

        # Custom agg alias 'avg'
        df_avg = data_to_dataframe(data, aggregate_by_sample=True, agg=["avg", "min", "max"])
        self.assertIn("avg_v_max (μM / s)", df_avg.columns)
        self.assertIn("min_v_max (μM / s)", df_avg.columns)
        self.assertIn("max_v_max (μM / s)", df_avg.columns)

        # Custom column headers
        df_custom_cols = data_to_dataframe(data, chamber_col="well_id", sample_col="variant")
        self.assertIn("well_id", df_custom_cols.columns)
        self.assertIn("variant", df_custom_cols.columns)

    def test_export_data_to_csv_file(self):
        dep_var = np.array([[100.0], [200.0], [300.0], [400.0]])
        data = Data2D(
            indep_vars=self.indep_vars,
            dep_var=dep_var,
            dep_var_type=["v_max"],
            dep_var_units=[units.uM / units.s],
            meta=Meta(),
        )

        out_csv = Path(self.temp_dir) / "subdir" / "test_export.csv"
        df = export_data_to_csv(data, file_path=out_csv, aggregate_by_sample=True)

        self.assertTrue(out_csv.exists())
        loaded_df = pd.read_csv(out_csv)
        self.assertEqual(len(loaded_df), 2)
        self.assertListEqual(list(loaded_df["sample"]), list(df["sample"]))

    def test_mercury_experiment_integration(self):
        dep_var = np.array([[10.0], [20.0], [30.0], [40.0]])
        data = Data2D(
            indep_vars=self.indep_vars,
            dep_var=dep_var,
            dep_var_type=["enzyme_conc"],
            dep_var_units=[units.nM],
            meta=Meta(description="enzyme_quant_run"),
        )
        db = DummyDBAPI({"enzymes": data})
        exp = MercuryExperiment(db)

        # 1. to_dataframe without saving
        df_in_memory = exp.to_dataframe("enzymes", aggregate_by_sample=False)
        self.assertEqual(len(df_in_memory), 4)

        # 2. export_run with default filename and output_dir
        df_exported = exp.export_run(
            "enzymes",
            output_dir=self.temp_dir,
            aggregate_by_sample=False,
        )
        expected_file = Path(self.temp_dir) / "enzymes_chamber_data.csv"
        self.assertTrue(expected_file.exists())
        self.assertEqual(len(df_exported), 4)

        # 3. export_run with aggregate_by_sample
        df_sample = exp.export_run(
            "enzymes",
            output_dir=self.temp_dir,
            aggregate_by_sample=True,
        )
        expected_sample_file = Path(self.temp_dir) / "enzymes_sample_data.csv"
        self.assertTrue(expected_sample_file.exists())
        self.assertEqual(len(df_sample), 2)

        # 4. export_to_csv alias with custom file_name
        exp.export_to_csv(
            "enzymes",
            output_dir=self.temp_dir,
            file_name="custom_name.csv",
        )
        self.assertTrue((Path(self.temp_dir) / "custom_name.csv").exists())

    def test_data_dataclass_methods(self):
        dep_var = np.array([[1.0], [2.0], [3.0], [4.0]])
        data = Data2D(
            indep_vars=self.indep_vars,
            dep_var=dep_var,
            dep_var_type=["val"],
            dep_var_units=[units.dimensionless],
            meta=Meta(),
        )

        df = data.to_dataframe()
        self.assertEqual(len(df), 4)

        out_csv = Path(self.temp_dir) / "data2d_direct.csv"
        data.export_to_csv(out_csv)
        self.assertTrue(out_csv.exists())


if __name__ == "__main__":
    unittest.main()
