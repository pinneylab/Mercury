import os
from pathlib import Path
import pytest


@pytest.fixture(scope="session")
def project_root():
    """Returns the root directory of the mercury repository."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def workspace_root(project_root):
    """Returns the workspace root directory (MiE_code_release)."""
    return project_root.parent


@pytest.fixture(scope="session")
def raw_example_data_dir(workspace_root):
    """
    Returns the path to raw image example data for Hayden's experiment.
    Can be overridden by setting the MERCURY_TEST_DATA_DIR environment variable.
    """
    custom_path = os.environ.get("MERCURY_TEST_DATA_DIR") or os.environ.get("HTBAM_TEST_DATA_DIR")
    if custom_path:
        return Path(custom_path)
    return workspace_root / "example_data" / "hayden" / "20260618_AdkSubset_DnaK"


@pytest.fixture(scope="session")
def reference_csv_dir(project_root):
    """Returns the path to pre-computed reference CSV files for standalone analysis tests."""
    return project_root / "test" / "test_data" / "adk_example_subset"

