# Development & Testing Workflow

This guide details the standard development, branching, and testing practices for developers contributing to the Mercury codebase.

---

## 1. Environment Setup

Mercury uses `poetry` for dependency management, but can also be installed in editable mode with standard `pip`.

### Activate or Create Your Development Environment

```bash
# Recommended: Python 3.12 environment
conda activate mercury_dev_local
```

### Install in Editable Mode

Make sure you are in the root of the repository:

```bash
cd /path/to/Mercury

# Option A: Editable install via pip
pip install -e .

# Option B: Install via Poetry (installs all dev & test dependencies)
poetry install
```

When installed in editable mode (`-e`), changes you make to files under `src/mercury/` immediately take effect without re-installing.

---

## 2. Branching Strategy

The `main` branch is protected. All modifications—whether bug fixes, performance improvements, or new features—must be developed on a dedicated branch and merged via Pull Request.

### Create a Descriptive Feature Branch

Always branch off the latest `main`:

```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

Naming conventions:
- `feature/<name>` for new functionality (e.g. `feature/ic50-fitting`)
- `fix/<issue>` for bug fixes (e.g. `fix/stitching-overlap-bounds`)
- `refactor/<module>` for internal code reorganizations

---

## 3. Implementing Changes & Writing Tests

Every change to `mercury` must be verified with automated tests under the `test/` directory.

### The `test/` Directory Structure

```text
Mercury/test/
├── conftest.py              # Shared fixtures (reference datasets, test paths)
├── test_data/               # Small mock datasets (e.g. adk_example_subset)
├── test_analysis_e2e.py     # End-to-end integration tests
├── test_stitching.py        # Raster & stitching unit tests
└── test_your_feature.py     # New test module for your code
```

### Writing Tests with PyTest

Create a dedicated test file under `test/` (e.g., `test/test_new_feature.py`).

```python
import pytest
import numpy as np
from mercury.analysis.experiment import MercuryExperiment
from mercury.db_api.mercury_db_api import LocalMercuryDBAPI

def test_feature_behavior(reference_csv_dir):
    """
    Test that the new calculation handles normal inputs correctly.
    Uses the pre-packaged reference_csv_dir fixture from conftest.py.
    """
    button_quant = reference_csv_dir / "button_quant.csv"
    assert button_quant.exists()
    
    # Execute function under test
    # ...
    # Verify outputs
    assert True
```

### PyTest Markers

Mark slow or integration-level tests using pytest markers defined in `pyproject.toml`:

```python
@pytest.mark.e2e
def test_full_pipeline(reference_csv_dir):
    ...

@pytest.mark.slow
def test_large_image_processing():
    ...
```

### Using LLMs to Accelerate Test Writing

LLMs (such as Antigravity/Gemini) are very effective for generating comprehensive PyTest test suites:
1. Provide the LLM with your new function or class signature and docstring.
2. Ask it to generate:
   - **Happy-path unit tests** verifying expected mathematical behavior.
   - **Edge-case tests** (e.g., all-NaN chambers, empty DataFrames, zero concentrations, division by zero).
   - **Mock fixture usage** relying on existing `test/conftest.py` fixtures.
3. Review and run the generated tests locally to ensure they pass.

### Running Tests Locally

Run the test suite before committing:

```bash
# Run all tests
pytest test/

# Run a specific test file
pytest test/test_new_feature.py

# Run only fast unit tests (skipping slow / e2e tests)
pytest test/ -m "not slow and not e2e"

# Run with verbose output and print statements
pytest test/ -v -s
```

---

## 4. Pushing and Merging into `main`

### Commit Your Changes

```bash
git add src/ test/
git commit -m "Add feature XYZ with unit tests"
```

### Push to GitHub

Push your feature branch to the remote repository:

```bash
git push -u origin feature/your-feature-name
```

### Open a Pull Request (PR)

1. Navigate to [github.com/pinneylab/Mercury](https://github.com/pinneylab/Mercury).
2. GitHub will prompt you to create a Pull Request from your branch.
3. In the PR description:
   - Summarize what changed and why.
   - State which test files were added or modified.
   - Mention any potential breaking changes to downstream template notebooks (`mercury_notebooks`).

### Continuous Integration (CI) & Review

- **Automated CI**: GitHub Actions (`.github/workflows/ci.yml`) will automatically trigger and run the test suite across supported Python environments.
- **Code Review**: Because `main` is a protected branch, at least one other lab member must review and approve your PR.
- **Merge**: Once CI passes and approval is received, use **Squash and merge** (or rebase merge) to merge your branch into `main`, keeping git history clean.
