# Releases, PyPI & Documentation Deployment

This guide covers how to bump versions, build distribution artifacts, publish official releases to PyPI, and deploy documentation updates to the lab's GitHub Pages site.

---

## 1. Creating a Release

When new features or bug fixes on `main` are ready to be distributed to lab members and the public, follow the release lifecycle below.

### Step 1: Version Bumping

Mercury follows [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`).
Update the version number in [`pyproject.toml`](file:///home/freitas/workspace/Mercury_Code_Development/Mercury/pyproject.toml):

```toml
[tool.poetry]
name = "mercury-kinetics"
version = "2.0.2"   # <-- Update this version string
```

Commit this version bump to `main`:

```bash
git add pyproject.toml
git commit -m "Bump version to 2.0.2"
git push origin main
```

### Step 2: Build Release Packages Locally

Verify that the packaging build succeeds and produces clean distribution artifacts:

```bash
# Ensure build tools are installed
pip install --upgrade build

# Clean any existing artifacts in dist/
rm -rf dist/

# Build sdist (.tar.gz) and wheel (.whl)
python -m build
```

This generates two files in `dist/`:
- `dist/mercury_kinetics-2.0.2.tar.gz` (Source distribution)
- `dist/mercury_kinetics-2.0.2-py3-none-any.whl` (Built wheel)

Test installing the newly built wheel into a fresh virtual environment:

```bash
pip install dist/mercury_kinetics-2.0.2-py3-none-any.whl
python -c "import mercury; print(mercury.__version__)"
```

### Step 3: Tag the Release in Git

Tag the commit corresponding to the release:

```bash
git tag -a v2.0.2 -m "Release v2.0.2: summary of major changes"
git push origin v2.0.2
```

### Step 4: Create a GitHub Release

1. Go to [github.com/pinneylab/Mercury/releases](https://github.com/pinneylab/Mercury/releases).
2. Click **Draft a new release**.
3. Select your newly pushed tag (e.g. `v2.0.2`).
4. Title the release (e.g. `Release v2.0.2`).
5. Add release notes detailing what was added or fixed.
6. (Optional) Drag and drop the `.whl` and `.tar.gz` files from `dist/` into the release binaries box.
7. Click **Publish release**.

---

## 2. Automated PyPI Publishing

Mercury uses **GitHub Trusted Publishing** (OIDC) to automatically deploy releases to PyPI without needing hardcoded API tokens.

### How it Works

The GitHub Actions workflow [`.github/workflows/publish.yml`](file:///home/freitas/workspace/Mercury_Code_Development/Mercury/.github/workflows/publish.yml) triggers automatically whenever a new GitHub Release is published (`types: [created]`):

```yaml
name: Upload Python Package to PyPI when a Release is Created

on:
  release:
    types: [created]

jobs:
  pypi-publish:
    name: Publish release to PyPI
    runs-on: ubuntu-latest
    environment:
      name: pypi
      url: https://pypi.org/p/mercury-kinetics
    permissions:
      id-token: write  # OpenID Connect token for PyPI authentication
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.12"
      - name: Build package
        run: |
          pip install build
          python -m build
      - name: Publish package distributions to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
```

### Monitoring the Deployment

1. Go to the **Actions** tab on GitHub: [github.com/pinneylab/Mercury/actions](https://github.com/pinneylab/Mercury/actions).
2. Click on the running `Upload Python Package to PyPI` workflow.
3. Once completed (usually ~1–2 minutes), check the published package at:
   [https://pypi.org/project/mercury-kinetics/](https://pypi.org/project/mercury-kinetics/)

Lab members and external users can then install the update via:

```bash
pip install --upgrade mercury-kinetics
```

---

## 3. Creating & Publishing Documentation

Mercury's documentation is powered by [MkDocs](https://www.mkdocs.org/) with the `readthedocs` theme and Python docstring parsing via `mkdocstrings`.

### Documentation Layout

```text
Mercury/
├── mkdocs.yml              # Site navigation, plugins, and metadata
└── docs/
    ├── index.md            # Homepage
    ├── quickstart.md       # Quickstart guide
    ├── guides/             # User & notebook walkthrough guides
    │   ├── notebook_analysis.md
    │   └── notebook_processing.md
    ├── development/        # Developer documentation
    ├── images/             # PNG figures and Mermaid (.mmd) source files
    └── css/custom.css      # Custom stylesheet overrides
```

### Writing and Previewing Locally

Activate your `mkdocs` environment and start the live development server:

```bash
conda activate mkdocs
cd /path/to/Mercury

mkdocs serve
```

Open `http://127.0.0.1:8000/` in your browser. The server watches all markdown files, diagrams, and stylesheets, automatically hot-reloading changes in real time.

### Updating Pipeline Diagrams

Diagrams are stored as Mermaid source files (`.mmd`) in `docs/images/`. If you modify any node or step:

```bash
cd docs/images
./render.sh
```

This compiles high-resolution, 2× scale PNGs (`analysis_pipeline.png`, `processing_pipeline.png`) using `@mermaid-js/mermaid-cli`.

### Building and Validating the Site

Before publishing, verify that all internal links, docstrings, and images compile without error:

```bash
mkdocs build
```

### Publishing to GitHub Pages (`pinneylab.github.io/Mercury`)

To push your latest documentation live to GitHub Pages, run:

```bash
cd /path/to/Mercury
mkdocs gh-deploy
```

**What this does behind the scenes:**
1. Compiles the markdown documentation into static HTML.
2. Switches to a clean `gh-pages` branch.
3. Commits the built static site and pushes it to `origin/gh-pages`.
4. GitHub Pages immediately detects the commit and serves it at [pinneylab.github.io/Mercury/](https://pinneylab.github.io/Mercury/).
