# Command Line Interface (CLI)

Mercury provides command-line entry points for interactive alignment and processing tasks.

---

## `pick-corners`

The `pick-corners` tool opens an interactive interface to select reference corners on microfluidic chip images for coordinate transformation and alignment.

### Usage

```bash
pick-corners --help
```

### Options

| Flag | Description |
| --- | --- |
| `-i`, `--image` | Path to input raw or stitched chip image file |
| `-o`, `--output` | Path to save corner coordinates JSON configuration |
| `-c`, `--config` | Path to existing chip config YAML/JSON |

### Example

```bash
pick-corners --image data/chip_overview.tif --output config_files/chip_corners.json
```
