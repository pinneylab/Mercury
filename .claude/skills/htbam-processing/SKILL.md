---
name: htbam-processing
description: Run HT-BAM image processing (stitch, background-subtract, corner check, feature finding, quantification) for one experiment with the mercury-run CLI, recording every decision. Use when asked to process, re-process, or check processing of a scope_data/YYYYMMDD experiment, or to set up a processing.yaml.
---

# HT-BAM processing with mercury-run

Processing is done only by the `mercury-run` CLI (`mercury.pipeline`). Your job is to
run its stages, read their results, make the small adjustments it allows, and hand
anything else to the user. The CLI writes the record; you never write it by hand.

## Files (all in the notebook entry folder, next to each other)

| File | Who writes it | What it is |
|---|---|---|
| `processing.yaml` | the user | Intent: root, devices, series, QC thresholds. **You never edit it** unless the user explicitly asks in their current message. |
| `processing.agent.yaml` | the CLI, via `override` | Your tier-1 adjustments, each with a reason. Never edit by hand. |
| `processing.resolved.yaml` | the CLI | The lockfile: every parameter actually used, with its source, plus QC. |
| `processing.log.jsonl` | the CLI | Append-only history of stages, decisions, escalations, notes. |

Large outputs go under the experiment root: `qc/` (seam/corner tables, logs, ROI cache),
`stitched_images/`, `bgsub_images/`, `summary_images/`, `processed/` (final CSVs).

## Command

```bash
conda run --no-capture-output -n mercury python -m mercury.pipeline <command> --config <entry>/processing.yaml
```
(`mercury-run <command> ...` is equivalent once the package is reinstalled.)

Commands: `preflight`, `calibrate`, `stitch`, `bgsub`, `corners`, `reference`, `map`,
`all` (in order, stops at the first non-pass), `status`, `override`, `note`, `template`.

The last stdout line is always `RESULT {json}`. Exit code: **0** pass/warn, **2** needs a
human, **1** error. Library output goes to `<root>/qc/logs/`, not the terminal. Read
those logs only when diagnosing a failure.

## Procedure

1. **No config yet?** Run `template` and draft `processing.yaml` with the user. Fill in only
   what is in their notebook, their old `process.ipynb`, or the data folder (imaging.csv,
   series_index.json). Do not invent corners, rotation, or which images are references;
   ask. Test rasters should have visible features (button quant, brightfield, high NADH),
   one or more per objective (nosepiece).
2. `status` to see where things stand.
3. `preflight`. Fix nothing yourself; relay errors (ambiguous backgrounds, unknown
   identifiers, missing pinlist) to the user.
4. Run the remaining stages one at a time (or `all`). After each, read the `RESULT` line:
   - **pass**: continue.
   - **warn**: continue, and include the warning in your report.
   - **needs_human** (exit 2): stop. Report the escalation's `reason` and `request`
     verbatim (e.g. "Run `pick-corners <image>` and set ..."). Do not work around it.
   - **fail** (exit 1): read `error`. If it is a prerequisite, input, or overwrite problem,
     tell the user. If it is a code error (traceback), report it; do not patch Mercury.
5. **Before `map`:** proceed on your own only if every earlier stage is `pass` (not `warn`)
   and you made no overrides in this session. Otherwise summarize what was decided (see
   Reporting) and ask the user to approve before running `map`.
6. Finish with the report below.

Stages skip themselves when their inputs (including the Mercury code version) are
unchanged; use `--redo` only to recompute deliberately.

## What you may change (tier 1): only via `override`

```bash
... override --config processing.yaml --key <dotted key> --value '<yaml>' --reason "<why>"
```
Allowed keys (the CLI refuses everything else):
- `stitching.test_rasters`: e.g. when a test raster is "unmeasurable" (too dim).
- `series.<name>.reference` / `.reference.pick` / `.reference.by` / `.reference.image`:
  e.g. `pick: max, by: concentration` for a standard curve, `pick: first, by: time_index`
  for kinetics.
- `devices.<d>.corners`, `series.<name>.corners.<d>`: **only** to enter corners the user
  picked and gave you. Never type in corners you estimated yourself; the `corners` stage
  refines guesses automatically.

The calibrate and corners stages already refit rotation, overlap and corners within the
QC gates. You do not need to, and must not, tune those by hand.

Use `note --reason "<text>"` to log any judgement call that is not an override.

## Needs the user (tier 2): ask, never do

- Changing any QC threshold (`qc.*`). **Never relax a gate to make a stage pass.**
- `features`, `coerce_chamber_center`, `identifiers`, `devices`, background matching,
  excluding images or series: anything that changes which data are reported, or how.
- `--overwrite` (replacing outputs not made by this record, e.g. a hand-run stitch).
- `--allow-dirty` (running with uncommitted Mercury changes). Acceptable for a test run
  the user asked for; say so in the report.
- Re-running a stage whose outputs were already used downstream in the notebook.

## Never (tier 3)

- Edit, move, or delete anything in `raw_images/`, `stitched_images/`, `bgsub_images/`
  or `processed/` yourself. Only the CLI writes there.
- Edit `processing.resolved.yaml`, `processing.agent.yaml`, or the log by hand.
- Modify Mercury source during a run. If logic needs to change, stop and propose it as a
  separate Mercury change with tests; the run is redone on the new commit.
- Produce final numbers with ad-hoc scripts. Exploration is fine (scratchpad), but final
  outputs come only from `map`.
- Commit or push unless the user asks.

## Reading the QC

- **calibrate**: `along_median_abs` / `across_median_abs` are seam mismatches in px at
  the chosen rotation / overlap (gate `qc.seam_max_residual_px`, default 1 px). Along is
  rotation error, across is overlap error. `source: refit` means the user's value failed
  and was re-estimated. `unmeasurable` means the raster had too few correlated seams;
  pick a brighter test raster.
- **corners**: `shift_first_px` is how far the starting guess was from where the
  finders put features. A pass needs the final shift <= `qc.corner_max_shift_px` and the
  ring check: each side's edge row must look like a unit cell and the ring outside must
  not (this catches a grid shifted by a whole row or column). Per-stamp scatter of
  4-7 px is normal finder noise. Details per stamp: `qc/corners/<series>_<device>.csv`.
- **bgsub**: `n_without_background` images were subtracted with zeros, which is worth
  reporting.

## Reporting

End with:
1. Stage table (`status` output).
2. Decisions with sources: rotation (user/refit and value), overlap per objective, corners
   per series/device (guess -> final, shift), reference images and the rule that chose them.
3. Warnings and any open escalations, with the exact request.
4. Outputs: paths, rows, images, sha256 (from the resolved file).
5. Overrides and notes you made, and whether `--allow-dirty` / `--overwrite` were used.
Point to `processing.resolved.yaml` as the record. Commit only if asked.
