"""mercury-run: staged, recorded HT-BAM image processing.

    python -m mercury.pipeline <command> --config processing.yaml [options]

Commands
    preflight | calibrate | stitch | bgsub | corners | reference | map   run one stage
    all          run stages in order, stopping at the first that does not pass
    status       show stage states and open escalations
    override     record an agent override (allowlisted keys only, reason required)
    note         append a free-text note to the log
    template     print a starter processing.yaml

Exit codes: 0 pass/warn, 2 needs a human (see escalations), 1 error.
The last stdout line of every command is `RESULT <json>` for machine parsing.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import yaml

from mercury.pipeline import config as cfgmod
from mercury.pipeline.record import Run, plain
from mercury.pipeline.stages import ORDER, STAGES, StageError

EXIT = {"pass": 0, "warn": 0, "needs_human": 2, "fail": 1}

TEMPLATE = Path(__file__).with_name("processing_template.yaml")


def _emit(payload: dict) -> None:
    print("RESULT " + json.dumps(plain(payload), default=str))


def _print_result(res: dict) -> None:
    print(f"[{res['stage']}] {res['status'].upper()}")
    print(yaml.safe_dump(res.get("summary", {}), sort_keys=False, width=120).rstrip())


def run_stage(run: Run, name: str, args) -> dict:
    if name != "preflight" and run.provenance.get("dirty") and not args.allow_dirty:
        msg = ("Mercury has uncommitted changes "
               f"({run.provenance.get('path')}). Outputs would not be reproducible. "
               "Commit first, or pass --allow-dirty (recorded) for a test run.")
        run.log("stage_refused", stage=name, reason=msg)
        return {"stage": name, "status": "fail", "summary": {"error": msg}}
    run.log("stage_start", stage=name, redo=args.redo, overwrite=args.overwrite,
            allow_dirty=args.allow_dirty, mercury=run.provenance)
    try:
        res = STAGES[name](run, redo=args.redo, overwrite=args.overwrite)
    except StageError as e:
        res = {"stage": name, "status": "fail", "summary": {"error": str(e)}}
    except Exception as e:  # unexpected: record and surface
        res = {"stage": name, "status": "fail",
               "summary": {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}}
    if res["status"] == "fail":
        run.set_stage(name, "fail", error=res["summary"].get("error"))
    run.log("stage_end", stage=name, status=res["status"])
    run.save()
    return res


def cmd_status(run: Run) -> dict:
    stages = run.resolved.get("stages", {})
    print(f"config:   {run.config_path}")
    print(f"resolved: {run.resolved_path}")
    print(f"mercury:  {run.provenance.get('branch')} {str(run.provenance.get('commit'))[:10]}"
          f"{' (dirty)' if run.provenance.get('dirty') else ''}")
    for s in ORDER:
        st = stages.get(s, {})
        print(f"  {s:10s} {st.get('status', '-'):12s} {st.get('finished', '')}")
    open_esc = [e for e in run.resolved.get("escalations", []) if not e.get("resolved")]
    if open_esc:
        print("open escalations:")
        for e in open_esc:
            print(f"  - [{e['stage']}] {e['item']}: {e['reason']}\n      -> {e['request']}")
    nxt = next((s for s in ORDER if stages.get(s, {}).get("status") not in ("pass", "warn")), None)
    return {"stages": {s: stages.get(s, {}).get("status") for s in ORDER},
            "open_escalations": open_esc, "next": nxt}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="mercury-run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=ORDER + ["all", "status", "override", "note", "template"])
    p.add_argument("--config", type=Path, help="processing.yaml")
    p.add_argument("--redo", action="store_true", help="re-run even if inputs are unchanged")
    p.add_argument("--overwrite", action="store_true",
                   help="allow replacing outputs not produced by this run record (ask the user first)")
    p.add_argument("--allow-dirty", action="store_true",
                   help="run with uncommitted Mercury changes (recorded; for testing only)")
    p.add_argument("--key", help="override: dotted key, e.g. series.nadh.reference.pick")
    p.add_argument("--value", help="override: YAML value, e.g. max or '[[1, 2], [3, 4]]'")
    p.add_argument("--reason", help="override/note: why")
    args = p.parse_args(argv)

    if args.command == "template":
        print(TEMPLATE.read_text())
        return 0
    if args.config is None:
        p.error("--config is required")

    try:
        run = Run(args.config)
    except cfgmod.ConfigError as e:
        print(str(e))
        _emit({"command": args.command, "status": "needs_human", "error": str(e)})
        return 2

    if args.command == "status":
        _emit({"command": "status", **cmd_status(run)})
        return 0

    if args.command == "override":
        if args.key is None or args.value is None:
            p.error("override needs --key, --value and --reason")
        try:
            entry = cfgmod.add_override(args.config, args.key, yaml.safe_load(args.value), args.reason or "")
        except cfgmod.ConfigError as e:
            print(str(e))
            _emit({"command": "override", "status": "refused", "error": str(e)})
            return 2
        run.log("override", **entry)
        print(f"override recorded: {entry['key']} = {entry['value']!r}")
        _emit({"command": "override", "status": "pass", **entry})
        return 0

    if args.command == "note":
        run.log("note", text=args.reason or "")
        _emit({"command": "note", "status": "pass"})
        return 0

    names = ORDER if args.command == "all" else [args.command]
    results = []
    for name in names:
        res = run_stage(run, name, args)
        _print_result(res)
        results.append({"stage": name, "status": res["status"]})
        if res["status"] not in ("pass", "warn"):
            break
    final = results[-1]["status"]
    _emit({"command": args.command, "status": final, "stages": results,
           "resolved": str(run.resolved_path),
           "open_escalations": [e for e in run.resolved.get("escalations", []) if not e.get("resolved")]})
    return EXIT[final]


if __name__ == "__main__":
    sys.exit(main())
