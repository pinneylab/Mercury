"""Run record: the resolved lockfile, the append-only event log, and provenance.

processing.resolved.yaml holds the *state* (what each stage decided/produced), so a run
can be replayed without the agent. processing.log.jsonl holds the *history* (every
stage run, decision, escalation and note), so you can see how the state was reached.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from mercury.pipeline import config as cfgmod


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def plain(obj: Any) -> Any:
    """Convert numpy/pandas/Path values to YAML/JSON-safe builtins (NaN -> None)."""
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if f != f else round(f, 6)
    if isinstance(obj, np.ndarray):
        return plain(obj.tolist())
    return obj


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def mercury_provenance() -> dict:
    """Version, git commit/branch and dirty state of the Mercury checkout being run."""
    import mercury

    pkg = Path(mercury.__file__).resolve().parent
    out = {"path": str(pkg), "commit": None, "branch": None, "dirty": None, "diff_sha256": None}
    try:
        from importlib.metadata import version
        out["version"] = version("mercury-kinetics")
    except Exception:
        out["version"] = None

    def git(*args, cwd=pkg):
        return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True).stdout

    try:
        top = Path(git("rev-parse", "--show-toplevel").strip())
        out["commit"] = git("rev-parse", "HEAD", cwd=top).strip()
        out["branch"] = git("rev-parse", "--abbrev-ref", "HEAD", cwd=top).strip()
        # Package source = everything under the mercury package dir, tracked or not
        # (new modules are untracked until committed). .gitignore'd files (pycache) excluded.
        rel = str(pkg.relative_to(top))
        status = git("status", "--porcelain", "--untracked-files=all", "--", rel, cwd=top)
        out["dirty"] = bool(status.strip())
        if out["dirty"]:
            h = hashlib.sha256(git("diff", "HEAD", "--", rel, cwd=top).encode())
            untracked = sorted(l[3:] for l in status.splitlines() if l.startswith("??"))
            for f in untracked:
                h.update(f.encode())
                h.update((top / f).read_bytes())
            out["diff_sha256"] = h.hexdigest()
            out["untracked"] = untracked
    except Exception:
        pass
    out["python"] = platform.python_version()
    return out


class Run:
    """Handle on one experiment's config, resolved lockfile and log."""

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path).resolve()
        self.cfg = cfgmod.load_config(self.config_path)
        self.root = Path(self.cfg["root"]).expanduser()
        self.resolved_path = cfgmod.resolved_path(self.config_path)
        self.log_path = cfgmod.log_path(self.config_path)
        self.resolved = self._load_resolved()
        self.provenance = mercury_provenance()

    # ----------------------------------------------------------------- resolved
    def _load_resolved(self) -> dict:
        if self.resolved_path.exists():
            return yaml.safe_load(self.resolved_path.read_text()) or {}
        return {}

    def save(self) -> None:
        r = self.resolved
        r.setdefault("meta", {})
        r["meta"].update({
            "config": str(self.config_path),
            "config_sha256": cfgmod.sha256_of({k: v for k, v in self.cfg.items() if not k.startswith("_")}),
            "agent_overrides": cfgmod.load_overrides(self.config_path),
            "root": str(self.root),
            "updated": now(),
            "mercury": self.provenance,
        })
        r["meta"].setdefault("created", now())
        # stable top-level order
        order = ["meta", "stages", "preflight", "stitching", "background", "series", "escalations"]
        out = {k: r[k] for k in order if k in r}
        out.update({k: v for k, v in r.items() if k not in out})
        self.resolved_path.write_text(yaml.safe_dump(plain(out), sort_keys=False, width=120))

    def stage(self, name: str) -> dict:
        return self.resolved.setdefault("stages", {}).get(name, {})

    def set_stage(self, name: str, status: str, **info) -> None:
        self.resolved.setdefault("stages", {})[name] = {
            "status": status, "finished": now(),
            "mercury_commit": self.provenance.get("commit"),
            "mercury_dirty": self.provenance.get("dirty"),
            **info,
        }

    def series_record(self, name: str) -> dict:
        return self.resolved.setdefault("series", {}).setdefault(name, {})

    # ----------------------------------------------------------------- log
    def log(self, event: str, **details) -> None:
        entry = {"at": now(), "event": event, **plain(details)}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def escalate(self, stage: str, item: str, reason: str, request: str) -> dict:
        esc = {"stage": stage, "item": item, "reason": reason, "request": request,
               "at": now(), "resolved": False}
        lst = self.resolved.setdefault("escalations", [])
        # one open escalation per (stage, item); replace with the newest
        lst[:] = [e for e in lst if not (e["stage"] == stage and e["item"] == item and not e["resolved"])]
        lst.append(esc)
        self.log("escalation", **esc)
        return esc

    def close_escalations(self, stage: str, item: Optional[str] = None) -> None:
        for e in self.resolved.get("escalations", []):
            if e["stage"] == stage and (item is None or e["item"] == item) and not e["resolved"]:
                e["resolved"] = True
                e["resolved_at"] = now()
