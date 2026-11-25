"""Processing config: load, validate, apply agent overrides.

Three files live next to each other (normally in the notebook entry, under git):

    processing.yaml           user intent; written by a person, never by the agent
    processing.agent.yaml     agent overrides; only keys in AGENT_OVERRIDABLE, each with a reason
                              (written via `mercury-run override`, never by hand)
    processing.resolved.yaml  what was actually used (see record.py); the lockfile

The effective config is processing.yaml with processing.agent.yaml applied on top.
"""

from __future__ import annotations

import copy
import fnmatch
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

FEATURES = ("chamber", "button", "all")
PICKS = ("brightest", "first", "last", "max", "min", "image")

DEFAULT_SETTINGS_TO_MATCH = [
    "temp", "hum", "setup", "dname", "lightsource", "channel", "exposure",
    "camera_mode", "binning", "nosepiece", "apply_ff_correction",
]

DEFAULTS: Dict[str, Any] = {
    "version": 1,
    "acqui_origin": [True, False],
    "n_workers": None,
    "output_dir": "processed",
    "stitching": {
        "rotation": None,
        "overlap_by_nosepiece": {},
        "test_rasters": [],
        "ff_correction": [],
        "method": "cut",
    },
    "background": {
        "images": "*background*",
        "settings_to_match": DEFAULT_SETTINGS_TO_MATCH,
    },
    "devices": {},
    "series": [],
    "qc": {
        "seam_max_residual_px": 1.0,
        "seam_min_inliers": 5,
        "corner_max_shift_px": 3.0,
        "corner_max_residual_px": 10.0,
        "corner_max_iter": 3,
    },
}

SERIES_DEFAULTS = {
    "features": "chamber",
    "devices": None,
    "corners": {},
    "reference": {"pick": "brightest"},
    "coerce_chamber_center": False,
    "save_summary_images": False,
}

# Keys the agent may change without asking (tier 1). Patterns use fnmatch on dotted
# paths; series are addressed by name. Everything else (QC thresholds, features,
# coerce_chamber_center, identifiers, background matching, which data are included)
# needs the user.
AGENT_OVERRIDABLE = [
    "stitching.test_rasters",
    "devices.*.corners",
    "series.*.corners.*",
    "series.*.reference",
    "series.*.reference.pick",
    "series.*.reference.by",
    "series.*.reference.image",
]


class ConfigError(ValueError):
    pass


def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def sha256_of(obj: Any) -> str:
    """Stable hash of a JSON-serialisable object."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def agent_path(config_path: Path) -> Path:
    return config_path.with_name(config_path.stem + ".agent.yaml")


def resolved_path(config_path: Path) -> Path:
    return config_path.with_name(config_path.stem + ".resolved.yaml")


def log_path(config_path: Path) -> Path:
    return config_path.with_name(config_path.stem + ".log.jsonl")


# --------------------------------------------------------------------------- overrides

def _series_index(cfg: dict, name: str) -> int:
    for i, s in enumerate(cfg["series"]):
        if s["name"] == name:
            return i
    raise ConfigError(f"No series named {name!r}")


def _set_dotted(cfg: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    node: Any = cfg
    i = 0
    while i < len(parts) - 1:
        p = parts[i]
        if p == "series" and isinstance(node, dict) and isinstance(node.get("series"), list):
            node = node["series"][_series_index(cfg, parts[i + 1])]
            i += 2
            continue
        if not isinstance(node, dict):
            raise ConfigError(f"Cannot set {key}: {p} is not a mapping")
        node = node.setdefault(p, {})
        if node is None:
            raise ConfigError(f"Cannot set {key}: {p} is null")
        i += 1
    node[parts[-1]] = value


def check_overridable(key: str) -> None:
    if not any(fnmatch.fnmatchcase(key, pat) for pat in AGENT_OVERRIDABLE):
        raise ConfigError(
            f"Agent may not override {key!r}. Allowed: {', '.join(AGENT_OVERRIDABLE)}. "
            "Ask the user to change processing.yaml instead."
        )


def load_overrides(config_path: Path) -> List[dict]:
    p = agent_path(config_path)
    if not p.exists():
        return []
    entries = yaml.safe_load(p.read_text()) or []
    for e in entries:
        check_overridable(e["key"])
        if not str(e.get("reason", "")).strip():
            raise ConfigError(f"Override {e['key']} has no reason")
    return entries


def add_override(config_path: Path, key: str, value: Any, reason: str) -> dict:
    """Validate and append one agent override; returns the entry."""
    check_overridable(key)
    if not reason or not reason.strip():
        raise ConfigError("An override needs a reason.")
    entry = {"key": key, "value": value, "reason": reason.strip(),
             "at": datetime.now().isoformat(timespec="seconds")}
    # Must apply cleanly to the current config.
    cfg = load_config(config_path)
    _set_dotted(cfg, key, value)
    validate(cfg)
    entries = load_overrides(config_path) + [entry]
    agent_path(config_path).write_text(yaml.safe_dump(entries, sort_keys=False))
    return entry


# --------------------------------------------------------------------------- load / validate

def load_config(config_path: Path, apply_overrides: bool = True) -> dict:
    """Effective config: defaults <- processing.yaml <- agent overrides (validated)."""
    config_path = Path(config_path)
    raw = yaml.safe_load(config_path.read_text()) or {}
    cfg = _merge(DEFAULTS, raw)
    cfg["series"] = [_merge(SERIES_DEFAULTS, s) for s in cfg.get("series") or []]
    for s in cfg["series"]:
        if s.get("reference") is not None and not isinstance(s["reference"], dict):
            raise ConfigError(f"series {s.get('name')}: reference must be a mapping or null")
    if apply_overrides:
        for e in load_overrides(config_path):
            _set_dotted(cfg, e["key"], e["value"])
    validate(cfg)
    cfg["_config_path"] = str(config_path.resolve())
    return cfg


def _corners_ok(c) -> bool:
    try:
        return len(c) == 4 and all(len(p) == 2 and all(isinstance(v, int) for v in p) for p in c)
    except TypeError:
        return False


def validate(cfg: dict) -> None:
    errs = []
    if cfg.get("version") != 1:
        errs.append("version must be 1")
    root = cfg.get("root")
    if not root:
        errs.append("root is required")
    ao = cfg.get("acqui_origin")
    if not (isinstance(ao, list) and len(ao) == 2 and all(isinstance(v, bool) for v in ao)):
        errs.append("acqui_origin must be [bool, bool]")

    st = cfg["stitching"]
    if st.get("rotation") is not None and not isinstance(st["rotation"], (int, float)):
        errs.append("stitching.rotation must be a number or null")
    if not isinstance(st.get("test_rasters"), list):
        errs.append("stitching.test_rasters must be a list")
    for k, v in (st.get("overlap_by_nosepiece") or {}).items():
        if not (isinstance(v, (int, float)) and 0 < v < 1):
            errs.append(f"stitching.overlap_by_nosepiece[{k}] must be in (0, 1)")

    devices = cfg.get("devices") or {}
    bg = cfg["background"]["images"]
    if not (isinstance(bg, str) or (isinstance(bg, list) and bg and all(isinstance(x, str) for x in bg))):
        errs.append("background.images must be a glob or a non-empty list of globs/names")
    if not devices:
        errs.append("at least one device is required")
    for d, dev in devices.items():
        if not isinstance(dev, dict):
            errs.append(f"devices.{d} must be a mapping")
            continue
        if not dev.get("setup"):
            errs.append(f"devices.{d}.setup is required")
        dims = dev.get("dims")
        if not (isinstance(dims, list) and len(dims) == 2 and all(isinstance(v, int) for v in dims)):
            errs.append(f"devices.{d}.dims must be [int, int]")
        if not dev.get("pinlist"):
            errs.append(f"devices.{d}.pinlist is required")
        if dev.get("corners") is not None and not _corners_ok(dev["corners"]):
            errs.append(f"devices.{d}.corners must be 4 [x, y] integer pairs (UL, UR, BL, BR)")

    names = set()
    for s in cfg["series"]:
        n = s.get("name")
        if not n:
            errs.append("every series needs a name")
            continue
        if n in names:
            errs.append(f"duplicate series name {n}")
        names.add(n)
        if not s.get("identifiers"):
            errs.append(f"series {n}: identifiers required")
        if s.get("features") not in FEATURES:
            errs.append(f"series {n}: features must be one of {FEATURES}")
        if not s.get("output"):
            errs.append(f"series {n}: output filename required")
        for d in s.get("devices") or []:
            if d not in devices:
                errs.append(f"series {n}: unknown device {d}")
        for d, c in (s.get("corners") or {}).items():
            if d not in devices:
                errs.append(f"series {n}: corners for unknown device {d}")
            elif not _corners_ok(c):
                errs.append(f"series {n}: corners.{d} must be 4 [x, y] integer pairs")
        ref = s.get("reference")
        if ref is not None:
            pick = ref.get("pick", "brightest")
            if pick not in PICKS:
                errs.append(f"series {n}: reference.pick must be one of {PICKS}")
            if pick in ("max", "min") and not ref.get("by"):
                errs.append(f"series {n}: reference.pick={pick} needs reference.by (a metadata column)")
            if pick == "image" and not ref.get("image"):
                errs.append(f"series {n}: reference.pick=image needs reference.image")
    if not cfg["series"]:
        errs.append("at least one series is required")
    if errs:
        raise ConfigError("Invalid config:\n  - " + "\n  - ".join(errs))


def series_devices(cfg: dict, s: dict) -> List[str]:
    return list(s.get("devices") or cfg["devices"].keys())


def section_hash(cfg: dict, *keys: str) -> str:
    return sha256_of({k: cfg.get(k) for k in keys})
