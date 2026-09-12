"""Config-driven hazard evaluator (§3.1).

The seven hazard formulas are strings in configs/disasters/*.yaml, not seven functions
here. Adding a disaster is a YAML file, not a code change.

Standalone:
    python -m src.hazard.formulas --config configs/disasters/earthquake.yaml --pga_ms2 3.4
"""
import argparse
import math
import pathlib

import yaml

# The only names a formula may reference besides its declared inputs.
_ALLOWED = {
    "min": min, "max": max, "abs": abs, "round": round,
    "exp": math.exp, "sqrt": math.sqrt, "log": math.log, "pow": pow,
}

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_config(path):
    """Load a disaster config. Fails loud if it is missing (§12.9)."""
    p = pathlib.Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"disaster config not found: {p}")
    cfg = yaml.safe_load(p.read_text())
    for key in ("name", "hazard"):
        if key not in cfg:
            raise ValueError(f"{p}: config is missing required key {key!r}")
    for key in ("formula", "inputs"):
        if key not in cfg["hazard"]:
            raise ValueError(f"{p}: hazard block is missing required key {key!r}")
    return cfg


def hazard_score(cfg, inputs):
    """Evaluate X for one cell. Returns a float in [0,1].

    Missing inputs raise. Never substitute a zero for absent data (§12.9) — a zero
    hazard score and an unmeasured cell are opposite claims.
    """
    required = cfg["hazard"]["inputs"]
    missing = [k for k in required if k not in inputs or inputs[k] is None]
    if missing:
        raise ValueError(f"{cfg['name']}: missing hazard inputs {missing}")

    env = dict(_ALLOWED)
    env.update({k: inputs[k] for k in required})
    try:
        x = eval(cfg["hazard"]["formula"], {"__builtins__": {}}, env)  # noqa: S307 - formula is repo config
    except ZeroDivisionError:
        raise ValueError(f"{cfg['name']}: division by zero evaluating {cfg['hazard']['formula']!r}")

    # Not every spec formula is self-clamping — landslide's weighted sum can exceed 1
    # when both terms are extreme. X is defined on [0,1] (§3), so clamp at the boundary.
    return max(0.0, min(1.0, float(x)))


def tier(x, thresholds):
    """Low / medium / high (§3.1)."""
    if x >= thresholds["high"]:
        return "high"
    return "medium" if x >= thresholds["low"] else "low"


def _cli():
    ap = argparse.ArgumentParser(description="Evaluate a hazard formula for one cell.")
    ap.add_argument("--config", required=True)
    args, rest = ap.parse_known_args()
    cfg = load_config(args.config)

    ap2 = argparse.ArgumentParser()
    for name in cfg["hazard"]["inputs"]:
        ap2.add_argument(f"--{name}", type=float, required=True)
    inputs = vars(ap2.parse_args(rest))

    x = hazard_score(cfg, inputs)
    thresholds = yaml.safe_load((REPO_ROOT / "configs/global.yaml").read_text())["thresholds"]
    print(f"{cfg['name']}: X = {x:.4f} ({tier(x, thresholds)})")


if __name__ == "__main__":
    _cli()
