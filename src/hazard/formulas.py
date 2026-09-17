"""Config-driven hazard evaluator (§3.1).

The seven hazard formulas are strings in configs/disasters/*.yaml, not seven functions
here. Adding a disaster is a YAML file, not a code change.

Standalone:
    python -m src.hazard.formulas --config configs/disasters/earthquake.yaml --pga_ms2 3.4
"""
import argparse
import ast
import math
import pathlib

import yaml

# The only names a formula may reference besides its declared inputs.
_ALLOWED = {
    "min": min, "max": max, "abs": abs, "round": round,
    "exp": math.exp, "sqrt": math.sqrt, "log": math.log, "pow": pow,
}

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# The node types a hazard formula is allowed to be made of. eval() with an
# emptied __builtins__ is not a sandbox -- `().__class__.__bases__[0]
# .__subclasses__()` walks straight back out to anything importable -- and the
# only thing keeping that unreachable today is that every formula comes from a
# YAML file in this repo, reached through a fixed dict in backend/main.py. That
# is one request parameter away from being untrue, so the formula is checked
# for shape before it is evaluated rather than trusted for its provenance.
_ALLOWED_NODES = (
    ast.Expression, ast.Constant, ast.Name, ast.Load,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Call,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)


def compile_formula(formula, allowed_names):
    """Parse a formula and reject anything that is not arithmetic.

    Returns a code object ready for eval(). Raises ValueError on any node type
    outside _ALLOWED_NODES, any name that is not a declared input or an entry
    in _ALLOWED, and any call to something other than those entries -- which
    together rule out attribute access, subscripting, comprehensions, lambdas,
    imports and walrus assignment.
    """
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"formula is not a Python expression: {formula!r} ({exc})")

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(
                f"{type(node).__name__} is not allowed in a hazard formula: {formula!r}")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValueError(f"unknown name {node.id!r} in formula {formula!r}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED:
                raise ValueError(f"only {sorted(_ALLOWED)} may be called: {formula!r}")
            if node.keywords:
                raise ValueError(f"keyword arguments are not allowed: {formula!r}")

    return compile(tree, "<hazard formula>", "eval")


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
    code = compile_formula(cfg["hazard"]["formula"], set(env))
    try:
        x = eval(code, {"__builtins__": {}}, env)  # noqa: S307 - shape-checked above
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
