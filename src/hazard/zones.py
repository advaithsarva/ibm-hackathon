"""Zone engine (section 4) -- the core novelty.

Turns hazard X, epistemic uncertainty U and green-zone suitability G into RED / BLUE /
GREEN, with hysteresis so cells stop flickering between classes on every cycle.

BLUE is not "medium danger". BLUE is "we do not know enough to call this safe", which
makes it a recon queue rather than a caveat.

Standalone:
    python -m src.hazard.zones --config configs/disasters/earthquake.yaml
        --cells data/mock/cell_inputs.earthquake.json -o data/mock/zones.json
"""
import argparse
import datetime as dt
import json
import math
import pathlib

import yaml

from src.grid import cell_id
from src.hazard.formulas import REPO_ROOT, hazard_score, load_config

SEVERITY = {"GREEN": 0, "BLUE": 1, "RED": 2}


def load_global(path="configs/global.yaml"):
    p = pathlib.Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"global config not found: {p}")
    return yaml.safe_load(p.read_text())


# --- 4.1 epistemic uncertainty ---------------------------------------------------

def u_model(sigma_ens, sigma_max):
    """Ensemble or cross-source disagreement. The cheap legitimate version of sigma_ens
    is the standard deviation across independent sources (gauge vs SAR vs forecast)."""
    if sigma_max <= 0:
        raise ValueError("sigma_max must be > 0")
    return min(1.0, sigma_ens / sigma_max)


def u_stale(data_age_hours, tau_data_hours):
    """1 - exp(-dt/tau). Disaster-specific tau: 1 h seismic, 6 h rainfall, 12 h SAR."""
    if tau_data_hours <= 0:
        raise ValueError("tau_data_hours must be > 0")
    if data_age_hours < 0:
        raise ValueError("data age cannot be negative")
    return 1.0 - math.exp(-data_age_hours / tau_data_hours)


def u_cover(area_sensed_fraction):
    """1 - A_sensed/A_cell. A cell nothing has looked at is maximally uncertain."""
    if not 0.0 <= area_sensed_fraction <= 1.0:
        raise ValueError("area_sensed_fraction must be in [0,1]")
    return 1.0 - area_sensed_fraction


def uncertainty(um, us, uc):
    """Noisy-OR: U = 1 - (1-u_model)(1-u_stale)(1-u_cover).

    Any one source of ignorance is enough to raise U; they compound rather than average.
    """
    return 1.0 - (1.0 - um) * (1.0 - us) * (1.0 - uc)


# --- 4.3 green-zone suitability --------------------------------------------------

def green_suitability(forecast_max_x, elevation_norm, shelter_capacity, pop_inflow,
                      access, travel_distance_norm, weights):
    """G = w1(1 - max_t X) + w2*z + w3*min(1, Cap/Inflow) + w4*Access - w5*d_road.

    forecast_max_x is the maximum X over the forecast horizon, NOT current X. A green
    zone that floods in six hours is a death trap.
    """
    capacity_ratio = 1.0 if pop_inflow <= 0 else min(1.0, shelter_capacity / pop_inflow)
    g = (weights["w_forecast_hazard"] * (1.0 - forecast_max_x)
         + weights["w_elevation"] * elevation_norm
         + weights["w_shelter_capacity"] * capacity_ratio
         + weights["w_access"] * access
         - weights["w_travel_distance"] * travel_distance_norm)
    return max(0.0, min(1.0, g))


# --- 4.2 zone assignment ---------------------------------------------------------

def classify(x, u, g, reachable, thresholds):
    """RED if X >= 0.75 or (X >= 0.40 and U >= 0.50).
    GREEN if X < 0.40 and U < 0.35 and G >= 0.60 and reachable. BLUE otherwise."""
    lo, hi = thresholds["low"], thresholds["high"]
    if x >= hi or (x >= lo and u >= 0.50):
        return "RED"
    if x < lo and u < 0.35 and g >= 0.60 and reachable:
        return "GREEN"
    return "BLUE"


def zone_reason(x, u, g, reachable, thresholds, zone):
    """Why this cell is this colour. The BLUE explanation is the demo's winning moment,
    so it names the binding constraint rather than saying "otherwise"."""
    lo, hi = thresholds["low"], thresholds["high"]
    if zone == "RED":
        return f"X >= {hi}" if x >= hi else f"X >= {lo} and U >= 0.50"
    if zone == "GREEN":
        return f"X < {lo}, U < 0.35, G >= 0.60, reachable"
    if not reachable:
        return "unreachable, cannot be called safe"
    if x >= lo:
        return f"X >= {lo} so not GREEN; U < 0.50 so not RED"
    if u >= 0.35:
        return f"U = {u:.2f} >= 0.35, too uncertain to call safe"
    if g < 0.60:
        return f"G = {g:.2f} < 0.60, unsuitable as a green zone"
    return "does not meet any GREEN criterion"


class Hysteresis:
    """A cell changes class only after crossing the threshold by +/- margin AND holding
    for hold_cycles consecutive cycles (section 4.2). Without this, cells sitting on a
    boundary flip every refresh and the map is unreadable.

    State is per cell and lives here, so callers just feed cycles in order.
    """

    def __init__(self, margin=0.05, hold_cycles=2, thresholds=None):
        self.margin = margin
        self.hold_cycles = hold_cycles
        self.thresholds = thresholds or {"low": 0.40, "high": 0.75}
        self._state = {}  # cell_id -> {"zone", "candidate", "count"}

    def update(self, cid, x, u, g, reachable):
        """Returns the zone this cell should display now."""
        raw = classify(x, u, g, reachable, self.thresholds)
        st = self._state.get(cid)
        if st is None:  # first sighting: no history to damp against
            self._state[cid] = {"zone": raw, "candidate": raw, "count": 0}
            return raw

        current = st["zone"]
        if raw == current:
            st["candidate"], st["count"] = raw, 0
            return current

        # Nudge X and U back toward the current class. If the cell still classifies as
        # raw under that handicap, the crossing is decisive rather than boundary noise.
        sign = 1 if SEVERITY[raw] > SEVERITY[current] else -1
        nudged = classify(x - sign * self.margin, u - sign * self.margin, g, reachable,
                          self.thresholds)
        if nudged != raw:
            st["candidate"], st["count"] = current, 0
            return current

        st["count"] = st["count"] + 1 if st["candidate"] == raw else 1
        st["candidate"] = raw
        if st["count"] >= self.hold_cycles:
            st["zone"], st["count"] = raw, 0
            return raw
        return current


# --- 4.4 exposure ----------------------------------------------------------------

def rescue_priority(x, pop, max_pop, asset_value, max_asset, isolation, max_isolation,
                    weights):
    """RescuePriority = X * (w1*pop/maxpop + w2*assets/maxasset + w3*isolation/maxiso).

    Feeds N_i in section 6.2. Ratios, so it compares cells within one incident, not
    across incidents.
    """
    def norm(v, m):
        return 0.0 if m <= 0 else v / m

    return x * (weights["w_population"] * norm(pop, max_pop)
                + weights["w_critical_assets"] * norm(asset_value, max_asset)
                + weights["w_isolation"] * norm(isolation, max_isolation))


# --- pipeline --------------------------------------------------------------------

def run(disaster_cfg, global_cfg, cells, hysteresis=None):
    """Raw per-cell inputs -> the section 2.1 /api/zones contract."""
    thresholds = disaster_cfg.get("thresholds", global_cfg["thresholds"])
    tau = disaster_cfg.get("uncertainty", {}).get(
        "tau_data_hours", global_cfg["uncertainty"]["tau_data_hours"]["seismic"])
    sigma_max = global_cfg["uncertainty"]["sigma_max"]
    gw = global_cfg["green_suitability"]

    out = []
    for c in cells:
        lat, lon = c["centroid"]
        cid = cell_id(lat, lon)
        if cid != c["cell_id"]:
            raise ValueError(f"{c['cell_id']}: centroid {lat},{lon} yields {cid}")

        x = hazard_score(disaster_cfg, c["hazard_inputs"])

        s = c["sensors"]
        um = u_model(s["sigma_ens"], sigma_max)
        us = u_stale(s["data_age_hours"], tau)
        uc = u_cover(s["area_sensed_fraction"])
        u = uncertainty(um, us, uc)

        gi = c["green_inputs"]
        g = green_suitability(gi["forecast_max_x"], gi["elevation_norm"],
                              gi["shelter_capacity"], gi["pop_inflow"], gi["access"],
                              gi["travel_distance_norm"], gw)

        zone = (hysteresis.update(cid, x, u, g, c["reachable"]) if hysteresis
                else classify(x, u, g, c["reachable"], thresholds))

        out.append({
            "cell_id": cid,
            "centroid": [lat, lon],
            "X": round(x, 4), "U": round(u, 4), "G": round(g, 4),
            "zone": zone,
            "zone_reason": zone_reason(x, u, g, c["reachable"], thresholds, zone),
            "pop": c["pop"],
            "assets": c.get("assets", []),
            "reachable": c["reachable"],
            "components": c.get("components", {}),
            "uncertainty_components": {"u_model": round(um, 4), "u_stale": round(us, 4),
                                       "u_cover": round(uc, 4)},
        })

    max_pop = max((c["pop"] for c in out), default=0)
    max_asset = max((sum(a.get("value", 0) for a in c["assets"]) for c in out), default=0)
    ew = global_cfg["exposure"]
    for c in out:
        c["rescue_priority"] = round(rescue_priority(
            c["X"], c["pop"], max_pop,
            sum(a.get("value", 0) for a in c["assets"]), max_asset,
            0.0 if c["reachable"] else 1.0, 1.0, ew), 4)

    return {
        "disaster": disaster_cfg["name"],
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "grid_cells": out,
    }


def _cli():
    ap = argparse.ArgumentParser(description="Run the zone engine over a cell-input file.")
    ap.add_argument("--config", required=True, help="configs/disasters/*.yaml")
    ap.add_argument("--cells", required=True, help="JSON list of raw per-cell inputs")
    ap.add_argument("--global-config", default="configs/global.yaml")
    ap.add_argument("-o", "--out", help="write the contract here instead of stdout")
    args = ap.parse_args()

    cells_path = pathlib.Path(args.cells)
    if not cells_path.is_absolute():
        cells_path = REPO_ROOT / cells_path
    if not cells_path.exists():
        raise FileNotFoundError(f"cell inputs not found: {cells_path}")

    result = run(load_config(args.config), load_global(args.global_config),
                 json.loads(cells_path.read_text()))

    text = json.dumps(result, indent=2)
    if args.out:
        out_path = pathlib.Path(args.out)
        if not out_path.is_absolute():
            out_path = REPO_ROOT / out_path
        out_path.write_text(text + "\n")
        counts = {}
        for c in result["grid_cells"]:
            counts[c["zone"]] = counts.get(c["zone"], 0) + 1
        print(f"{out_path.relative_to(REPO_ROOT)}: {len(result['grid_cells'])} cells, " +
              " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    else:
        print(text)


if __name__ == "__main__":
    _cli()
