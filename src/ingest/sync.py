"""Cleaning and synchronization (spec section 1.1).

The stage everything else depends on: put every layer on one grid, one CRS and one
timestamp, then emit the per-cell input records the zone engine consumes.

Layers arrive at different resolutions and ages. Rather than resampling them into
agreement and losing that fact, sync records each layer's age so section 4.1 can turn it
into u_stale. A layer that is 11 hours old should make its cells uncertain, not silently
pass as current.

    python -m src.ingest.sync --demo
    python -m src.ingest.sync --config configs/disasters/flood.yaml -o data/mock/cell_inputs.flood.json
"""
import argparse
import datetime as dt
import json
import pathlib

from src.grid import cell_id
from src.hazard.formulas import REPO_ROOT, load_config
from src.hazard.zones import load_global


def align_timestamps(layers, now=None):
    """{name: iso timestamp} -> {name: age in hours}. Raises on a future timestamp."""
    now = now or dt.datetime.now(dt.timezone.utc)
    ages = {}
    for name, stamp in layers.items():
        if isinstance(stamp, str):
            stamp = dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=dt.timezone.utc)
        age = (now - stamp).total_seconds() / 3600.0
        if age < -0.017:                     # a minute of clock skew is tolerable
            raise ValueError(f"layer {name!r} is timestamped {abs(age):.1f} h in the future")
        ages[name] = max(0.0, age)
    return ages


def oldest_layer(ages):
    """The layer driving staleness. A cell is only as fresh as its stalest input."""
    if not ages:
        raise ValueError("no layers to age")
    name = max(ages, key=ages.get)
    return name, ages[name]


def source_spread(values):
    """Standard deviation across independent sources, which is sigma_ens in section 4.1.

    The spec's cheap and legitimate version of ensemble disagreement: if the gauge, the
    SAR and the forecast disagree, that spread is the uncertainty.
    """
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return 0.0                           # one source cannot disagree with itself
    mean = sum(vals) / len(vals)
    return (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5


def build_cell(lat, lon, hazard_inputs, layer_ages, sensed_fraction, pop, assets,
               reachable, green_inputs, source_values=None, components=None):
    """One record in the shape src.hazard.zones.run() expects."""
    _, age = oldest_layer(layer_ages)
    return {
        "cell_id": cell_id(lat, lon),
        "centroid": [round(lat, 4), round(lon, 4)],
        "hazard_inputs": hazard_inputs,
        "sensors": {
            "sigma_ens": round(source_spread(source_values or []), 5),
            "data_age_hours": round(age, 4),
            "area_sensed_fraction": round(sensed_fraction, 4),
        },
        "green_inputs": green_inputs,
        "pop": pop,
        "assets": assets,
        "reachable": reachable,
        "components": components or {},
    }


def validate(cells, disaster_cfg):
    """Fail loud before the engine runs, naming every cell that is wrong."""
    required = set(disaster_cfg["hazard"]["inputs"])
    problems = []
    seen = set()
    for i, c in enumerate(cells):
        where = c.get("cell_id", f"index {i}")
        if c.get("cell_id") in seen:
            problems.append(f"{where}: duplicate cell_id")
        seen.add(c.get("cell_id"))

        for key in ("cell_id", "centroid", "hazard_inputs", "sensors", "green_inputs",
                    "pop", "reachable"):
            if key not in c:
                problems.append(f"{where}: missing {key!r}")
        if "centroid" in c and "cell_id" in c:
            lat, lon = c["centroid"]
            if cell_id(lat, lon) != c["cell_id"]:
                problems.append(f"{where}: cell_id does not match its centroid")
        missing = required - set(c.get("hazard_inputs", {}))
        if missing:
            problems.append(f"{where}: hazard inputs missing {sorted(missing)}")
        if c.get("pop", 0) < 0:
            problems.append(f"{where}: negative population")

    if problems:
        raise ValueError(f"{len(problems)} problem(s) in cell inputs:\n  "
                         + "\n  ".join(problems[:20]))
    return cells


def _demo_cells(disaster="flood"):
    """A small monsoon scene: a river corridor flooding, high ground to the south.

    Synthetic, and labelled as such. It exists so the flood path can be run end to end
    before the IMD and DEM exports land, not to stand in for real data on stage.
    """
    now = dt.datetime.now(dt.timezone.utc)
    layers = {
        "imd_rainfall": (now - dt.timedelta(hours=1.5)).isoformat(),
        "cwc_gauge": (now - dt.timedelta(hours=3.0)).isoformat(),
        "sentinel1_sar": (now - dt.timedelta(hours=11.2)).isoformat(),
        "ground_report": (now - dt.timedelta(minutes=25)).isoformat(),
    }
    ages = align_timestamps(layers, now)

    # lat, lon, r24, gauge, sensed, pop, assets, reachable, forecast_max_x, elev, cap
    rows = [
        (12.9716, 77.5946, 288.0, 760.0, 0.94, 1450,
         [("hospital", "City General", 1.0)], True, 0.88, 0.15, 200),
        (12.9750, 77.5890, 301.0, 780.0, 0.92, 2310,
         [("school", "Corporation School 4", 0.7)], True, 0.93, 0.10, 100),
        (12.9702, 77.5988, 214.0, 690.0, 0.83, 980,
         [("bridge", "Canal Rd Overpass", 0.9)], False, 0.70, 0.22, 300),
        (12.9688, 77.6021, 186.0, 640.0, 0.94, 1120, [], True, 0.58, 0.31, 600),
        (12.9801, 77.6102, 142.0, None, 0.97, 1670,
         [("clinic", "Ward 12 PHC", 0.6)], True, 0.47, 0.60, 1600),
        (12.9640, 77.6210, 118.0, None, 0.95, 640, [], True, 0.41, 0.46, 1000),
        (12.9601, 77.6033, 44.0, None, 0.96, 410,
         [("shelter", "Govt High School", 1.0)], True, 0.16, 0.91, 2400),
        (12.9555, 77.6150, 71.0, None, 0.92, 520,
         [("shelter", "Community Hall", 0.8)], True, 0.26, 0.76, 1800),
    ]

    q_max = 900.0

    def flood_x(r24, q):
        """The flood formula from configs/disasters/flood.yaml, used here to put the
        cross-source spread in X units. sigma_max is 0.25 on a 0-1 scale, so feeding it
        a spread in cumecs would saturate u_model on every cell."""
        return min(1.0, 0.5 * (r24 / 300.0) + 0.5 * (q / q_max))

    cells = []
    for lat, lon, r24, gauge, sensed, pop, assets, reach, fmax, elev, cap in rows:
        rain_derived = r24 * 2.4                                # rating-curve stand-in
        discharge = gauge if gauge is not None else rain_derived

        # A cell confirmed by a river gauge is only as stale as the gauge. A cell whose
        # only water evidence is the SAR pass inherits the SAR's age, which is what puts
        # it in the recon queue.
        cell_layers = {"imd_rainfall": ages["imd_rainfall"]}
        cell_layers["cwc_gauge" if gauge is not None else "sentinel1_sar"] = (
            ages["cwc_gauge"] if gauge is not None else ages["sentinel1_sar"])

        # A designated shelter has staff on site, and a phone call is a sensor. Ground
        # truth overrides the stale satellite pass, which is what lets these cells be
        # called GREEN instead of sitting in the recon queue.
        if any(t == "shelter" for t, _n, _v in assets):
            cell_layers = {"ground_report": ages["ground_report"]}

        cells.append(build_cell(
            lat, lon,
            hazard_inputs={"r24_mm": r24, "q_cumecs": discharge, "q_max": q_max},
            layer_ages=cell_layers,
            sensed_fraction=sensed,
            pop=pop,
            assets=[{"type": t, "name": n, "value": v} for t, n, v in assets],
            reachable=reach,
            green_inputs={
                "forecast_max_x": fmax, "elevation_norm": elev,
                "shelter_capacity": cap, "pop_inflow": 2000,
                "access": 0.9 if reach else 0.1,
                "travel_distance_norm": round(abs(lat - 12.9601) * 12, 3),
            },
            # Gauge and rainfall imply different discharges. Their disagreement, in X
            # units, is sigma_ens: the cheap legitimate ensemble from section 4.1.
            source_values=([flood_x(r24, gauge), flood_x(r24, rain_derived)]
                           if gauge is not None else []),
            components={"r24_mm": r24, "river_gauge_m": gauge,
                        "x_from_gauge": round(flood_x(r24, gauge), 3) if gauge else None,
                        "x_from_rainfall": round(flood_x(r24, rain_derived), 3),
                        "evidence": "gauge" if gauge is not None else "sar_only"},
        ))
    return cells


def _cli():
    ap = argparse.ArgumentParser(description="Regrid and align layers into cell inputs.")
    ap.add_argument("--config", default="configs/disasters/flood.yaml")
    ap.add_argument("--demo", action="store_true", help="synthetic monsoon scene")
    ap.add_argument("-o", "--out")
    args = ap.parse_args()

    cfg = load_config(args.config)
    load_global()                             # fail early if the global config is broken

    if not args.demo:
        raise SystemExit(
            "No real layers wired yet. Fetch them first:\n"
            "    python -m src.ingest.sources --list\n"
            "then run with --demo to exercise the path with a synthetic scene."
        )

    cells = validate(_demo_cells(cfg["name"]), cfg)
    text = json.dumps(cells, indent=2)
    if args.out:
        path = pathlib.Path(args.out)
        if not path.is_absolute():
            path = REPO_ROOT / path
        path.write_text(text + "\n")
        ages = {c["cell_id"]: c["sensors"]["data_age_hours"] for c in cells}
        print(f"{path.relative_to(REPO_ROOT)}: {len(cells)} cells, "
              f"stalest layer {max(ages.values()):.1f} h old")
    else:
        print(text)


if __name__ == "__main__":
    _cli()
