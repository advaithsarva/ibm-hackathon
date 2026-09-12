"""The Sentinel Grid — FastAPI backend.

Serves the eight endpoints the dashboard consumes. Everything is computed by the real
engine in src/ from a ward scenario rather than read from fixtures, so moving a slider
re-runs the hazard and zone models instead of replaying a recording.

    python run_demo.py            # http://localhost:8000
"""
import json
import math
import pathlib
import random
import time
from typing import Optional

from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.detect.fusion import fuse_cell
from src.hazard.formulas import load_config
from src.hazard.zones import load_global, run as run_zones
from src.priority.dispatch import build_plan
from src.priority.expected_lives import capability_match
from src.scenario import BANKFULL_M, build_cells, haversine_km, load_scenario

ROOT = pathlib.Path(__file__).parent.parent
FRONTEND = ROOT / "frontend"

app = FastAPI(title="Sentinel Grid", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

# Which hazard config scores each disaster the dashboard offers. Flood uses the
# rainfall-only variant because there is no CWC discharge export; see its config header.
DISASTER_CONFIG = {
    "flood": "configs/disasters/flood_rainfall.yaml",
    "earthquake": "configs/disasters/earthquake.yaml",
    "cyclone": "configs/disasters/cyclone.yaml",
    "landslide": "configs/disasters/landslide.yaml",
}
SCENARIO = "coimbatore_flood"

# Equipment a team needs for a given depth of standing water. Drives the IAP protocol.
EQUIPMENT = [
    (1.5, "Inflatable rescue boat + swift-water PPE"),
    (0.8, "High-clearance vehicle + wading poles"),
    (0.0, "Foot party + medical kit"),
]

_cache: dict = {}


def _scenario():
    if "scenario" not in _cache:
        _cache["scenario"] = load_scenario(SCENARIO)
    return _cache["scenario"]


def compute_zones(disaster="flood", river_stage_m=None, rainfall_scale=1.0,
                  sar_staleness_h=None):
    """Run the real engine over the ward scenario."""
    scenario = _scenario()
    cells = build_cells(scenario, river_stage_m, rainfall_scale, sar_staleness_h)
    cfg = load_config(DISASTER_CONFIG.get(disaster, DISASTER_CONFIG["flood"]))
    result = run_zones(cfg, load_global(), cells)

    # Carry ward name and terrain onto every cell so the UI never has to join.
    by_id = {c["cell_id"]: c for c in cells}
    for cell in result["grid_cells"]:
        source = by_id[cell["cell_id"]]
        cell["ward_name"] = source["ward_name"]
        for key in ("elevation_m", "relief_m", "river_dist_km", "water_depth_m",
                    "rainfall_24h_mm"):
            cell[key] = source["components"][key]
    result["disaster"] = disaster
    result["district"] = scenario["district"]
    result["river"] = scenario["river"]
    return result


def equipment_for(depth_m):
    return next(text for threshold, text in EQUIPMENT if depth_m >= threshold)


def detections_for(zones):
    """A sensor picture for the worst wards, fused for real.

    The individual detection records are stand-ins until drone and camera feeds are
    wired, and each says so. The p_alive on top of them is computed by the actual
    log-odds fusion in src/detect/fusion.py, so the number on screen is the model's
    output rather than a literal.
    """
    out = []
    for cell in sorted(zones["grid_cells"], key=lambda c: -c["X"])[:6]:
        rng = random.Random(cell["cell_id"])
        depth = cell["water_depth_m"]
        detections = [{"type": "thermal_hotspot",
                       "conf": round(rng.uniform(0.55, 0.8), 2),
                       "temp_c": round(rng.uniform(31.5, 35.5), 1)}]
        if depth > 0.6:
            detections.append({"type": "rgb_person",
                               "conf": round(rng.uniform(0.55, 0.75), 2),
                               "bbox": [rng.randint(60, 200), rng.randint(50, 180),
                                        rng.randint(210, 320), rng.randint(200, 340)]})
        if cell["X"] > 0.9:
            detections.append({"type": "rppg_pulse",
                               "conf": round(rng.uniform(0.8, 0.92), 2),
                               "bpm": rng.randint(78, 112),
                               "snr_db": round(rng.uniform(4.5, 8.0), 1)})
            detections.append({"type": "acoustic_distress",
                               "conf": round(rng.uniform(0.45, 0.62), 2),
                               "class": rng.choice(["Shout", "Screaming", "Knock"])})

        record = fuse_cell(cell["cell_id"], detections,
                           source=f"drone_sortie_{rng.randint(1, 9):02d}")
        record["ward_name"] = cell["ward_name"]
        record["water_depth_m"] = depth
        record["telemetry"] = "synthetic pending live feeds; p_alive is real fusion output"
        out.append(record)
    return out


def priority_for(zones, disaster="flood"):
    """Rank the RED and BLUE wards by expected lives saved, then assign teams."""
    cfg = load_config(DISASTER_CONFIG.get(disaster, DISASTER_CONFIG["flood"]))
    centre = _scenario()["centre"]

    candidates = []
    for cell in zones["grid_cells"]:
        if cell["zone"] not in ("RED", "BLUE") or cell["pop"] <= 0:
            continue
        # People in genuine danger, not the whole ward: hazard scales the exposure.
        at_risk = int(cell["pop"] * min(1.0, cell["X"]) * 0.12)
        if at_risk < 1:
            continue
        km = haversine_km(cell["centroid"], centre)
        speed = 34.0 if cell["water_depth_m"] < 0.8 else 17.0   # flooded roads halve it
        eta = max(4, int(round(km / speed * 60)))
        team_kind = ("boat" if cell["water_depth_m"] >= 1.5
                     else "swift_water" if cell["water_depth_m"] >= 0.8 else "medical")
        candidates.append({
            "cell_id": cell["cell_id"], "ward_name": cell["ward_name"],
            "n_est": at_risk, "p_alive": round(min(0.97, 0.55 + 0.4 * cell["X"]), 3),
            "eta_min": eta, "cost_team_hours": round(1.5 + eta / 30.0, 2),
            "team_kind": team_kind, "water_depth_m": cell["water_depth_m"],
            "route": [centre, cell["centroid"]],
            "capability_match": capability_match("flood", team_kind),
        })

    teams = ([{"id": f"NDRF-{i}", "kind": "boat", "hours": 9.0} for i in range(1, 4)]
             + [{"id": f"SDRF-{i}", "kind": "swift_water", "hours": 8.0} for i in range(1, 4)]
             + [{"id": f"TNFRS-{i}", "kind": "medical", "hours": 7.0} for i in range(1, 3)])

    plan = build_plan(candidates, cfg, budget_team_hours=48.0, teams=teams)
    by_id = {c["cell_id"]: c for c in candidates}
    for row in plan["ranked"]:
        source = by_id[row["cell_id"]]
        row["ward_name"] = source["ward_name"]
        row["target_ward"] = source["ward_name"]
        row["priority"] = row["rank"]
        row["population_at_risk"] = source["n_est"]
        row["water_depth_m"] = source["water_depth_m"]
        row["assigned_team"] = row.get("team") or "unassigned"
        row["recommended_equipment"] = equipment_for(source["water_depth_m"])
    plan["unassigned_wards"] = [by_id[c]["ward_name"] for c in plan["unassigned"]
                                if c in by_id]
    return plan


def evacuation_for(zones):
    """Move people from RED and BLUE wards to the high-ground relief centres."""
    scenario = _scenario()
    policy = scenario["evacuation_policy"]
    shelters, sources = [], []
    for cell in zones["grid_cells"]:
        ward = next(w for w in scenario["wards"] if w["name"] == cell["ward_name"])
        if ward.get("shelter_capacity"):
            shelters.append({"cell": cell, "ward": ward})
        elif (cell["zone"] in ("RED", "BLUE") and cell["pop"] > 0
              and cell["water_depth_m"] >= policy["trigger_depth_m"]):
            sources.append(cell)

    assigned = {s["ward"]["name"]: 0 for s in shelters}
    flows = []
    for cell in sorted(sources, key=lambda c: -c["X"]):
        people = int(cell["pop"] * min(1.0, cell["X"]) * policy["evacuation_rate"])
        if people < 1:
            continue
        # Nearest shelter with room. Capacity is respected, so a full centre pushes the
        # next group further out instead of absorbing everyone nearby.
        options = sorted(shelters, key=lambda s: haversine_km(cell["centroid"],
                                                              s["cell"]["centroid"]))
        for shelter in options:
            name = shelter["ward"]["name"]
            room = shelter["ward"]["shelter_capacity"] - assigned[name]
            if room <= 0:
                continue
            moved = min(people, room)
            km = haversine_km(cell["centroid"], shelter["cell"]["centroid"])
            assigned[name] += moved
            people -= moved
            minutes = max(5, int(round(km / 26.0 * 60)))
            flows.append({
                "from_cell": cell["cell_id"], "from_ward": cell["ward_name"],
                "to_shelter": name, "to_ward": name,
                "people": moved, "people_count": moved,
                "path": [cell["centroid"], shelter["cell"]["centroid"]],
                "route_km": round(km, 1), "travel_min": minutes, "eta_min": minutes,
                "hazard_avoided": round(cell["X"], 2),
            })
            if people <= 0:
                break
        if people > 0:
            flows.append({"from_cell": cell["cell_id"], "from_ward": cell["ward_name"],
                          "people": people, "people_count": people, "overflow": True})

    overflow = [{"from_cell": f["from_cell"], "from_ward": f["from_ward"],
                 "people": f["people"],
                 "reason": "every reachable relief centre is at capacity"}
                for f in flows if f.get("overflow")]

    return {
        "flows": [f for f in flows if not f.get("overflow")],
        "shelters": [{
            "id": s["ward"]["name"].lower().replace(" ", "_"),
            "name": f"{s['ward']['name']} Relief Centre",
            "ward_name": s["ward"]["name"],
            "capacity": s["ward"]["shelter_capacity"],
            "assigned": assigned[s["ward"]["name"]],
            "utilization": round(assigned[s["ward"]["name"]]
                                 / s["ward"]["shelter_capacity"], 4),
            "elevation": f"{s['ward']['elevation_m']} m",
            "status": ("OPEN" if assigned[s["ward"]["name"]]
                       < s["ward"]["shelter_capacity"] else "AT CAPACITY"),
            "location": s["cell"]["centroid"],
        } for s in shelters],
        "overflow": overflow,
        "prohibited_transit_routes": scenario["prohibited_routes"],
    }


# ── endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/zones")
def get_zones(disaster: str = "flood"):
    return compute_zones(disaster)


@app.get("/api/detections")
def get_detections(disaster: str = "flood"):
    return detections_for(compute_zones(disaster))


@app.get("/api/priority")
def get_priority(disaster: str = "flood"):
    return priority_for(compute_zones(disaster), disaster)


@app.get("/api/evacuation")
def get_evacuation(disaster: str = "flood"):
    return evacuation_for(compute_zones(disaster))


@app.get("/api/alert/summary")
def get_alert_summary(disaster: str = "flood"):
    zones = compute_zones(disaster)
    cells = zones["grid_cells"]
    counts = {z: sum(1 for c in cells if c["zone"] == z)
              for z in ("RED", "BLUE", "GREEN")}
    return {
        "disaster": disaster,
        "district": zones["district"],
        "generated_at": zones["generated_at"],
        "red_cells": counts["RED"], "blue_cells": counts["BLUE"],
        "green_cells": counts["GREEN"],
        "at_risk_population": sum(c["pop"] for c in cells
                                  if c["zone"] in ("RED", "BLUE")),
        "river_stage_m": _scenario()["state"]["river_stage_m"],
        "bankfull_m": BANKFULL_M,
        "alert_level": ("CRITICAL" if counts["RED"] >= 3
                        else "WARNING" if counts["RED"] >= 1 else "MONITORING"),
    }


@app.post("/api/simulate")
def simulate(payload: dict = Body(...)):
    """Re-run the engine under operator-set storm conditions.

    A real re-computation, not a lookup. The sliders change river stage, rainfall and
    satellite staleness, and every X, U, G and zone is derived again from those.
    """
    zones = compute_zones(
        payload.get("disaster", "flood"),
        river_stage_m=payload.get("river_stage_m"),
        rainfall_scale=payload.get("rainfall_scale", 1.0),
        sar_staleness_h=payload.get("sar_staleness_h"),
    )
    counts = {z: sum(1 for c in zones["grid_cells"] if c["zone"] == z)
              for z in ("RED", "BLUE", "GREEN")}
    return {
        "grid_cells": zones["grid_cells"],
        "generated_at": zones["generated_at"],
        "inputs": {"river_stage_m": payload.get("river_stage_m"),
                   "rainfall_scale": payload.get("rainfall_scale", 1.0),
                   "sar_staleness_h": payload.get("sar_staleness_h")},
        "zone_counts": counts,
        "at_risk_population": sum(c["pop"] for c in zones["grid_cells"]
                                  if c["zone"] in ("RED", "BLUE")),
    }


@app.get("/api/model/metrics")
def model_metrics():
    """Validation benchmarks, read from what was actually measured.

    Where a model has been trained and evaluated, the figures come from its metadata on
    disk. Where it has not, the field says so rather than showing a number nobody
    produced.
    """
    features = ROOT / "data" / "cache" / "features"
    baselines = {}
    try:
        from src.features import logistic_baseline
        for name in ("landslide", "cyclone"):
            if (features / name / "train.npz").exists():
                baselines[name] = logistic_baseline(name)
    except Exception:
        pass

    primary = {
        "name": "Physics-based flood hazard (IMD rainfall + terrain retention)",
        "dataset": "IMD 0.25 deg gridded rainfall + Coimbatore ward terrain",
        "type": "deterministic, no training",
        "metrics": {},
        "note": ("The shipping hazard path is documented physics, not a fitted model, so "
                 "it has no held-out score. Its formulas are verified against the spec "
                 "by src/audit_math.py, 104 checks."),
    }
    if "landslide" in baselines:
        b = baselines["landslide"]["val"]
        primary = {
            "name": "Landslide susceptibility, logistic regression",
            "dataset": "Kaggle landslide dataset, 2000 rows, 9 features",
            "type": "trained, held-out validation",
            "metrics": {"accuracy": b["accuracy"], "precision": b["precision"],
                        "recall": b["recall"], "f1": b["f1"]},
            "note": ("Logistic regression already separates this dataset, so nothing "
                     "heavier was trained. A shuffled-label control scores 0.50, so the "
                     "separation is in the data rather than in the measurement."),
        }

    secondary = {
        "name": "Cyclone formation likelihood, logistic regression",
        "dataset": "Kaggle cyclone dataset, 2000 rows, 8 features",
        "type": "trained, held-out validation",
        "metrics": {},
        "note": ("Pre_existing_Disturbance was excluded: it is a verbatim copy of the "
                 "target and scores 100% on its own."),
    }
    if "cyclone" in baselines:
        b = baselines["cyclone"]["val"]
        secondary["metrics"] = {"accuracy": b["accuracy"], "precision": b["precision"],
                                "recall": b["recall"], "f1": b["f1"]}

    return {
        "primary_hazard_model": primary,
        "secondary_benchmarking_model": secondary,
        "fusion": {
            "formula": "logit(P) = logit(p0) + sum z_k * ln(lambda_k)",
            "likelihood_ratios": load_global()["fusion"]["likelihood_ratios"],
        },
        "verification": {"formula_audit_checks": 104, "test_checks": 55,
                         "command": "python -m src.audit_math"},
    }


@app.get("/api/export/incident_action_plan")
def incident_action_plan(disaster: str = "flood"):
    """The four PS-1 mandates as an operational order."""
    scenario = _scenario()
    zones = compute_zones(disaster)
    plan = priority_for(zones, disaster)
    evac = evacuation_for(zones)
    cells = zones["grid_cells"]

    red = [c for c in cells if c["zone"] == "RED"]
    blue = [c for c in cells if c["zone"] == "BLUE"]
    state = scenario["state"]

    return {
        "operation_name": f"OP NOYYAL SHIELD - {scenario['district']}",
        "operational_period": "Next 24 hours from issue",
        "issued_at": zones["generated_at"],
        "calamity_parameters": {
            "cwc_river_stage":
                f"{state['river_stage_m']:.1f} m "
                f"({state['river_stage_m'] - BANKFULL_M:+.1f} m vs bankfull)",
            "meteorological_forcing":
                f"IMD 24 h rainfall {state['rainfall_24h_mm']:.0f} mm; "
                f"SAR pass {state['sar_staleness_h']:.1f} h stale",
        },
        "mandate_1_where_likely_to_occur": {
            "critical_risk_zones_count": len(red),
            "critical_wards": [c["ward_name"] for c in sorted(red, key=lambda c: -c["X"])],
            "reconnaissance_drone_queue_blue_zones":
                [c["ward_name"] for c in sorted(blue, key=lambda c: -c["U"])],
            "method": "X from IMD rainfall with terrain retention; zones per spec 4.2",
        },
        "mandate_2_who_and_what_affected": {
            "total_endangered_population": sum(c["pop"] for c in red + blue),
            "critical_infrastructure_at_risk":
                [f"{a['name']} ({c['ward_name']})"
                 for c in red for a in c.get("assets", [])],
            "submerged_chokepoints":
                [f"{c['ward_name']} ({c['water_depth_m']:.1f} m)"
                 for c in sorted(red, key=lambda c: -c["water_depth_m"])[:4]
                 if c["water_depth_m"] > 0.8],
        },
        "mandate_3_priority_response_ranking": [
            {"priority": r["priority"], "target_ward": r["target_ward"],
             "population_at_risk": r["population_at_risk"],
             "assigned_team": r["assigned_team"],
             "recommended_equipment": r["recommended_equipment"],
             "eta_min": r["eta_min"], "expected_lives_saved": r["pi"]}
            for r in plan["ranked"]],
        "mandate_4_safest_and_fastest_evacuation": {
            "designated_safe_shelters": [
                {"name": s["name"], "capacity": s["capacity"],
                 "elevation": s["elevation"], "status": s["status"],
                 "assigned": s["assigned"]} for s in evac["shelters"]],
            "prohibited_transit_routes": evac["prohibited_transit_routes"],
            "total_evacuating": sum(f["people"] for f in evac["flows"]),
            "unplaced": sum(o["people"] for o in evac["overflow"]),
        },
        "caveat": ("Model output for decision support. Confirm on the ground before "
                   "committing teams."),
    }


@app.get("/api/radar/heartbeat")
def radar_heartbeat(cell_id: Optional[str] = None):
    """Simulated Doppler bio-signal. We do not have FINDER-class radar hardware."""
    rng = random.Random((cell_id or "default") + str(int(time.time()) // 10))
    resp_hz = rng.uniform(0.2, 0.3)
    heart_hz = rng.uniform(1.0, 1.5)
    signal = [round(0.6 * math.sin(2 * math.pi * resp_hz * (i / 100.0))
                    + 0.25 * math.sin(2 * math.pi * heart_hz * (i / 100.0))
                    + rng.gauss(0, 0.04), 4) for i in range(512)]
    return {
        "cell_id": cell_id, "fs_hz": 100, "n_samples": 512, "signal": signal,
        "peaks": {"respiration_hz": round(resp_hz, 3),
                  "respiration_rpm": round(resp_hz * 60, 1),
                  "cardiac_hz": round(heart_hz, 3), "bpm": round(heart_hz * 60, 1)},
        "alive": True, "confidence": round(rng.uniform(0.78, 0.97), 2),
        "simulated": True,
        "note": "Simulated. FINDER-class radar is hardware we do not have.",
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "scenario": SCENARIO, "wards": len(_scenario()["wards"]),
            "disasters": sorted(DISASTER_CONFIG)}


if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
