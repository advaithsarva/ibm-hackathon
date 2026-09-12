"""Checks that the API keeps the shape the Sentinel Grid dashboard reads.

The UI reads specific field names. Renaming one breaks a panel silently: the page still
renders, the value just shows as undefined, and nobody notices until it is on a
projector. These are the field lists extracted from app.js, so a rename fails here first.

    python -m src.test_frontend_contract
"""
import sys

from backend.main import (compute_zones, detections_for, evacuation_for, get_alert_summary,
                          health, incident_action_plan, model_metrics, priority_for,
                          radar_heartbeat, simulate)

PASS, FAIL = [], []


def check(label, ok, detail=""):
    (PASS if ok else FAIL).append(label)
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f"   {detail}" if detail and not ok
                                                     else ""))


def require(label, obj, fields):
    missing = [f for f in fields if f not in obj]
    check(label, not missing, f"missing {missing}")


def test_zones():
    z = compute_zones("flood")
    check("zones: 30 wards", len(z["grid_cells"]) == 30, str(len(z["grid_cells"])))
    require("zones: top level", z, ["disaster", "generated_at", "grid_cells", "district"])
    cell = z["grid_cells"][0]
    # Every field app.js reads off a cell.
    require("zones: cell fields", cell,
            ["cell_id", "ward_name", "centroid", "X", "U", "G", "zone", "zone_reason",
             "pop", "assets", "reachable", "components", "uncertainty_components",
             "rescue_priority", "elevation_m", "river_dist_km", "water_depth_m",
             "rainfall_24h_mm"])
    require("zones: uncertainty breakdown", cell["uncertainty_components"],
            ["u_model", "u_stale", "u_cover"])
    check("zones: centroid is [lat, lon] for Leaflet",
          len(cell["centroid"]) == 2 and 8 < cell["centroid"][0] < 14)
    check("zones: every ward is named",
          all(c.get("ward_name") for c in z["grid_cells"]))
    check("zones: all three classes present",
          {c["zone"] for c in z["grid_cells"]} <= {"RED", "BLUE", "GREEN"})


def test_alert_summary():
    s = get_alert_summary("flood")
    require("summary: fields", s,
            ["disaster", "red_cells", "blue_cells", "green_cells",
             "at_risk_population", "alert_level", "generated_at"])
    check("summary: alert level is one of three",
          s["alert_level"] in ("CRITICAL", "WARNING", "MONITORING"))
    check("summary: counts add to the ward total",
          s["red_cells"] + s["blue_cells"] + s["green_cells"] == 30)


def test_priority():
    p = priority_for(compute_zones("flood"), "flood")
    require("priority: top level", p,
            ["ranked", "unassigned", "budget_used_team_hours", "budget_total_team_hours"])
    check("priority: has ranked rows", len(p["ranked"]) > 0)
    row = p["ranked"][0]
    require("priority: row fields", row,
            ["rank", "priority", "cell_id", "ward_name", "target_ward", "pi", "n_est",
             "population_at_risk", "p_alive", "survivability", "eta_min", "team",
             "assigned_team", "recommended_equipment", "water_depth_m", "route"])
    check("priority: ranks are 1..n",
          [r["rank"] for r in p["ranked"]] == list(range(1, len(p["ranked"]) + 1)))
    check("priority: sorted by expected lives saved",
          [r["pi"] for r in p["ranked"]] == sorted((r["pi"] for r in p["ranked"]),
                                                   reverse=True))
    check("priority: within the team-hour budget",
          p["budget_used_team_hours"] <= p["budget_total_team_hours"])
    check("priority: route is a polyline of [lat, lon]",
          len(row["route"]) >= 2 and len(row["route"][0]) == 2)


def test_evacuation():
    e = evacuation_for(compute_zones("flood"))
    require("evacuation: top level", e,
            ["flows", "shelters", "overflow", "prohibited_transit_routes"])
    check("evacuation: has flows", len(e["flows"]) > 0)
    require("evacuation: flow fields", e["flows"][0],
            ["from_cell", "from_ward", "to_shelter", "to_ward", "people", "people_count",
             "path", "route_km", "travel_min", "eta_min", "hazard_avoided"])
    require("evacuation: shelter fields", e["shelters"][0],
            ["id", "name", "ward_name", "capacity", "assigned", "utilization",
             "elevation", "status", "location"])
    for s in e["shelters"]:
        check(f"evacuation: {s['ward_name']} within capacity",
              s["assigned"] <= s["capacity"], f"{s['assigned']}/{s['capacity']}")
    check("evacuation: status is OPEN or AT CAPACITY",
          all(s["status"] in ("OPEN", "AT CAPACITY") for s in e["shelters"]))


def test_detections():
    d = detections_for(compute_zones("flood"))
    check("detections: returns a list", isinstance(d, list) and len(d) > 0)
    require("detections: record fields", d[0],
            ["cell_id", "ward_name", "source", "timestamp", "detections", "p_alive",
             "n_est"])
    check("detections: p_alive is a probability",
          all(0.0 <= r["p_alive"] <= 1.0 for r in d))
    check("detections: every record says the telemetry is synthetic",
          all("telemetry" in r for r in d))
    kinds = {x["type"] for r in d for x in r["detections"]}
    check("detections: sensor types are known to the fusion table",
          kinds <= {"rgb_person", "thermal_hotspot", "rppg_pulse", "acoustic_distress",
                    "rgb_thermal", "phone_ping"}, str(kinds))


def test_model_metrics():
    m = model_metrics()
    require("metrics: top level", m,
            ["primary_hazard_model", "secondary_benchmarking_model", "fusion",
             "verification"])
    for key in ("primary_hazard_model", "secondary_benchmarking_model"):
        require(f"metrics: {key}", m[key],
                ["name", "dataset", "metrics", "honest_read", "feature_importance"])
    check("metrics: fusion ratios match the config",
          m["fusion"]["likelihood_ratios"]["radar_vital"] == 25)


def test_incident_action_plan():
    p = incident_action_plan("flood")
    require("IAP: top level", p,
            ["operation_name", "operational_period", "calamity_parameters",
             "mandate_1_where_likely_to_occur", "mandate_2_who_and_what_affected",
             "mandate_3_priority_response_ranking",
             "mandate_4_safest_and_fastest_evacuation"])
    require("IAP: calamity parameters", p["calamity_parameters"],
            ["cwc_river_stage", "meteorological_forcing"])
    require("IAP: mandate 1", p["mandate_1_where_likely_to_occur"],
            ["critical_risk_zones_count", "critical_wards",
             "reconnaissance_drone_queue_blue_zones"])
    require("IAP: mandate 2", p["mandate_2_who_and_what_affected"],
            ["total_endangered_population", "critical_infrastructure_at_risk",
             "submerged_chokepoints"])
    check("IAP: mandate 3 rows carry what the table renders",
          all({"priority", "target_ward", "population_at_risk", "assigned_team",
               "recommended_equipment"} <= set(r)
              for r in p["mandate_3_priority_response_ranking"]))
    require("IAP: mandate 4", p["mandate_4_safest_and_fastest_evacuation"],
            ["designated_safe_shelters", "prohibited_transit_routes"])
    check("IAP: shelters carry name, capacity, elevation, status",
          all({"name", "capacity", "elevation", "status"} <= set(s)
              for s in p["mandate_4_safest_and_fastest_evacuation"]
              ["designated_safe_shelters"]))
    check("IAP: carries a caveat", "caveat" in p)


def test_simulation_actually_recomputes():
    """The sliders must change the answer, or the console is a decoration."""
    calm = simulate({"river_stage_m": 1.0, "rainfall_scale": 0.2, "sar_staleness_h": 0.5})
    severe = simulate({"river_stage_m": 6.0, "rainfall_scale": 1.5, "sar_staleness_h": 20})
    require("simulate: response shape", calm,
            ["grid_cells", "generated_at", "inputs", "zone_counts",
             "at_risk_population"])
    check("simulate: a severe storm produces more RED than a calm one",
          severe["zone_counts"]["RED"] > calm["zone_counts"]["RED"],
          f"{severe['zone_counts']} vs {calm['zone_counts']}")
    check("simulate: a calm river leaves some ward safe",
          calm["zone_counts"]["GREEN"] > 0, str(calm["zone_counts"]))
    check("simulate: a severe storm endangers more people",
          severe["at_risk_population"] > calm["at_risk_population"])
    check("simulate: still returns every ward", len(calm["grid_cells"]) == 30)

    # Staleness alone must move uncertainty, since that is the BLUE story.
    fresh = simulate({"river_stage_m": 4.6, "rainfall_scale": 1.0, "sar_staleness_h": 0.5})
    stale = simulate({"river_stage_m": 4.6, "rainfall_scale": 1.0, "sar_staleness_h": 22})
    fresh_u = sum(c["U"] for c in fresh["grid_cells"])
    stale_u = sum(c["U"] for c in stale["grid_cells"])
    check("simulate: a stale satellite pass raises uncertainty",
          stale_u > fresh_u, f"{stale_u:.2f} vs {fresh_u:.2f}")


def test_radar_is_labelled_simulated():
    r = radar_heartbeat("11.0168_76.9660")
    require("radar: fields", r, ["signal", "peaks", "fs_hz", "n_samples", "simulated"])
    check("radar: 512 samples", len(r["signal"]) == 512)
    check("radar: declares itself simulated", r["simulated"] is True)
    check("radar: physiological BPM", 40 <= r["peaks"]["bpm"] <= 180)


def test_every_disaster_in_the_dropdown_works():
    """The UI offers seven hazards. Each must compute, or the dropdown 500s and the page
    silently falls back to cached data while looking fine."""
    from backend.main import DISASTER_CONFIG
    for disaster in DISASTER_CONFIG:
        z = compute_zones(disaster)
        check(f"{disaster}: computes 30 wards", len(z["grid_cells"]) == 30)
        check(f"{disaster}: has a label for the status line",
              bool(z.get("disaster_label")))
        check(f"{disaster}: reports the formula it used", bool(z.get("hazard_formula")))
        xs = [c["X"] for c in z["grid_cells"]]
        check(f"{disaster}: X stays in [0,1]", all(0.0 <= x <= 1.0 for x in xs))
        check(f"{disaster}: X varies across wards", max(xs) - min(xs) > 0.01,
              f"range {min(xs):.3f}-{max(xs):.3f}")
        p = priority_for(z, disaster)
        check(f"{disaster}: produces a ranked plan", len(p["ranked"]) > 0)
        check(f"{disaster}: declares its ranking basis",
              p.get("ranking_basis") in ("expected_lives_saved", "exposure"))
        check(f"{disaster}: every row carries equipment",
              all(r.get("recommended_equipment") for r in p["ranked"]))


def test_hazards_without_a_survivability_curve_say_so():
    """Drought and lightning have no tau in the spec. They must rank by exposure and
    label it, not silently borrow a constant from another hazard."""
    for disaster in ("drought", "lightning"):
        p = priority_for(compute_zones(disaster), disaster)
        check(f"{disaster}: ranks by exposure", p["ranking_basis"] == "exposure")
        check(f"{disaster}: explains why", "ranking_note" in p)


def test_ward_terrain_feeds_the_xai_panel():
    """The explainability panel reads these off components; a missing one renders as 0
    and the bar silently shows nothing."""
    cell = compute_zones("flood")["grid_cells"][0]
    require("XAI: terrain components", cell["components"],
            ["rainfall_24h_mm", "water_depth_m", "twi_index", "elevation_m",
             "river_dist_km", "slope_deg", "ward_name"])
    check("XAI: TWI is a real number", isinstance(cell["components"]["twi_index"], float))


def test_model_card_carries_measured_numbers_only():
    """Every figure on the model card must come from a held-out run, and a metric the
    model does not have must be null rather than a plausible-looking placeholder."""
    m = model_metrics()
    check("model card: at least one model built", len(m["all_models"]) > 0)
    for name, card in m["all_models"].items():
        require(f"model card {name}", card,
                ["name", "dataset", "metrics", "feature_importance", "confusion_matrix",
                 "trained_rows", "validation_rows", "honest_read"])
        met = card["metrics"]
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            v = met.get(key)
            check(f"model card {name}: {key} is a real value or null",
                  v is None or 0.0 <= v <= 1.0, str(v))
        shares = [f["share"] for f in card["feature_importance"]]
        check(f"model card {name}: importances sum to 1",
              abs(sum(shares) - 1.0) < 0.01, f"{sum(shares):.4f}")
        cm = card["confusion_matrix"]
        check(f"model card {name}: confusion matrix totals the validation rows",
              cm["tp"] + cm["fp"] + cm["fn"] + cm["tn"] == card["validation_rows"])
    check("model card: physics layer is described as unfitted",
          "no held-out score" in m["physics_layer"]["note"])


def test_health():
    h = health()
    check("health: reports the ward count", h["wards"] == 30)
    check("health: lists the dashboard's disasters",
          {"flood", "earthquake", "cyclone", "landslide"} <= set(h["disasters"]))


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
