"""Ward-level scenario model (Coimbatore monsoon, the PS-1 case).

The grid engine works in 100 m cells keyed by centroid. An operations director works in
wards with names. This maps between the two: each ward carries a real centroid, so it is
a grid cell like any other, and it also carries the name, elevation, population and river
distance a control room needs on screen.

Distance to the Noyyal is computed from the river's course rather than stored per ward,
because it is the strongest single driver of inundation here and it must stay consistent
when a ward moves.

    python -m src.scenario --list
    python -m src.scenario --build coimbatore_flood -o data/cache/cell_inputs.cbe.json
"""
import argparse
import json
import math
import pathlib

from src.grid import cell_id
from src.hazard.formulas import REPO_ROOT

SCENARIOS = REPO_ROOT / "data" / "scenarios"

# The Noyyal's own course, west to east across the south of the city, past the Ukkadam
# tank. These are river points, deliberately not ward centroids: reusing ward coordinates
# as the channel puts those wards at zero distance from it, which is circular and makes
# the strongest driver of inundation meaningless.
NOYYAL = [(10.9750, 76.8850), (10.9800, 76.9200), (10.9845, 76.9560),
          (10.9900, 76.9850), (10.9980, 77.0250), (11.0080, 77.0620),
          (11.0200, 77.1100)]

# Bankfull is the level at which the Noyyal begins to spill into the adjoining wards.
# Above this the river is the hazard rather than the drainage.
BANKFULL_M = 3.2


def haversine_km(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def distance_to_river_km(point, course=NOYYAL):
    """Shortest distance from a ward centroid to the river course.

    Measured to the nearest segment, not the nearest vertex: a ward sitting beside a long
    straight reach is close to the river even when it is far from either end of it.
    """
    best = float("inf")
    for i in range(len(course) - 1):
        best = min(best, _point_to_segment_km(point, course[i], course[i + 1]))
    return round(best, 3)


def _point_to_segment_km(p, a, b):
    # Local flat-earth projection is fine over a district; degrees to km at this latitude.
    lat_km, lon_km = 111.0, 111.0 * math.cos(math.radians(p[0]))
    px, py = p[1] * lon_km, p[0] * lat_km
    ax, ay = a[1] * lon_km, a[0] * lat_km
    bx, by = b[1] * lon_km, b[0] * lat_km
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def inundation_depth_m(ward, river_stage_m, rainfall_scale=1.0):
    """Standing water in a ward, from river stage, local relief and distance to channel.

    Two contributions. The river spills once it passes bankfull, and the spill reaches
    less far the further a ward sits from the channel and the higher it stands above the
    floodplain. Pluvial flooding from rainfall alone is added on top, since Coimbatore's
    low-lying wards flood from drainage failure without the river doing anything.
    """
    overtop = max(0.0, river_stage_m - BANKFULL_M)
    reach_km = 1.2 + 0.9 * overtop                      # spill reaches further as it rises
    if ward["river_dist_km"] < reach_km and overtop > 0:
        attenuation = 1.0 - (ward["river_dist_km"] / reach_km)
        fluvial = overtop * attenuation * max(0.0, 1.0 - ward["relief_m"] / 12.0)
    else:
        fluvial = 0.0

    # Pluvial: drainage-limited, worse in low relief, independent of the river.
    pluvial = 0.55 * rainfall_scale * max(0.0, 1.0 - ward["relief_m"] / 18.0)

    return round(fluvial + pluvial, 2)


def load_scenario(name):
    path = SCENARIOS / f"{name}.json"
    if not path.exists():
        available = ", ".join(p.stem for p in SCENARIOS.glob("*.json")) or "none"
        raise FileNotFoundError(f"scenario {name!r} not found. Available: {available}")
    scenario = json.loads(path.read_text(encoding="utf-8"))
    for ward in scenario["wards"]:
        ward.setdefault("river_dist_km",
                        distance_to_river_km((ward["lat"], ward["lon"])))
    return scenario


def slope_deg_for(ward):
    """Ground slope in degrees, from the scenario if it carries one.

    The fallback derives slope from relief spread over a ward roughly 2 km across. An
    earlier version read 22 m of relief as a 37 degree slope, which is a cliff; over 2 km
    it is well under one degree. Coimbatore is a plateau, and the only real slope is the
    Western Ghats margin on its western edge, so those wards carry a measured value.
    """
    if "slope_deg" in ward:
        return ward["slope_deg"]
    return round(max(0.3, ward["relief_m"] / 2000.0 * 57.3), 2)


def topographic_wetness_index(ward):
    """TWI = ln(a / tan(beta)): where water collects.

    `a` is the upslope area draining through a point, approximated here from local
    relief, since a ward on the floodplain drains a far larger catchment than one on a
    ridge. High TWI is flat low ground that water runs into and does not leave, which is
    exactly where urban flooding shows up first.
    """
    a = 120.0 / (ward["relief_m"] + 1.0)
    beta = math.radians(max(0.6, slope_deg_for(ward)))
    return round(math.log(a / math.tan(beta)), 3)


def hazard_inputs_for(disaster, ward, r24, depth, retention, events):
    """The inputs each disaster's formula declares, derived for one ward.

    One ward model, several hazards. Each uses the physics already in
    src/hazard/physics.py rather than a second set of constants: the earthquake path
    runs the GMPE, the cyclone path runs the Holland wind field.
    """
    from src.hazard.physics import amplify_vs30, gmpe_pga, holland_wind

    if disaster in ("flood", "flood_rainfall"):
        return {"r24_mm": round(r24 * retention + depth * 64.0, 1)}

    if disaster == "landslide":
        event = events.get("landslide", {})
        return {"r3d_mm": round(r24 * event.get("r3d_multiplier", 2.2), 1),
                "slope_deg": slope_deg_for(ward)}

    if disaster == "earthquake":
        event = events["earthquake"]
        distance = haversine_km((ward["lat"], ward["lon"]), event["epicentre"])
        pga = gmpe_pga(event["magnitude_mw"], distance, event["depth_km"])
        # Soft floodplain soil amplifies; the ridge wards sit closer to rock.
        vs30 = 200.0 + ward["relief_m"] * 22.0
        return {"pga_ms2": round(amplify_vs30(pga, vs30), 4)}

    if disaster == "cyclone":
        event = events["cyclone"]
        distance = max(1.0, haversine_km((ward["lat"], ward["lon"]), event["centre"]))
        wind_ms = holland_wind(distance, event["central_pressure_hpa"],
                               event["ambient_pressure_hpa"], event["rmw_km"],
                               event.get("b", 1.4))
        return {"wind_kmh": round(wind_ms * 3.6, 1)}

    if disaster == "drought":
        event = events["drought"]
        # Well-drained high ground dries out first; the floodplain holds moisture.
        smi = event["smi_base"] * (0.6 + 0.8 * retention)
        return {"smi": round(min(1.0, max(0.0, smi)), 4)}

    if disaster == "wildfire":
        event = events["wildfire"]
        # Fire weather rises where water does not linger, so it tracks 1 - retention.
        return {"fwi": round(event["fwi_base"] * (0.5 + 1.0 * (1.0 - retention)), 2)}

    if disaster == "lightning":
        event = events["lightning"]
        return {"cape_j_kg": round(event["cape_base"] * ward.get("rain_factor", 1.0), 1)}

    raise ValueError(f"no hazard inputs defined for disaster {disaster!r}")


def build_cells(scenario, river_stage_m=None, rainfall_scale=1.0, sar_staleness_h=None,
                disaster="flood"):
    """Scenario + storm state -> cell inputs for the zone engine.

    river_stage_m, rainfall_scale and sar_staleness_h are the three simulation controls
    the dashboard exposes. Defaults come from the scenario's own current state.
    """
    state = scenario["state"]
    river_stage_m = state["river_stage_m"] if river_stage_m is None else river_stage_m
    sar_staleness_h = (state["sar_staleness_h"] if sar_staleness_h is None
                       else sar_staleness_h)
    r24_base = state["rainfall_24h_mm"]

    cells = []
    for ward in scenario["wards"]:
        r24 = round(r24_base * rainfall_scale * ward.get("rain_factor", 1.0), 1)
        depth = inundation_depth_m(ward, river_stage_m, rainfall_scale)

        # A ward with a gauge or a ground report is fresh; the rest inherit the SAR pass.
        has_ground_truth = ward.get("gauge") or ward.get("shelter")
        age_h = 0.5 if has_ground_truth else sar_staleness_h

        # Rainfall is near-uniform across a district; flooding is not. What separates
        # wards is how much of that rain stays: a ward 20 m above the floodplain sheds
        # it, a ward 1.5 m above retains it. Feeding raw rainfall to the hazard formula
        # scores every ward identically and the map comes out uniformly red, which is
        # the same failure as having no model at all.
        retention = min(1.0, max(0.15, 1.0 - ward["relief_m"] / 25.0))
        effective_mm = round(r24 * retention + depth * 64.0, 1)

        # Two independent estimates of the SAME quantity, both in X units on the same
        # scale, so their spread is genuine disagreement. Comparing a rainfall-derived
        # score against a depth-derived one on different denominators guarantees they
        # differ and saturates u_model on every ward, which is disagreement invented by
        # the code rather than found in the data.
        x_rain = min(1.0, (r24 * retention) / 160.0)
        x_depth = min(1.0, (depth * 64.0) / 160.0)
        sigma = abs(x_rain - x_depth) / 2.0

        # Sensor coverage is about what has actually been observed, not geography. A ward
        # with a gauge or staff on site is well covered; a deeply flooded ward is poorly
        # covered because drones cannot launch and crews cannot walk it. Tying coverage
        # to river distance, as an earlier version did, made remote high ground look
        # unobserved when it is simply far away.
        if has_ground_truth:
            sensed = 0.95
        elif depth > 1.5:
            sensed = 0.55                      # cannot get eyes into a submerged ward
        else:
            sensed = 0.80

        # Section 4.3's w5 is how far responders and evacuees must travel to reach the
        # cell, measured from the population centre. Using distance to the river instead
        # penalises exactly the high-ground wards that make the best shelters.
        travel_km = haversine_km((ward["lat"], ward["lon"]), scenario["centre"])

        cells.append({
            "cell_id": cell_id(ward["lat"], ward["lon"]),
            "ward_name": ward["name"],
            "centroid": [ward["lat"], ward["lon"]],
            "hazard_inputs": hazard_inputs_for(disaster, ward, r24, depth, retention,
                                               scenario.get("events", {})),
            "sensors": {"sigma_ens": round(sigma, 5),
                        "data_age_hours": round(age_h, 3),
                        "area_sensed_fraction": sensed},
            "green_inputs": {
                "forecast_max_x": round(min(1.0, max(x_rain, x_depth) * 1.15), 3),
                "elevation_norm": round(min(1.0, ward["relief_m"] / 25.0), 3),
                "shelter_capacity": ward.get("shelter_capacity", 0),
                # A designated relief centre is sized against its catchment, so its
                # capacity ratio is what the sizing achieved. Everywhere else the inflow
                # is the share of its own residents who would need placing.
                "pop_inflow": (ward.get("shelter_capacity") or max(1, int(ward["pop"] * 0.4))),
                "access": 0.15 if depth > 1.5 else 0.9,
                "travel_distance_norm": round(min(1.0, travel_km / 30.0), 3),
            },
            "pop": ward["pop"],
            "assets": ward.get("assets", []),
            "reachable": depth < 2.0,
            "components": {
                "ward_name": ward["name"],
                "elevation_m": ward["elevation_m"],
                "relief_m": ward["relief_m"],
                "river_dist_km": ward["river_dist_km"],
                "slope_deg": slope_deg_for(ward),
                "twi_index": topographic_wetness_index(ward),
                "travel_from_centre_km": round(travel_km, 2),
                "rainfall_24h_mm": r24,
                "effective_rainfall_mm": effective_mm,
                "runoff_retention": round(retention, 3),
                "water_depth_m": depth,
                "river_stage_m": river_stage_m,
                "bankfull_m": BANKFULL_M,
            },
        })
    return cells


def _cli():
    ap = argparse.ArgumentParser(description="Build cell inputs from a ward scenario.")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--build", default="coimbatore_flood")
    ap.add_argument("--river-stage", type=float)
    ap.add_argument("--rainfall-scale", type=float, default=1.0)
    ap.add_argument("--sar-staleness", type=float)
    ap.add_argument("-o", "--out")
    args = ap.parse_args()

    if args.list:
        for p in sorted(SCENARIOS.glob("*.json")):
            s = json.loads(p.read_text(encoding="utf-8"))
            print(f"{p.stem:<22} {len(s['wards'])} wards, {s['district']}, "
                  f"{s['disaster']}")
        return

    scenario = load_scenario(args.build)
    cells = build_cells(scenario, args.river_stage, args.rainfall_scale,
                        args.sar_staleness)
    text = json.dumps(cells, indent=2)
    if args.out:
        out = pathlib.Path(args.out)
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
        flooded = sum(1 for c in cells if c["components"]["water_depth_m"] > 0.3)
        print(f"{out.relative_to(REPO_ROOT)}: {len(cells)} wards, "
              f"{flooded} with standing water")
    else:
        print(text)


if __name__ == "__main__":
    _cli()
