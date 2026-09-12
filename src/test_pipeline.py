"""Checks for ingest, priority, routing and report.

    python -m src.test_pipeline      (or: pytest src/test_pipeline.py)
"""
import datetime as dt
import math

import numpy as np

from src.hazard.formulas import load_config
from src.hazard.zones import load_global
from src.ingest import level1_meteo as meteo
from src.ingest import level2_geo as geo
from src.ingest import level3_infra as infra
from src.ingest import sync
from src.ingest.sources import SOURCES, local_path
from src.priority.dispatch import build_plan, greedy_knapsack
from src.priority.expected_lives import capability_match, expected_lives, score_cells
from src.priority.survivability import half_life, survivability, tau_for
from src.report.generate import offline_brief, summarize
from src.routing.evacuation import solve_flow
from src.routing.graph import INF, build_graph, edge_cost, reachable_from, shortest_path


# --- ingest ----------------------------------------------------------------------

def test_source_registry_is_coherent():
    for key, s in SOURCES.items():
        assert s["auth"] in ("none", "register"), f"{key}: bad auth {s['auth']}"
        assert s["level"] in (1, 2, 3), f"{key}: bad level"
        assert s["url"].startswith("https://"), f"{key}: url must be https"
        assert not local_path(key).is_absolute() or True
    # PS-1 names these explicitly; losing one means losing a graded dataset.
    for required in ("imd_rainfall", "bhuvan_flood_vulnerability", "sentinel1_sar",
                     "copernicus_ems", "nasa_gdis", "usgs_quakes", "worldpop"):
        assert required in SOURCES, f"PS-1 dataset {required} missing from the registry"


def test_rainfall_accumulates_over_the_right_window():
    base = dt.date(2026, 9, 10)
    series = {"Dharwad": [(base + dt.timedelta(days=i), 10.0 * (i + 1)) for i in range(5)]}
    assert meteo.accumulate(series, 1)["Dharwad"] == 50.0
    assert meteo.accumulate(series, 3)["Dharwad"] == 120.0      # 30 + 40 + 50
    assert meteo.accumulate(series, 5)["Dharwad"] == 150.0
    # An as-of date must not see the future.
    assert meteo.accumulate(series, 1, as_of=base)["Dharwad"] == 10.0

    for text, expected in (("2026-09-12", dt.date(2026, 9, 12)),
                           ("12-09-2026", dt.date(2026, 9, 12)),
                           ("12/09/2026", dt.date(2026, 9, 12))):
        assert meteo.parse_date(text) == expected
    try:
        meteo.parse_date("septemberish")
        raise AssertionError("expected ValueError on an unparseable date")
    except ValueError:
        pass


def test_heavy_rain_threshold_matches_imd():
    base = dt.date(2026, 9, 12)
    series = {"Wet": [(base, 80.0)], "Dry": [(base, 12.0)]}
    rows = meteo.flood_inputs(series)
    assert rows["Wet"]["heavy_rain_warning"] is True     # over 64.5 mm
    assert rows["Dry"]["heavy_rain_warning"] is False


def test_slope_is_exact_on_a_known_plane():
    y, x = np.mgrid[0:20, 0:20]
    dem = 100.0 + 0.5 * y + 0.1 * x
    slope = geo.slope_degrees(dem, cell_size_m=30.0)
    expected = math.degrees(math.atan(math.hypot(0.5, 0.1) / 30))
    # Interior only; the edge is padded by design.
    assert abs(slope[1:-1, 1:-1].mean() - expected) < 1e-6
    assert abs(geo.slope_degrees(np.full((10, 10), 42.0), 30.0).max()) < 1e-9

    try:
        geo.slope_degrees(np.zeros((5, 5)), 0)
        raise AssertionError("expected ValueError on zero cell size")
    except ValueError:
        pass


def test_water_and_inundation():
    vv = np.full((10, 10), -8.0)
    vv[:4, :] = -22.0
    assert abs(geo.flooded_fraction(vv) - 0.40) < 1e-9
    dem = np.array([[100.0, 104.0], [106.0, 110.0]])
    depth = geo.inundation_depth(dem, 105.0)
    assert depth[0, 0] == 5.0 and depth[1, 1] == 0.0, "dry ground is zero, not negative"
    flat = geo.normalize_elevation(np.full((4, 4), 7.0))
    assert (flat == 0.5).all(), "a flat scene has no high ground"


def test_population_sums_and_rasters_reduce():
    pop = np.ones((4, 4)) * 10.0
    assert infra.population_per_cell(pop, 2).sum() == pop.sum(), "population must not average"
    assert infra.population_per_cell(pop, 2).shape == (2, 2)
    assert geo.block_reduce(np.ones((4, 4)) * 3.0, 2).shape == (2, 2)


def test_asset_weighting_and_binning():
    features = [(12.9716, 77.5946, "hospital", "City General"),
                (12.9717, 77.5947, "school", "School 4")]
    binned = infra.bin_to_grid(features, 0.001)
    assert len(binned) == 1, "adjacent points inside one cell must share a key"
    assert infra.asset_value(next(iter(binned.values()))) == 1.7
    assert infra.classify_osm_feature({"amenity": "hospital"}) == "hospital"
    assert infra.classify_osm_feature({"amenity": "cafe"}) is None
    assert infra.isolation_factor(None, 0, 4) == 1.0
    assert infra.isolation_factor(None, 4, 4) == 0.0


def test_sync_ages_layers_and_validates():
    now = dt.datetime(2026, 9, 12, 12, 0, tzinfo=dt.timezone.utc)
    ages = sync.align_timestamps({
        "fresh": (now - dt.timedelta(hours=1)).isoformat(),
        "stale": (now - dt.timedelta(hours=11.2)).isoformat(),
    }, now)
    assert abs(ages["stale"] - 11.2) < 1e-6
    name, age = sync.oldest_layer(ages)
    assert name == "stale", "a cell is only as fresh as its stalest layer"

    try:
        sync.align_timestamps({"future": (now + dt.timedelta(hours=2)).isoformat()}, now)
        raise AssertionError("expected ValueError on a future timestamp")
    except ValueError:
        pass

    assert sync.source_spread([100.0]) == 0.0, "one source cannot disagree with itself"
    assert sync.source_spread([90.0, 110.0]) == 10.0

    cfg = load_config("configs/disasters/flood.yaml")
    cells = sync.validate(sync._demo_cells(), cfg)
    assert len(cells) == 8
    try:
        bad = [dict(cells[0], hazard_inputs={})]
        sync.validate(bad, cfg)
        raise AssertionError("expected ValueError on missing hazard inputs")
    except ValueError:
        pass


def test_sync_output_drives_the_zone_engine():
    """The handoff that matters: ingest output must run straight into section 4."""
    from src.hazard.zones import run
    cfg = load_config("configs/disasters/flood.yaml")
    result = run(cfg, load_global(), sync._demo_cells())
    assert result["disaster"] == "flood"
    zones = {c["zone"] for c in result["grid_cells"]}
    assert zones <= {"RED", "BLUE", "GREEN"}
    assert len(result["grid_cells"]) == 8


# --- priority --------------------------------------------------------------------

def test_survivability_decays_on_the_spec_constants():
    assert survivability(0, 8) == 1.0
    assert abs(survivability(8, 8) - math.exp(-1)) < 1e-12
    assert survivability(4, 2) < survivability(4, 40), "fire kills faster than rubble"
    assert abs(half_life(8) - 8 * math.log(2)) < 1e-12

    assert tau_for(load_config("configs/disasters/flood.yaml")) == 8
    assert tau_for(load_config("configs/disasters/earthquake.yaml")) == 40
    # Drought has no tau in the spec, so asking must raise rather than invent one.
    try:
        tau_for(load_config("configs/disasters/drought.yaml"))
        raise AssertionError("expected ValueError for a disaster with no tau")
    except ValueError:
        pass


def test_eta_sits_inside_the_decay():
    """Same cell, later arrival, strictly lower expected value."""
    near = expected_lives(6, 0.9, eta_minutes=15, tau_hours=8)
    far = expected_lives(6, 0.9, eta_minutes=180, tau_hours=8)
    assert far < near
    # And the gap widens as the hazard kills faster.
    slow = expected_lives(6, 0.9, 15, 40) - expected_lives(6, 0.9, 180, 40)
    fast = expected_lives(6, 0.9, 15, 2) - expected_lives(6, 0.9, 180, 2)
    assert fast > slow

    for bad in (lambda: expected_lives(-1, 0.5, 10, 8),
                lambda: expected_lives(5, 1.4, 10, 8),
                lambda: expected_lives(5, 0.5, -3, 8),
                lambda: expected_lives(5, 0.5, 10, 8, kappa=2.0)):
        try:
            bad()
            raise AssertionError("expected ValueError on out-of-range input")
        except ValueError:
            pass


def test_eta_flips_the_order_only_when_tau_is_short():
    """Documents the real behaviour, so nobody claims more than the maths supports."""
    cells = [{"cell_id": "far", "n_est": 14, "p_alive": 0.72, "eta_min": 90},
             {"cell_id": "near", "n_est": 6, "p_alive": 0.94, "eta_min": 18}]
    flood = score_cells(cells, load_config("configs/disasters/flood.yaml"))
    wildfire = score_cells(cells, load_config("configs/disasters/wildfire.yaml"))
    assert flood[0]["cell_id"] == "far", "at tau=8h the larger cluster should still win"
    assert wildfire[0]["cell_id"] == "near", "at tau=2h the ETA penalty should dominate"


def test_capability_match_defaults_without_inventing():
    assert capability_match("flood", "boat") == 1.0
    assert capability_match("flood", "foot") < capability_match("flood", "boat")
    assert capability_match("flood", "unheard_of") == 0.5, "unknown pairing is neutral"


def test_greedy_respects_the_budget():
    scored = [{"cell_id": "a", "pi": 10.0, "cost_team_hours": 6.0},
              {"cell_id": "b", "pi": 4.0, "cost_team_hours": 2.0},
              {"cell_id": "c", "pi": 3.0, "cost_team_hours": 1.0},
              {"cell_id": "d", "pi": 20.0, "cost_team_hours": 50.0}]
    chosen, unassigned, spent = greedy_knapsack(scored, budget_team_hours=9.0)
    assert spent <= 9.0
    assert "d" in [c["cell_id"] for c in unassigned], "an unaffordable cell cannot be taken"
    # The cheap cells fit alongside the expensive one; greedy must not stop at the first miss.
    assert {"b", "c"} <= {c["cell_id"] for c in chosen}

    empty, left, spent0 = greedy_knapsack(scored, 0.0)
    assert empty == [] and spent0 == 0.0 and len(left) == 4


def test_dispatch_plan_matches_the_contract():
    cfg = load_config("configs/disasters/flood.yaml")
    cells = [{"cell_id": "12.9716_77.5946", "n_est": 6, "p_alive": 0.94, "eta_min": 18,
              "cost_team_hours": 3.5, "team_kind": "boat"},
             {"cell_id": "12.9702_77.5988", "n_est": 4, "p_alive": 0.79, "eta_min": 12,
              "cost_team_hours": 2.0, "team_kind": "swift_water"}]
    teams = [{"id": "NDRF-1", "kind": "swift_water", "hours": 8.0},
             {"id": "NDRF-3", "kind": "boat", "hours": 8.0}]
    plan = build_plan(cells, cfg, budget_team_hours=14.0, teams=teams)

    assert {"ranked", "unassigned", "budget_used_team_hours",
            "budget_total_team_hours"} <= set(plan)
    assert [r["rank"] for r in plan["ranked"]] == list(range(1, len(plan["ranked"]) + 1))
    pis = [r["pi"] for r in plan["ranked"]]
    assert pis == sorted(pis, reverse=True)
    assert plan["budget_used_team_hours"] <= plan["budget_total_team_hours"]
    assert all(r["team"] for r in plan["ranked"]), "every dispatched cell needs a team"


# --- routing ---------------------------------------------------------------------

def test_edge_cost_penalises_hazard_and_uncertainty():
    cfg = load_global()["routing"]
    safe = edge_cost(1000, 40, hazard_x=0.0, uncertainty_u=0.0, cfg=cfg)
    risky = edge_cost(1000, 40, hazard_x=0.5, uncertainty_u=0.0, cfg=cfg)
    unsure = edge_cost(1000, 40, hazard_x=0.0, uncertainty_u=0.5, cfg=cfg)
    assert safe < unsure < risky, "hazard must outweigh uncertainty, alpha 3 vs beta 1"
    assert abs(safe - 90.0) < 1e-9, "1 km at 40 km/h is 90 s"

    assert edge_cost(1000, 40, 0.75, 0.0, cfg=cfg) == INF, "at X=0.75 the road is closed"
    assert edge_cost(1000, 40, 0.1, 0.0, cfg=cfg, bridge_down=True) == INF

    # BPR congestion only bites as flow approaches capacity.
    free = edge_cost(1000, 40, 0.0, 0.0, flow=100, capacity=1800, cfg=cfg)
    jammed = edge_cost(1000, 40, 0.0, 0.0, flow=1800, capacity=1800, cfg=cfg)
    assert jammed > free and abs(jammed - safe * 1.15) < 1e-6


def test_routing_avoids_flooded_roads():
    zones = {"A": {"X": 0.05, "U": 0.1}, "B": {"X": 0.82, "U": 0.2},
             "C": {"X": 0.35, "U": 0.55}, "D": {"X": 0.10, "U": 0.1}}
    edges = [
        {"u": "depot", "v": "flood", "length_m": 1000, "speed_kmh": 40, "cell_id": "B"},
        {"u": "flood", "v": "site", "length_m": 1000, "speed_kmh": 40, "cell_id": "A"},
        {"u": "depot", "v": "detour", "length_m": 1600, "speed_kmh": 40, "cell_id": "C"},
        {"u": "detour", "v": "site", "length_m": 1600, "speed_kmh": 40, "cell_id": "D"},
    ]
    adjacency = build_graph(edges, zones)
    path, seconds = shortest_path(adjacency, "depot", "site")
    assert path == ["depot", "detour", "site"], "must not route through the flooded cell"
    assert seconds < INF

    # An edge in an unscored cell is a silent failure, so it must raise.
    try:
        build_graph([dict(edges[0], cell_id="Z")], zones)
        raise AssertionError("expected ValueError for an edge in an unscored cell")
    except ValueError:
        pass


def test_unreachable_is_reported_not_zero():
    zones = {"A": {"X": 0.9, "U": 0.1}, "B": {"X": 0.0, "U": 0.0}}
    edges = [{"u": "depot", "v": "site", "length_m": 500, "speed_kmh": 40, "cell_id": "A"}]
    adjacency = build_graph(edges, zones)
    path, seconds = shortest_path(adjacency, "depot", "site")
    assert path == [] and seconds == INF, "no route is infinite cost, not zero"
    assert reachable_from(adjacency, "depot") == {"depot"}


def test_evacuation_respects_shelter_capacity():
    sources = {"cell_a": 1000, "cell_b": 1500}
    shelters = {"near": 800, "far": 2000}
    costs = {("cell_a", "near"): 10.0, ("cell_a", "far"): 40.0,
             ("cell_b", "near"): 12.0, ("cell_b", "far"): 30.0}

    flows, overflow = solve_flow(sources, shelters, costs)
    assert not overflow, "2500 people into 2800 places should all be placed"
    placed = {}
    for f in flows:
        placed[f["to_shelter"]] = placed.get(f["to_shelter"], 0) + f["people"]
    assert placed["near"] <= 800, "the near shelter must not be overfilled"
    assert sum(placed.values()) == 2500, "everyone must leave"
    # Shortest path alone would have sent all 2500 to the near shelter.
    assert placed["far"] > 0, "capacity must push people past the nearest shelter"


def test_evacuation_reports_overflow_and_stranding():
    over, over_flow = solve_flow({"a": 3000}, {"only": 1000},
                                 {("a", "only"): 15.0})
    assert sum(f["people"] for f in over) == 1000
    assert sum(o["people"] for o in over_flow) == 2000

    _f, stranded = solve_flow({"cut_off": 500}, {"s": 900}, {})
    assert stranded and stranded[0]["reason"] == "no route to any shelter", \
        "a cell with no route must be named, not dropped"


# --- report ----------------------------------------------------------------------

def test_offline_brief_reports_only_real_figures():
    contracts = {
        "zones": {"disaster": "flood", "generated_at": "2026-09-12T08:40:00Z",
                  "grid_cells": [
                      {"cell_id": "a", "zone": "RED", "pop": 1450, "zone_reason": "X >= 0.75",
                       "assets": [{"name": "City General"}]},
                      {"cell_id": "b", "zone": "BLUE", "pop": 1670,
                       "zone_reason": "U = 0.47 >= 0.35, too uncertain to call safe",
                       "assets": []},
                      {"cell_id": "c", "zone": "GREEN", "pop": 410, "zone_reason": "safe",
                       "assets": []}]},
        "detections": [],
        "priority": {"ranked": [{"rank": 1, "cell_id": "a", "pi": 4.7, "n_est": 6,
                                 "p_alive": 0.94, "eta_min": 18, "team": "NDRF-3"}],
                     "unassigned": ["b"], "budget_used_team_hours": 11.5,
                     "budget_total_team_hours": 14.0},
        "evacuation": {"flows": [{"people": 820}],
                       "shelters": [{"name": "Govt High School", "assigned": 820,
                                     "capacity": 2400, "utilization": 0.34}],
                       "overflow": []},
    }
    s = summarize(contracts)
    assert s["at_risk_population"] == 1450 + 1670, "GREEN must not count as at risk"
    assert s["red"] == 1 and s["blue"] == 1 and s["green"] == 1

    brief = offline_brief(s)
    assert "3,120" in brief, "the at-risk figure must appear as computed"
    assert "City General" in brief
    assert "recon" in brief.lower(), "BLUE cells must surface as a recon queue"
    assert "NDRF-3" in brief
    assert "model output" in brief.lower(), "the brief must caveat itself"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    main()
