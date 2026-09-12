"""Checks for the hazard and zone engine. No framework needed.

    python -m src.hazard.test_hazard      (or: pytest src/hazard/test_hazard.py)
"""
import json
import math

from src.grid import cell_id, centroid
from src.hazard import physics
from src.hazard.formulas import REPO_ROOT, hazard_score, load_config, tier
from src.hazard.zones import (Hysteresis, classify, green_suitability, load_global,
                              rescue_priority, run, u_cover, u_model, u_stale,
                              uncertainty)

TH = {"low": 0.40, "high": 0.75}


def test_cell_id_roundtrip():
    assert cell_id(12.97161, 77.59456) == "12.9716_77.5946"
    assert centroid("12.9716_77.5946") == (12.9716, 77.5946)
    # A malformed key must fail rather than silently produce a wrong join.
    try:
        centroid("not-a-cell")
        raise AssertionError("expected ValueError on malformed cell_id")
    except ValueError:
        pass


def test_hazard_formula_is_config_driven():
    eq = load_config("configs/disasters/earthquake.yaml")
    assert hazard_score(eq, {"pga_ms2": 3.28}) == 0.82
    assert hazard_score(eq, {"pga_ms2": 40.0}) == 1.0, "X must clamp at 1"

    # Landslide's weighted sum is not self-clamping in the spec; the evaluator clamps it.
    ls = load_config("configs/disasters/landslide.yaml")
    assert hazard_score(ls, {"r3d_mm": 400.0, "slope_deg": 60.0}) == 1.0

    # Missing data must raise. A zero here would claim "no hazard" for an unmeasured cell.
    try:
        hazard_score(eq, {})
        raise AssertionError("expected ValueError on missing hazard input")
    except ValueError:
        pass


def test_every_disaster_config_evaluates():
    """All seven configs must load and evaluate, or the config-driven design is a lie."""
    sample = {"pga_ms2": 2.0, "r24_mm": 150.0, "q_cumecs": 400.0, "q_max": 800.0,
              "wind_kmh": 180.0, "smi": 0.3, "r3d_mm": 120.0, "slope_deg": 35.0,
              "fwi": 30.0, "cape_j_kg": 2800.0}
    names = ["earthquake", "flood", "cyclone", "landslide", "drought", "wildfire",
             "lightning"]
    for name in names:
        cfg = load_config(f"configs/disasters/{name}.yaml")
        x = hazard_score(cfg, sample)
        assert 0.0 <= x <= 1.0, f"{name}: X={x} outside [0,1]"
    assert tier(0.80, TH) == "high" and tier(0.50, TH) == "medium" and tier(0.1, TH) == "low"


def test_uncertainty_is_noisy_or():
    assert u_model(0.0225, 0.25) == 0.09
    assert u_model(10.0, 0.25) == 1.0, "u_model must saturate at 1"
    assert abs(u_stale(0.0, 1.0)) < 1e-9, "fresh data carries no staleness"
    assert abs(u_stale(1.0, 1.0) - (1 - math.exp(-1))) < 1e-9
    assert u_cover(1.0) == 0.0 and u_cover(0.0) == 1.0

    # Compounding, not averaging: three small ignorances exceed any one of them.
    u = uncertainty(0.2, 0.2, 0.2)
    assert u > 0.2 and abs(u - (1 - 0.8 ** 3)) < 1e-9

    # Certainty on any single axis cannot rescue a cell nothing has looked at.
    assert uncertainty(0.0, 0.0, 1.0) == 1.0


def test_zone_boundaries():
    """The thresholds are exact, so test on them, not near them."""
    assert classify(0.75, 0.0, 0.0, True, TH) == "RED", "X = 0.75 is RED, not BLUE"
    assert classify(0.7499, 0.0, 0.0, True, TH) == "BLUE"
    assert classify(0.40, 0.50, 0.0, True, TH) == "RED", "moderate hazard + high U is RED"
    assert classify(0.40, 0.4999, 0.0, True, TH) == "BLUE"
    assert classify(0.39, 0.34, 0.60, True, TH) == "GREEN"
    assert classify(0.39, 0.35, 0.60, True, TH) == "BLUE", "U = 0.35 blocks GREEN"
    assert classify(0.39, 0.34, 0.5999, True, TH) == "BLUE", "G < 0.60 blocks GREEN"
    assert classify(0.0, 0.0, 1.0, False, TH) == "BLUE", "unreachable is never GREEN"


def test_hysteresis_damps_boundary_flicker():
    h = Hysteresis(margin=0.05, hold_cycles=2, thresholds=TH)
    assert h.update("c1", 0.30, 0.10, 0.80, True) == "GREEN"  # first sighting takes raw

    # A marginal crossing (inside the 0.05 band) must not move the cell at all.
    for _ in range(5):
        assert h.update("c1", 0.42, 0.10, 0.80, True) == "GREEN", "flipped inside the margin"

    # A decisive crossing still waits for 2 consecutive cycles before it lands.
    h2 = Hysteresis(margin=0.05, hold_cycles=2, thresholds=TH)
    h2.update("c2", 0.30, 0.10, 0.80, True)
    assert h2.update("c2", 0.90, 0.10, 0.80, True) == "GREEN", "changed on the first cycle"
    assert h2.update("c2", 0.90, 0.10, 0.80, True) == "RED"
    assert h2.update("c2", 0.90, 0.10, 0.80, True) == "RED", "should stay put once changed"

    # An interrupted run restarts the count rather than accumulating across a gap.
    h3 = Hysteresis(margin=0.05, hold_cycles=2, thresholds=TH)
    h3.update("c3", 0.30, 0.10, 0.80, True)
    assert h3.update("c3", 0.90, 0.10, 0.80, True) == "GREEN"
    assert h3.update("c3", 0.30, 0.10, 0.80, True) == "GREEN"
    assert h3.update("c3", 0.90, 0.10, 0.80, True) == "GREEN", "count must reset"


def test_green_suitability_uses_forecast_not_current():
    w = load_global()["green_suitability"]
    common = dict(elevation_norm=0.8, shelter_capacity=2000, pop_inflow=2000,
                  access=0.9, travel_distance_norm=0.1)
    safe_now_safe_later = green_suitability(forecast_max_x=0.1, **common, weights=w)
    safe_now_flooded_later = green_suitability(forecast_max_x=0.9, **common, weights=w)
    assert safe_now_safe_later > safe_now_flooded_later, \
        "a zone that floods in six hours must score below one that does not"
    assert 0.0 <= safe_now_flooded_later <= 1.0

    # Shelter capacity below expected arrivals must cost the cell something.
    tight = green_suitability(0.1, 0.8, 200, 2000, 0.9, 0.1, w)
    assert tight < safe_now_safe_later


def test_physics_behaves_physically():
    # Shaking attenuates with distance and grows with magnitude.
    near, far = physics.gmpe_pga(6.8, 5.0, 10.0), physics.gmpe_pga(6.8, 80.0, 10.0)
    assert near > far
    assert physics.gmpe_pga(7.5, 20.0, 10.0) > physics.gmpe_pga(6.0, 20.0, 10.0)
    # Soft soil amplifies relative to rock.
    assert physics.amplify_vs30(1.0, 200) > physics.amplify_vs30(1.0, 760)

    # Steeper and wetter slopes are less stable; flat ground never fails this way.
    assert physics.factor_of_safety(5, 18, 2, 20, 30) > physics.factor_of_safety(5, 18, 2, 45, 30)
    assert physics.factor_of_safety(5, 18, 2, 35, 30, 0.0) > \
        physics.factor_of_safety(5, 18, 2, 35, 30, 1.5)
    assert physics.factor_of_safety(5, 18, 2, 0.0, 30) == float("inf")

    # Cyclone pressure rises away from the eye and stays inside the ambient bound.
    assert physics.holland_pressure(10, 950, 1010, 25) < physics.holland_pressure(200, 950, 1010, 25)
    assert physics.holland_pressure(500, 950, 1010, 25) < 1010

    assert physics.vci(0.18, 0.18, 0.76) == 0.0
    assert physics.vci(0.76, 0.18, 0.76) == 100.0
    assert physics.flood_depth(3.0, 5.0) == 0.0, "dry ground is not negative depth"
    assert physics.manning_discharge(0.03, 120, 2.4, 0.001) > 0

    for bad in [lambda: physics.manning_discharge(0, 120, 2.4, 0.001),
                lambda: physics.vci(0.5, 0.8, 0.2),
                lambda: physics.holland_pressure(0, 950, 1010, 25)]:
        try:
            bad()
            raise AssertionError("expected ValueError on invalid physics input")
        except ValueError:
            pass


def test_rescue_priority_scales_with_hazard_and_exposure():
    w = load_global()["exposure"]
    a = rescue_priority(0.8, 1000, 1000, 1.0, 1.0, 0.0, 1.0, w)
    b = rescue_priority(0.4, 1000, 1000, 1.0, 1.0, 0.0, 1.0, w)
    assert a > b, "same exposure, double the hazard, must rank higher"
    assert rescue_priority(0.8, 0, 1000, 0.0, 1.0, 0.0, 1.0, w) == 0.0
    # No population anywhere must not divide by zero.
    assert rescue_priority(0.8, 0, 0, 0.0, 0.0, 0.0, 0.0, w) == 0.0


def test_engine_output_matches_the_frozen_contract():
    cells = json.loads((REPO_ROOT / "data/mock/cell_inputs.earthquake.json").read_text())
    result = run(load_config("configs/disasters/earthquake.yaml"), load_global(), cells)

    assert result["disaster"] == "earthquake"
    assert len(result["grid_cells"]) == len(cells)
    required = {"cell_id", "centroid", "X", "U", "G", "zone", "zone_reason", "pop",
                "assets", "reachable", "components"}
    for c in result["grid_cells"]:
        assert required <= set(c), f"{c['cell_id']}: missing {required - set(c)}"
        assert c["zone"] == classify(c["X"], c["U"], c["G"], c["reachable"], TH)
        uc = c["uncertainty_components"]
        assert abs(uncertainty(uc["u_model"], uc["u_stale"], uc["u_cover"]) - c["U"]) < 1e-3

    zones = [c["zone"] for c in result["grid_cells"]]
    assert set(zones) == {"RED", "BLUE", "GREEN"}, "fixture must exercise all three classes"

    # Every BLUE cell explains itself with a distinct binding constraint, which is the
    # thing the demo clicks on.
    blue_reasons = {c["zone_reason"] for c in result["grid_cells"] if c["zone"] == "BLUE"}
    assert len(blue_reasons) == zones.count("BLUE"), "BLUE cells give duplicate reasons"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    main()
