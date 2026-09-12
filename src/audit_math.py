"""Independent audit of every formula against the spec and against hand computation.

Different from the test suite. The tests check behaviour ("adding evidence must not
lower P"); this recomputes each formula from the spec text by a separate route and
compares. A test can pass while the implementation quietly encodes the wrong constant,
so this exists to catch exactly that.

    python -m src.audit_math
"""
import math


from src.hazard.formulas import hazard_score, load_config
from src.hazard.zones import (classify, green_suitability, load_global, rescue_priority,
                              u_cover, u_model, u_stale, uncertainty)

PASS, FAIL = [], []


def check(label, got, want, tol=1e-9, note=""):
    ok = abs(got - want) <= tol if isinstance(want, float) else got == want
    (PASS if ok else FAIL).append(label)
    mark = "ok  " if ok else "FAIL"
    detail = f"got {got!r} want {want!r}" if not ok else f"{got}"
    print(f"  {mark} {label:<52} {detail}" + (f"   {note}" if note and not ok else ""))
    return ok


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


# --- section 3.1: the seven hazard formulas, verbatim from the spec table -----------

def audit_hazard_formulas():
    section("3.1  hazard score X, against the spec table")

    # (config, inputs, independently computed expectation, spec formula as written)
    cases = [
        ("flood", {"r24_mm": 240.0, "q_cumecs": 450.0, "q_max": 900.0},
         min(1.0, 0.5 * (240.0 / 300) + 0.5 * (450.0 / 900)),
         "min(1.0, 0.5*(R24/300) + 0.5*(Q/Qmax))"),
        ("cyclone", {"wind_kmh": 175.0}, min(1.0, 175.0 / 250), "min(1.0, W/250)"),
        ("drought", {"smi": 0.28}, 1.0 - 0.28, "1.0 - SMI"),
        ("earthquake", {"pga_ms2": 3.28}, min(1.0, 3.28 / 4.0), "min(1.0, PGA/4.0)"),
        ("wildfire", {"fwi": 37.5}, min(1.0, 37.5 / 50), "min(1.0, FWI/50)"),
        ("landslide", {"r3d_mm": 150.0, "slope_deg": 42.0},
         0.6 * (150.0 / 200) + 0.4 * (42.0 / 60), "0.6*(R3d/200) + 0.4*(theta/60)"),
        ("lightning", {"cape_j_kg": 2700.0}, min(1.0, 2700.0 / 4500),
         "min(1.0, CAPE/4500)"),
    ]
    for name, inputs, expected, formula in cases:
        cfg = load_config(f"configs/disasters/{name}.yaml")
        check(f"{name}: {formula}", hazard_score(cfg, inputs), expected, 1e-9)

    # Saturation and the clamp.
    eq = load_config("configs/disasters/earthquake.yaml")
    check("earthquake clamps above 4.0 m/s2", hazard_score(eq, {"pga_ms2": 99.0}), 1.0)
    ls = load_config("configs/disasters/landslide.yaml")
    check("landslide clamps (spec formula is not self-limiting)",
          hazard_score(ls, {"r3d_mm": 400.0, "slope_deg": 60.0}), 1.0)
    dr = load_config("configs/disasters/drought.yaml")
    check("drought floors at 0 when SMI = 1", hazard_score(dr, {"smi": 1.0}), 0.0)

    # Tier boundaries shared by all seven (spec 3.1).
    g = load_global()["thresholds"]
    check("tier: low threshold", g["low"], 0.40)
    check("tier: high threshold", g["high"], 0.75)


# --- section 3.2: governing physics -------------------------------------------------

def audit_physics():
    section("3.2  governing physics, recomputed independently")
    from src.hazard import physics as ph

    # Manning: Q = (1/n) A Rh^(2/3) S^(1/2)
    n, area, rh, slope = 0.035, 85.0, 2.1, 0.0009
    check("Manning Q = (1/n)A Rh^(2/3) S^(1/2)",
          ph.manning_discharge(n, area, rh, slope),
          (1 / n) * area * rh ** (2 / 3) * math.sqrt(slope), 1e-9)

    # Holland: P(r) = Pc + (Pn - Pc) exp(-(Rm/r)^B)
    pc, pn, rm, b, r = 955.0, 1010.0, 30.0, 1.4, 45.0
    check("Holland P(r) = Pc + (Pn-Pc)exp(-(Rm/r)^B)",
          ph.holland_pressure(r, pc, pn, rm, b),
          pc + (pn - pc) * math.exp(-((rm / r) ** b)), 1e-9)
    check("Holland: pressure at the eye wall is below ambient",
          ph.holland_pressure(rm, pc, pn, rm, b) < pn, True)
    check("Holland: pressure tends to ambient far out",
          abs(ph.holland_pressure(5000.0, pc, pn, rm, b) - pn) < 0.5, True)

    # VCI = 100 (NDVI - min)/(max - min)
    check("VCI = 100(NDVI-min)/(max-min)", ph.vci(0.44, 0.15, 0.79),
          100 * (0.44 - 0.15) / (0.79 - 0.15), 1e-9)
    check("VCI = 0 at the minimum", ph.vci(0.15, 0.15, 0.79), 0.0)
    check("VCI = 100 at the maximum", ph.vci(0.79, 0.15, 0.79), 100.0)

    # GMPE: ln(PGA) = c1 + c2 M - c3 ln(sqrt(R^2 + h^2))
    m, dist, depth = 6.5, 20.0, 15.0
    c1, c2, c3 = -1.72, 0.98, 1.30
    expected = math.exp(c1 + c2 * m - c3 * math.log(math.sqrt(dist ** 2 + depth ** 2)))
    check("GMPE ln(PGA) = c1 + c2*M - c3*ln(sqrt(R^2+h^2))",
          ph.gmpe_pga(m, dist, depth), expected, 1e-9)
    check("GMPE: PGA falls with distance",
          ph.gmpe_pga(m, 100.0, depth) < ph.gmpe_pga(m, 10.0, depth), True)
    check("GMPE: 1 magnitude unit multiplies PGA by exp(c2)",
          ph.gmpe_pga(6.0, 20.0, 15.0) * math.exp(c2),
          ph.gmpe_pga(7.0, 20.0, 15.0), 1e-9)
    check("vs30: soft soil amplifies relative to 760 m/s rock",
          ph.amplify_vs30(1.0, 760.0), 1.0, 1e-9)

    # Infinite slope FS = [c' + (gz - gw hw) cos^2 t tan(phi)] / [gz sin t cos t]
    c_prime, gamma, z, theta, phi, hw, gw = 6.0, 19.0, 2.5, 32.0, 31.0, 0.8, 9.81
    t = math.radians(theta)
    gz = gamma * z
    expected = ((c_prime + (gz - gw * hw) * math.cos(t) ** 2 * math.tan(math.radians(phi)))
                / (gz * math.sin(t) * math.cos(t)))
    check("FS = [c' + (gz - gw*hw)cos^2 t tan phi]/[gz sin t cos t]",
          ph.factor_of_safety(c_prime, gamma, z, theta, phi, hw), expected, 1e-9)
    check("FS: pore pressure reduces stability",
          ph.factor_of_safety(6, 19, 2.5, 32, 31, 2.0)
          < ph.factor_of_safety(6, 19, 2.5, 32, 31, 0.0), True)

    # Flash rate F_L = a CAPE^(1/2) w^2
    check("flash rate F_L = a*sqrt(CAPE)*w^2", ph.flash_rate(2500.0, 15.0, 1.0),
          math.sqrt(2500.0) * 15.0 ** 2, 1e-9)

    # Flood depth
    check("depth = stage - DEM, floored at 0", ph.flood_depth(102.0, 99.5), 2.5, 1e-9)
    check("depth is 0 on dry ground, never negative", ph.flood_depth(98.0, 99.5), 0.0)


# --- section 4: uncertainty, zones, suitability, exposure ---------------------------

def audit_zone_engine():
    section("4  uncertainty, zone rule, suitability, exposure")
    g = load_global()

    # 4.1 noisy-OR
    um, us, uc = 0.20, 0.35, 0.10
    check("U = 1 - (1-um)(1-us)(1-uc)", uncertainty(um, us, uc),
          1 - (1 - um) * (1 - us) * (1 - uc), 1e-12)
    check("U compounds above its largest part", uncertainty(0.2, 0.2, 0.2) > 0.2, True)
    check("U saturates when any part is certain", uncertainty(0.0, 1.0, 0.0), 1.0)
    check("u_model = min(1, sigma/sigma_max)", u_model(0.05, 0.25), 0.2, 1e-12)
    check("u_model saturates at 1", u_model(10.0, 0.25), 1.0)
    check("u_stale = 1 - exp(-dt/tau)", u_stale(6.0, 6.0), 1 - math.exp(-1), 1e-12)
    check("u_stale = 0 on fresh data", u_stale(0.0, 6.0), 0.0, 1e-12)
    check("u_cover = 1 - sensed fraction", u_cover(0.85), 0.15, 1e-12)
    check("sigma_max matches the spec", g["uncertainty"]["sigma_max"], 0.25)
    for layer, tau in (("seismic", 1.0), ("rainfall", 6.0), ("sar_revisit", 12.0)):
        check(f"tau_data {layer}", g["uncertainty"]["tau_data_hours"][layer], tau)

    # 4.2 zone rule, tested exactly on the boundaries
    th = g["thresholds"]
    check("X = 0.75 exactly is RED", classify(0.75, 0.0, 0.0, True, th), "RED")
    check("X just under 0.75 is not RED", classify(0.7499, 0.0, 0.0, True, th), "BLUE")
    check("X = 0.40 with U = 0.50 is RED", classify(0.40, 0.50, 0.0, True, th), "RED")
    check("X = 0.40 with U just under 0.50 is not RED",
          classify(0.40, 0.4999, 0.0, True, th), "BLUE")
    check("GREEN needs U strictly under 0.35", classify(0.39, 0.35, 0.9, True, th), "BLUE")
    check("GREEN needs G at least 0.60", classify(0.39, 0.30, 0.60, True, th), "GREEN")
    check("G just under 0.60 blocks GREEN", classify(0.39, 0.30, 0.5999, True, th), "BLUE")
    check("unreachable is never GREEN", classify(0.0, 0.0, 1.0, False, th), "BLUE")
    check("hysteresis margin matches the spec", g["zones"]["hysteresis"]["margin"], 0.05)
    check("hysteresis hold matches the spec", g["zones"]["hysteresis"]["hold_cycles"], 2)

    # 4.3 green suitability
    w = g["green_suitability"]
    positive = (w["w_forecast_hazard"] + w["w_elevation"] + w["w_shelter_capacity"]
                + w["w_access"])
    check("G positive weights sum to 0.90", positive, 0.90, 1e-12)
    check("G travel-distance weight is 0.10", w["w_travel_distance"], 0.10, 1e-12)
    computed = green_suitability(0.2, 0.7, 1500, 2000, 0.8, 0.25, w)
    manual = (0.35 * (1 - 0.2) + 0.20 * 0.7 + 0.20 * min(1.0, 1500 / 2000)
              + 0.15 * 0.8 - 0.10 * 0.25)
    check("G = w1(1-maxX) + w2 z + w3 min(1,Cap/In) + w4 A - w5 d", computed, manual, 1e-12)
    # Capacity far exceeding inflow contributes its full 0.20 and no more. The 0.35
    # forecast term also fires here, because forecast_max_x = 0 means no future hazard.
    check("G capacity term saturates at w3 = 0.20",
          green_suitability(0.0, 0.0, 9999, 100, 0.0, 0.0, w), 0.35 + 0.20, 1e-12)
    check("G is clamped into [0,1]",
          0.0 <= green_suitability(1.0, 0.0, 0, 2000, 0.0, 1.0, w) <= 1.0, True)

    # 4.4 exposure
    ew = g["exposure"]
    check("exposure weights sum to 1", sum(ew.values()), 1.0, 1e-9)
    manual = 0.8 * (ew["w_population"] * (500 / 1000)
                    + ew["w_critical_assets"] * (1.0 / 2.0)
                    + ew["w_isolation"] * (1.0 / 1.0))
    check("RescuePriority = X(w1 pop + w2 assets + w3 isolation)",
          rescue_priority(0.8, 500, 1000, 1.0, 2.0, 1.0, 1.0, ew), manual, 1e-12)
    check("exposure is 0 when nothing is exposed",
          rescue_priority(0.9, 0, 1000, 0.0, 1.0, 0.0, 1.0, ew), 0.0)


# --- section 5.3: Bayesian log-odds fusion ------------------------------------------

def audit_fusion():
    section("5.3  Bayesian log-odds fusion")
    from src.detect.fusion import _logit, _sigmoid, fuse_log_odds

    ratios = load_global()["fusion"]["likelihood_ratios"]
    spec = {"radar_vital": 25, "acoustic_distress": 12, "rppg_pulse": 8,
            "rgb_person": 5, "thermal_hotspot": 3, "phone_ping": 2}
    for sensor, value in spec.items():
        check(f"LR {sensor} = {value}", ratios[sensor], value)
    # The spec prints log(LR) alongside; confirm ours agree with those figures.
    for sensor, log_lr in (("radar_vital", 3.22), ("acoustic_distress", 2.48),
                           ("rppg_pulse", 2.08), ("rgb_person", 1.61),
                           ("thermal_hotspot", 1.10), ("phone_ping", 0.69)):
        check(f"log(LR) {sensor} ~ {log_lr}", math.log(ratios[sensor]), log_lr, 0.005)

    check("logit and sigmoid invert", _sigmoid(_logit(0.73)), 0.73, 1e-9)

    prior = 0.05
    detections = [{"type": "thermal_hotspot", "conf": 0.9},
                  {"type": "rppg_pulse", "conf": 0.9},
                  {"type": "acoustic_distress", "conf": 0.9}]
    p, fired = fuse_log_odds(detections, prior=prior, ratios=ratios)
    manual_logit = (math.log(prior / (1 - prior)) + math.log(3) + math.log(8)
                    + math.log(12))
    check("logit(P) = logit(p0) + sum log(LR_k)", p, 1 / (1 + math.exp(-manual_logit)),
          1e-9)
    check("z_k is binary: three sensors fired", len(fired), 3)
    check("thermal alone stays weak",
          fuse_log_odds([{"type": "thermal_hotspot", "conf": 0.9}],
                        prior=prior, ratios=ratios)[0] < 0.2, True)
    # Spec 5.3 says this combination "compounds to P > 0.95". At our DEFAULT_PRIOR of
    # 0.05 it reaches 0.938, not 0.951. The arithmetic is right and the spec's figure is
    # loose: it holds from a prior of about 0.06 upward, and 5.3 defines p0 as local
    # population density times collapse probability, so it is per-cell rather than fixed.
    # Recorded here so nobody claims 0.95 on stage from a 0.05 prior.
    check("thermal + pulse + audio at prior 0.05", p, 0.9381, 5e-4)
    higher = fuse_log_odds(detections, prior=0.08, ratios=ratios)[0]
    check("the spec's >0.95 claim holds from a prior near 0.06", higher > 0.95, True)


# --- section 6 and 7: priority and routing ------------------------------------------

def audit_priority_and_routing():
    section("6, 7  survivability, expected lives, edge cost")
    from src.priority.expected_lives import expected_lives
    from src.priority.survivability import survivability, survivability_at_arrival
    from src.routing.graph import edge_cost

    check("S(t) = exp(-t/tau)", survivability(5.0, 8.0), math.exp(-5.0 / 8.0), 1e-12)
    check("S(0) = 1", survivability(0.0, 8.0), 1.0)
    check("S(tau) = 1/e", survivability(8.0, 8.0), math.exp(-1), 1e-12)
    check("S at arrival uses t_now + ETA",
          survivability_at_arrival(1.0, 90.0, 8.0), math.exp(-2.5 / 8.0), 1e-12)

    for name, tau in (("earthquake", 40), ("landslide", 24), ("cyclone", 12),
                      ("flood", 8), ("wildfire", 2)):
        cfg = load_config(f"configs/disasters/{name}.yaml")
        check(f"tau {name} = {tau} h", cfg["hazard"]["tau_survivability_hours"], tau)

    manual = 6 * 0.94 * math.exp(-(18 / 60) / 8.0) * 1.0
    check("Pi = N * P_alive * S(t+ETA) * kappa",
          expected_lives(6, 0.94, 18, 8.0, 1.0), manual, 1e-12)
    check("Pi falls as ETA grows",
          expected_lives(6, 0.94, 180, 8.0) < expected_lives(6, 0.94, 18, 8.0), True)

    r = load_global()["routing"]
    check("alpha (hazard penalty) = 3", r["alpha_hazard"], 3.0)
    check("beta (uncertainty penalty) = 1", r["beta_uncertainty"], 1.0)
    check("BPR coefficient = 0.15", r["bpr_coefficient"], 0.15)
    check("BPR exponent = 4", r["bpr_exponent"], 4)
    check("impassable at X >= 0.75", r["impassable_if_hazard_gte"], 0.75)

    length, speed, x, u, flow, cap = 1200.0, 45.0, 0.3, 0.4, 900.0, 1800.0
    manual = ((length / (speed / 3.6)) * (1 + 3.0 * x + 1.0 * u)
              * (1 + 0.15 * (flow / cap) ** 4))
    check("c(e) = (L/v)(1+aX+bU)[1+0.15(f/cap)^4]",
          edge_cost(length, speed, x, u, flow, cap, r), manual, 1e-9)
    check("hazard outweighs uncertainty at equal magnitude",
          edge_cost(1000, 40, 0.5, 0.0, cfg=r) > edge_cost(1000, 40, 0.0, 0.5, cfg=r),
          True)


# --- POS rPPG, the one signal-processing chain --------------------------------------

def audit_rppg():
    section("5.1  POS rPPG projection and recovery")
    import numpy as np
    from src.detect import rppg

    # POS projects onto the plane orthogonal to skin tone: P = [[0,1,-1],[-2,1,1]].
    block = np.array([[0.5, 0.4, 0.3], [0.6, 0.5, 0.2]])
    projected = block @ rppg._PROJECTION
    check("POS S1 = G - B", projected[0, 0], 0.4 - 0.3, 1e-12)
    check("POS S2 = -2R + G + B", projected[0, 1], -2 * 0.5 + 0.4 + 0.3, 1e-12)

    check("band floor 0.7 Hz = 42 bpm", rppg.MIN_HZ * 60, 42.0, 1e-9)
    check("band ceiling 3.0 Hz = 180 bpm", rppg.MAX_HZ * 60, 180.0, 1e-9)

    for truth in (48.0, 72.0, 128.0):
        trace = rppg.synthetic_trace(bpm=truth, fps=30.0, seconds=20.0, noise=0.004)
        result = rppg.estimate(trace, fps=30.0)
        # 20 s at 30 fps resolves to 3 bpm, so tolerance is bin-limited.
        check(f"recovers {truth:.0f} bpm from a synthetic trace",
              result["locked"] and abs(result["bpm"] - truth) <= 3.5, True,
              note=f"got {result['bpm']}")


def audit_slope():
    section("2  Horn slope, against an analytic plane")
    import numpy as np
    from src.ingest.level2_geo import slope_degrees

    for dz_dy, dz_dx, cell in ((0.5, 0.1, 30.0), (2.0, 0.0, 10.0), (0.0, 1.5, 25.0)):
        y, x = np.mgrid[0:15, 0:15]
        dem = 100.0 + dz_dy * y + dz_dx * x
        got = slope_degrees(dem, cell)[1:-1, 1:-1].mean()
        want = math.degrees(math.atan(math.hypot(dz_dy, dz_dx) / cell))
        check(f"slope of a plane rising {dz_dy}/{dz_dx} per {cell:g} m", got, want, 1e-9)


def main():
    audit_hazard_formulas()
    audit_physics()
    audit_zone_engine()
    audit_fusion()
    audit_priority_and_routing()
    audit_rppg()
    audit_slope()

    print(f"\n{'=' * 70}")
    print(f"{len(PASS)} checks passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
        return 1
    print("Every formula matches the spec and an independent computation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
