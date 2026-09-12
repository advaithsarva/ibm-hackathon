"""Expected lives saved, the objective function (spec section 6.2).

    Pi_i = N_i * P_alive_i * S_d(t_now + ETA_i) * kappa_i

ETA sits inside the decay term. That is the whole design: a large cluster ninety minutes
away is worth less than a small one eighteen minutes away, because the ranking accounts
for who is still alive when the team arrives. Move ETA outside and the system becomes an
ordinary risk map that sorts by headcount.

    python -m src.priority.expected_lives --demo
"""
import argparse

from src.hazard.formulas import load_config
from src.priority.survivability import survivability_at_arrival, tau_for

# kappa: does the dispatched team carry the right equipment for this hazard? Section 6.2
# defines the term but gives no table, so this is a starting point to calibrate against
# the team roster, not a finding.
CAPABILITY = {
    ("flood", "boat"): 1.0,
    ("flood", "swift_water"): 1.0,
    ("flood", "medical"): 0.6,
    ("flood", "foot"): 0.3,
    ("landslide", "heavy_digging"): 1.0,
    ("landslide", "canine"): 0.85,
    ("landslide", "medical"): 0.5,
    ("earthquake", "heavy_digging"): 1.0,
    ("earthquake", "canine"): 0.9,
    ("earthquake", "medical"): 0.6,
}
DEFAULT_CAPABILITY = 0.5


def capability_match(disaster, team_kind):
    """kappa in [0,1]. An unknown pairing gets the neutral default, not a zero."""
    return CAPABILITY.get((disaster, team_kind), DEFAULT_CAPABILITY)


def expected_lives(n_est, p_alive, eta_minutes, tau_hours, kappa=1.0, elapsed_hours=0.0):
    """Pi for one cell. Expected number of people saved by reaching it."""
    if n_est < 0:
        raise ValueError("n_est cannot be negative")
    if not 0.0 <= p_alive <= 1.0:
        raise ValueError(f"p_alive must be in [0,1], got {p_alive}")
    if not 0.0 <= kappa <= 1.0:
        raise ValueError(f"kappa must be in [0,1], got {kappa}")
    if eta_minutes < 0:
        raise ValueError("eta cannot be negative")

    s = survivability_at_arrival(elapsed_hours, eta_minutes, tau_hours)
    return n_est * p_alive * s * kappa


def score_cells(cells, disaster_cfg, elapsed_hours=0.0):
    """Score and rank cells. Each cell needs cell_id, n_est, p_alive, eta_min.

    Optional: team, team_kind. Returns the list sorted by Pi, highest first.
    """
    tau = tau_for(disaster_cfg)
    disaster = disaster_cfg["name"]

    scored = []
    for c in cells:
        for key in ("cell_id", "n_est", "p_alive", "eta_min"):
            if key not in c:
                raise ValueError(f"cell {c.get('cell_id', '?')}: missing {key!r}")
        kappa = c.get("capability_match")
        if kappa is None:
            kappa = capability_match(disaster, c.get("team_kind", "foot"))
        s = survivability_at_arrival(elapsed_hours, c["eta_min"], tau)
        scored.append({
            **c,
            "pi": round(expected_lives(c["n_est"], c["p_alive"], c["eta_min"], tau,
                                       kappa, elapsed_hours), 4),
            "survivability": round(s, 4),
            "capability_match": round(kappa, 3),
        })

    return sorted(scored, key=lambda c: c["pi"], reverse=True)


def _demo():
    """How much ETA moves the ranking depends on how fast survivability decays.

    The same two cells are scored under flood (tau 8 h) and wildfire (tau 2 h). Under
    flood the larger distant cluster still wins: ninety minutes costs little against an
    eight-hour constant, and fourteen people are fourteen people. Under wildfire the
    same ninety minutes flips the order.

    Worth knowing before the pitch. "ETA changes the ranking" is true, but it is not
    unconditional, and a judge who works out the arithmetic will ask.
    """
    cells = [
        {"cell_id": "12.9750_77.5890", "n_est": 14, "p_alive": 0.72, "eta_min": 90,
         "team_kind": "boat"},
        {"cell_id": "12.9716_77.5946", "n_est": 6, "p_alive": 0.94, "eta_min": 18,
         "team_kind": "boat"},
        {"cell_id": "12.9702_77.5988", "n_est": 4, "p_alive": 0.79, "eta_min": 12,
         "team_kind": "swift_water"},
    ]

    for name in ("flood", "wildfire"):
        cfg = load_config(f"configs/disasters/{name}.yaml")
        ranked = score_cells(cells, cfg)
        print(f"\n{name}, tau = {tau_for(cfg):g} h")
        print(f"{'rank':<5} {'cell':<18} {'n':>4} {'p_alive':>8} {'eta':>5} "
              f"{'S':>7} {'Pi':>7}")
        for i, c in enumerate(ranked, 1):
            print(f"{i:<5} {c['cell_id']:<18} {c['n_est']:>4} {c['p_alive']:>8.2f} "
                  f"{c['eta_min']:>5} {c['survivability']:>7.3f} {c['pi']:>7.3f}")
        top = ranked[0]
        print(f"      -> {top['cell_id']} first "
              f"({top['n_est']} people, ETA {top['eta_min']} min)")

    print("\nSame cells, same ETAs, different order. ETA earns its place inside the")
    print("decay term when the hazard kills quickly; under a slow one, headcount wins.")


def _cli():
    ap = argparse.ArgumentParser(description="Expected lives saved.")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        _demo()
    else:
        ap.print_help()


if __name__ == "__main__":
    _cli()
