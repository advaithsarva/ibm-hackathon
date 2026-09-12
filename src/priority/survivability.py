"""Survivability decay, the golden hours (spec section 6.1).

    S_d(t) = exp(-t / tau_d)

tau is per disaster because people die of different things at different speeds. The
values come from the spec and live in configs/disasters/*.yaml, not here.

    python -m src.priority.survivability --config configs/disasters/flood.yaml
"""
import argparse
import math

from src.hazard.formulas import load_config

# Spec section 6.1. Drought and lightning have no value in the spec, so they are absent
# rather than guessed; asking for one raises.
TAU_HOURS = {
    "earthquake": 40,     # crush syndrome, dehydration
    "landslide": 24,      # burial asphyxiation
    "cyclone": 12,        # exposure
    "flood": 8,           # hypothermia, exhaustion
    "wildfire": 2,        # smoke inhalation
}


def tau_for(disaster_cfg):
    """Hours. Config wins; the table is the fallback."""
    tau = disaster_cfg.get("hazard", {}).get("tau_survivability_hours")
    if tau is None:
        tau = TAU_HOURS.get(disaster_cfg["name"])
    if tau is None:
        raise ValueError(
            f"no survivability tau for {disaster_cfg['name']!r}. The spec gives none, so "
            f"set tau_survivability_hours in its config rather than assuming one."
        )
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    return float(tau)


def survivability(elapsed_hours, tau_hours):
    """Probability someone trapped at t=0 is still alive at t=elapsed_hours."""
    if elapsed_hours < 0:
        raise ValueError("elapsed hours cannot be negative")
    if tau_hours <= 0:
        raise ValueError("tau must be > 0")
    return math.exp(-elapsed_hours / tau_hours)


def survivability_at_arrival(now_hours, eta_minutes, tau_hours):
    """S(t_now + ETA). The term that makes ETA change the ranking, not just the schedule."""
    return survivability(now_hours + eta_minutes / 60.0, tau_hours)


def half_life(tau_hours):
    """Hours until survivability halves. Easier to brief than tau itself."""
    return tau_hours * math.log(2)


def _cli():
    ap = argparse.ArgumentParser(description="Survivability decay curve.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--hours", type=float, nargs="*",
                    default=[0, 1, 2, 4, 8, 12, 24, 48])
    args = ap.parse_args()

    cfg = load_config(args.config)
    tau = tau_for(cfg)
    print(f"{cfg['name']}: tau = {tau:g} h, half-life {half_life(tau):.1f} h")
    for h in args.hours:
        s = survivability(h, tau)
        bar = "#" * int(s * 40)
        print(f"  t+{h:>5.1f} h  S = {s:.3f}  {bar}")


if __name__ == "__main__":
    _cli()
