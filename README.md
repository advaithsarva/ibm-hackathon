# Disaster Response Intelligence Pipeline

Hazard scoring, uncertainty-aware zoning, multi-sensor survivor detection, and
capacity-aware evacuation routing — one offline dashboard.

**Build spec: [`DISASTER_PIPELINE_SPEC.md`](DISASTER_PIPELINE_SPEC.md).** Read §0 and §2 first.

## The idea

Every cell on a 100 m grid gets three numbers: hazard `X`, epistemic uncertainty `U`, and
green-zone suitability `G`. Those produce three zones:

- **RED** — known danger. Dispatch rescue.
- **GREEN** — known safe, reachable, has shelter capacity. Evacuate here.
- **BLUE** — *we don't know enough to call it safe.* Send a drone next.

BLUE is the contribution. Uncertainty becomes a dispatchable recon queue instead of a
footnote on a heatmap.

Detections (RGB, thermal, rPPG pulse, acoustic distress) fuse by Bayesian log-odds into one
calibrated `P_alive` per cell. Rescue priority is expected lives saved,
`Π = N · P_alive · S(t + ETA) · κ` — with ETA inside the survivability decay, so a smaller
reachable cluster correctly outranks a larger unreachable one.

## Status

Scaffolding. Vertical slice is **earthquake**; the other six disasters are YAML config over
the same engine.

## Run

```bash
python run_demo.py    # offline, reads only from data/
```
