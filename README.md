# Disaster Response Intelligence Pipeline

Hazard scoring, uncertainty-aware zoning, multi-sensor survivor detection, and
capacity-aware evacuation routing, in one dashboard that runs with the network unplugged.

The dashboard ships as **The Sentinel Grid**. Build spec:
[`DISASTER_PIPELINE_SPEC.md`](DISASTER_PIPELINE_SPEC.md), which governs every design
decision below.

## Quick start

```bash
git clone https://github.com/advaithsarva/ibm-hackathon
cd ibm-hackathon
pip install -r requirements.txt
python run_demo.py
```

The API comes up on `http://localhost:8000` and the browser opens the dashboard. Nothing
touches the network after the install.

## The idea

Every cell on a 100 m grid carries three numbers: hazard `X`, epistemic uncertainty `U`,
and green-zone suitability `G`. Those produce three zones.

| Zone | Meaning | Action |
|---|---|---|
| RED | known danger | dispatch rescue |
| BLUE | we don't know enough to call it safe | send a drone here next |
| GREEN | safe, reachable, has shelter capacity | evacuate here |

```
RED    if  X >= 0.75  OR  (X >= 0.40 AND U >= 0.50)
GREEN  if  X < 0.40   AND U < 0.35  AND G >= 0.60  AND reachable
BLUE   otherwise
```

BLUE is what the rest of the system is built around. Most risk maps put uncertainty in a
confidence interval nobody acts on — here it's a zone class with a queue position. A cell
turns blue when the SAR revisit is 11 hours stale, or when the sensors only covered part
of it, and that cell goes to the top of the recon list.

`U` is a noisy-OR over those three failure modes: model disagreement, data staleness, and
sensor coverage gaps. Any one of them is enough to raise it; they compound rather than
average. Cells change class only after crossing a threshold by 0.05 and holding for two
cycles, so a cell sitting on a boundary does not flicker.

## Pipeline

```
  ingest ─── level 1  meteorological   rainfall · gauges · CAPE · SST
        ├─── level 2  geospatial       DEM · slope · Sentinel-1 SAR · NDVI
        └─── level 3  infrastructure   OSM roads/hospitals · WorldPop density
                │
                ▼   common grid · common CRS · common timestamp
         hazard engine ──▶  X ∈ [0,1]        physics: Manning · Holland · GMPE · FS · VCI
                │
         zone engine  ──▶  U, G  →  RED / BLUE / GREEN
                │
         detection    ──▶  P_alive           RGB · thermal · rPPG · acoustic
                │                            fused by Bayesian log-odds
         priority     ──▶  Π = N · P_alive · S(t + ETA) · κ
                │
         routing      ──▶  risk-weighted Dijkstra + min-cost flow
                │
              dashboard
```

Two decisions in there are worth knowing about before you read the code.

`ETA` sits inside the survivability decay term of `Π`, not outside it. A cluster of six
people 18 minutes away outranks a larger cluster 90 minutes away, because the ranking
accounts for who is still alive when the team arrives.

Evacuation is min-cost flow rather than shortest path. Shortest path sends everyone to the
nearest shelter and overflows it; the flow formulation holds each shelter to its capacity
and spills the remainder to the next one.

## API

Four read-only contracts, frozen at minute 15, plus two the dashboard adds. Every response
is served from `data/mock/` until engine output replaces it, which is what let the
dashboard get built before any engine existed.

| Endpoint | Returns |
|---|---|
| `GET /api/zones` | grid cells with `X`, `U`, `G`, `zone`, population, assets |
| `GET /api/detections` | per-cell sensor hits, fused `p_alive`, `n_est` |
| `GET /api/priority` | ranked dispatch plan with `pi`, `eta_min`, team, route |
| `GET /api/evacuation` | shelter flow assignment, utilization, overflow |
| `GET /api/alert/summary` | zone counts, at-risk population, alert level |
| `GET /api/radar/heartbeat` | Doppler bio-signal waveform and peaks |

Every response joins on `cell_id`, the `"{lat:.4f}_{lon:.4f}"` centroid string.

`/api/radar/heartbeat` returns a **simulated** waveform. FINDER-class radar is the
ground-truth sensor for through-rubble vitals and it is hardware we do not have, so the
endpoint models what the panel would show rather than claiming a reading. The rPPG pulse
in `/api/detections` is real signal processing on a real camera feed.

## Layout

```
configs/
  global.yaml                grid, thresholds, zone rule, uncertainty, fusion table, routing
  disasters/*.yaml           seven hazards, one shared engine; only the formula for X changes
src/
  grid.py                    cell_id conventions
  hazard/
    formulas.py              config-driven X evaluator
    physics.py               Manning, Holland, GMPE, factor of safety, VCI, flash rate
    zones.py                 U, G, RED/BLUE/GREEN, hysteresis, exposure score
    test_hazard.py           10 checks
  detect/
    rppg.py                  POS pulse extraction, pure numpy, CPU
    thermal.py               adaptive-percentile hotspot blobs
    rgb_yolo.py              YOLO person detection
    audio_yamnet.py          YAMNet acoustic distress
    fusion.py                RGB-T late NMS merge, Bayesian log-odds
    loader.py                sequential GPU model guard
    test_detect.py           14 checks
backend/main.py              FastAPI, serves the contracts and the frontend
frontend/                    Sentinel Grid dashboard: map, zones, priority, detections
data/
  mock/                      the frozen contracts, plus the engine's cell inputs
  raw/, cache/               pre-downloaded inputs and weights, gitignored
run_demo.py                  starts the server and opens the dashboard
```

## Running the modules

Every module runs standalone and fails loudly on missing data rather than substituting
zeros. A zero hazard score and an unmeasured cell are opposite claims.

```bash
# hazard score for one cell
python -m src.hazard.formulas --config configs/disasters/earthquake.yaml --pga_ms2 3.4

# zone engine over a grid, writes the /api/zones contract
python -m src.hazard.zones --config configs/disasters/earthquake.yaml \
    --cells data/mock/cell_inputs.earthquake.json -o data/mock/zones.json

python -m src.hazard.physics            # governing physics, worked examples
python -m src.detect.rppg --demo        # synthetic pulse, no camera needed
python -m src.detect.rppg --webcam      # live, needs opencv-python
python -m src.detect.fusion --demo      # the p_alive ladder
```

`data/mock/zones.json` is generated by the zone engine from
`cell_inputs.earthquake.json` rather than hand-written, so the fixture cannot drift from
the code. Regenerate it with the command above and diff.

## Configuration

Adding a disaster is a YAML file, not a code change. Each config carries the hazard
formula from the spec, the fields its physics needs, and its time constants:

```yaml
name: earthquake
hazard:
  formula: "min(1.0, pga_ms2 / 4.0)"
  inputs: [pga_ms2]
  tau_survivability_hours: 40
uncertainty:
  tau_data_hours: 1.0
```

Shared weights and thresholds live in `configs/global.yaml`: the zone rule, the noisy-OR
parameters, the green-suitability weights, the log-odds likelihood ratios, and the routing
penalties. Two blocks there are calibration knobs rather than findings, and both say so in
place. The GMPE attenuation coefficients need refitting per region before any absolute PGA
is trustworthy, and the §4.4 exposure weights have no values in the spec, so they sit at
equal thirds.

## Testing

```bash
python -m src.hazard.test_hazard     # 10 checks
python -m src.detect.test_detect     # 14 checks
python data/mock/check_mocks.py      # fixtures obey the contracts
```

No framework, no fixtures. The mock checker re-derives every zone from the rule, verifies
`U` against its own noisy-OR components, confirms each `cell_id` matches its centroid,
resolves every cross-file cell reference, and sums the flows into each shelter to catch
capacity violations.

## Constraints

- 4 GB VRAM on a single machine. Models load sequentially, never concurrently, and stay
  under 2.5 GB resident. `src/detect/loader.py` raises rather than letting a second model
  load.
- Train nothing at build time. Everything is inference-only or classical. Where weights
  don't exist the physics formula runs directly, labelled as a physics-based baseline.
- Assume the network fails at demo time. All data is pre-downloaded to `data/raw/` and
  nothing is fetched at runtime.
- Earthquake is the shipping vertical slice. The other six disasters are config over the
  same engine.

Model weights are not in the repository. Put them under `data/raw/weights/`; the YOLO and
YAMNet wrappers raise a named error when a file is missing, so a model that never ran is
never mistaken for a cell with nobody in it.

## Status

- [x] `data/mock/*.json`, the four contracts
- [x] hazard `X`, config-driven over all seven disasters
- [x] zone engine: noisy-OR `U`, green suitability `G`, hysteresis, exposure score
- [x] detection stack: POS rPPG, thermal blobs, RGB-T merge, log-odds fusion
- [x] FastAPI backend and the Sentinel Grid dashboard
- [ ] YOLO and YAMNet weights, training on the team's own machines
- [ ] ingest: real USGS, SRTM, WorldPop and OSM layers replacing the fixtures
- [ ] priority ranker and risk-aware routing against live graph data
- [ ] report generator
