# Disaster Response Intelligence Pipeline

Hazard scoring, uncertainty-aware zoning, multi-sensor survivor detection, and
capacity-aware evacuation routing, in one dashboard that runs with the network unplugged.

Build spec: [`DISASTER_PIPELINE_SPEC.md`](DISASTER_PIPELINE_SPEC.md). Read §0 and §2 first.

## The idea

Every cell on a 100 m grid carries three numbers: hazard `X`, epistemic uncertainty `U`, and
green-zone suitability `G`. Those produce three zones.

| Zone | Meaning | Action |
|---|---|---|
| RED | known danger | dispatch rescue |
| BLUE | we don't know enough to call it safe | send a drone here next |
| GREEN | safe, reachable, has shelter capacity | evacuate here |

```
RED    if  X ≥ 0.75  OR  (X ≥ 0.40 AND U ≥ 0.50)
GREEN  if  X < 0.40  AND  U < 0.35  AND  G ≥ 0.60  AND reachable
BLUE   otherwise
```

BLUE is what the rest of the system is built around. Most risk maps put uncertainty in a
confidence interval nobody acts on — here it's a zone class with a queue position. A cell
turns blue when the SAR revisit is 11 hours stale, or when the sensors only covered part of
it, and that cell goes to the top of the recon list.

`U` is a noisy-OR over those three failure modes: model disagreement, data staleness, and
sensor coverage gaps.

## Pipeline

```
 LEVEL 1  ATMOSPHERIC & METEOROLOGICAL    IMD rainfall · CWC gauges · GloFAS · CAPE · SST
 LEVEL 2  GEOSPATIAL & TERRAIN            Bhuvan DEM/slope · Sentinel-1/2 · SRTM · NDVI
 LEVEL 3  INFRASTRUCTURE & SOCIO-ECONOMIC OSM roads/bridges/hospitals · WorldPop · HRSL
                          │
                          ▼
        [CLEANING & SYNCHRONIZATION]          common grid · common CRS · common timestamp
                          │
                          ▼
        [PREPROCESSING & FEATURE ENGINEERING]
                          │
                          ▼
        [ML INFERENCE ENGINE]                 physics: Manning · Holland · GMPE · FS · VCI
                          │
                          ▼
        [NORMALIZED THRESHOLD MAPPING]        X ∈ [0,1]
                          │
                          ▼
        [ZONE ENGINE]                         U, G  →  RED / BLUE / GREEN
                          │
                          ▼
        [RISK & IMPACT MAP GENERATION]
          ├── Affected Areas
          ├── Critical Infrastructure Exposure
          └── Population Risk & Rescue Priority Score
                          │
                          ▼
        [DETECTION & VITALS FUSION]           RGB · thermal · rPPG · acoustic
                          │                   → P_alive by Bayesian log-odds
                          ▼
        [EVACUATION PRIORITY RANKER]          Π = N · P_alive · S(t + ETA) · κ
                          │
                          ▼
        [ROUTE OPTIMIZER]                     risk-weighted Dijkstra + min-cost flow
                          │
                          ▼
        [RESCUE PLAN GENERATOR & DASHBOARD]
          ├── Interactive GIS Map
          ├── Alert Dashboard
          └── Report Generator
```

| Stage | Status |
|---|---|
| Levels 1–3 ingest, cleaning, synchronization | `src/ingest/`, running on a synthetic scene |
| ML inference engine | physics baselines in `src/hazard/physics.py`; no trained models yet |
| Normalized threshold mapping | `src/hazard/formulas.py` |
| Zone engine | `src/hazard/zones.py` |
| Risk & impact map, rescue priority score | exposure score in `zones.py`; map layers not built |
| Detection & vitals fusion | `src/detect/` |
| Evacuation priority ranker | `src/priority/` |
| Route optimizer | `src/routing/` |
| Report generator | `src/report/generate.py` |
| Alert dashboard | `/api/alert/summary` only; no UI |
| Interactive GIS map | not built |

Two decisions in there are worth knowing about before you read the code.

`ETA` sits inside the survivability decay term of `Π`, not outside it. A cluster of six people
18 minutes away outranks a larger cluster 90 minutes away, because the ranking accounts for who
is still alive when the team arrives.

Evacuation is min-cost flow rather than shortest path. Shortest path sends everyone to the
nearest shelter and overflows it; the flow formulation holds each shelter to its capacity and
spills the remainder to the next one.

## API

Four read-only contracts, frozen at minute 15, plus two the alert panel adds. Responses
are served from `data/mock/` until engine output replaces them.

| Endpoint | Returns |
|---|---|
| `GET /api/zones` | grid cells with `X`, `U`, `G`, `zone`, population, assets |
| `GET /api/detections` | per-cell sensor hits, fused `p_alive`, `n_est` |
| `GET /api/priority` | ranked dispatch plan with `pi`, `eta_min`, team, route |
| `GET /api/evacuation` | shelter flow assignment, utilization, overflow |
| `GET /api/alert/summary` | zone counts, at-risk population, alert level |
| `GET /api/radar/heartbeat` | Doppler bio-signal waveform and peaks |

Every response joins on `cell_id`, the `"{lat:.4f}_{lon:.4f}"` centroid string.

`/api/radar/heartbeat` returns a simulated waveform. FINDER-class radar is the ground-truth
sensor for through-rubble vitals and it is hardware we do not have, so the endpoint models
what the panel would show rather than claiming a reading. The rPPG pulse in
`/api/detections` is real signal processing on a real camera feed.

## Layout

```
configs/
  global.yaml              grid, thresholds, zone rule, uncertainty, fusion table, routing
  disasters/*.yaml         seven hazards, one shared engine; only the formula for X changes
src/
  grid.py                  cell_id conventions
  hazard/
    formulas.py            config-driven X evaluator
    physics.py             Manning · Holland · GMPE · factor of safety · VCI · flash rate
    zones.py               U, G, RED/BLUE/GREEN, hysteresis, exposure score
    test_hazard.py         10 checks
  detect/
    rppg.py                POS pulse extraction, pure numpy, CPU
    thermal.py             adaptive-percentile hotspot blobs
    rgb_yolo.py            YOLO person detection
    audio_yamnet.py        YAMNet acoustic distress
    fusion.py              RGB-T late NMS merge, Bayesian log-odds
    loader.py              sequential GPU model guard
    test_detect.py         14 checks
  ingest/
    sources.py             dataset registry, offline-first fetcher
    level1_meteo.py        IMD rainfall accumulation, gauge levels
    level2_geo.py          slope, inundation depth, SAR water mask
    level3_infra.py        OSM assets, population binning, isolation
    sync.py                regrid, age layers, emit cell inputs
  priority/
    survivability.py       S(t) = exp(-t/tau)
    expected_lives.py      Pi = N * P_alive * S(t+ETA) * kappa
    dispatch.py            greedy knapsack under a team-hour budget
  routing/
    graph.py               risk-weighted edges, Dijkstra, reachability
    evacuation.py          min-cost flow to shelters
  report/generate.py       incident brief, Claude API with a template fallback
  test_pipeline.py         21 checks
backend/main.py            FastAPI, serves the contracts
data/
  mock/                    the frozen contracts, plus the engine's cell inputs
  raw/, cache/             pre-downloaded inputs and weights, gitignored
run_demo.py                starts the API server
```

The dashboard is the remaining gap. Everything else in the spec's §1.3 layout exists.

## Constraints

- Models load sequentially, never concurrently, so the stack runs on a single GPU.
- Train nothing. Everything is inference-only or classical. Where weights don't exist the
  physics formula runs directly, labelled as a physics-based baseline.
- Assume the network fails at demo time. All data is pre-downloaded and `run_demo.py` reads
  only from disk.
- Earthquake is the shipping vertical slice. The other six disasters are config over the same
  engine.

## Run

```bash
pip install -r requirements.txt

# API server on http://localhost:8000
python run_demo.py

# hazard score for one cell
python -m src.hazard.formulas --config configs/disasters/earthquake.yaml --pga_ms2 3.4

# zone engine over a grid, writes the /api/zones contract
python -m src.hazard.zones --config configs/disasters/earthquake.yaml \
    --cells data/mock/cell_inputs.earthquake.json -o data/mock/zones.json

python -m src.hazard.physics         # governing physics, worked examples

# the flood path end to end, which is the PS-1 scenario
python -m src.ingest.sync --demo --config configs/disasters/flood.yaml     -o data/mock/cell_inputs.flood.json
python -m src.hazard.zones --config configs/disasters/flood.yaml     --cells data/mock/cell_inputs.flood.json -o data/cache/zones.flood.json

python -m src.ingest.sources --list    # which datasets are present
python -m src.priority.expected_lives --demo
python -m src.priority.dispatch --demo
python -m src.routing.graph --demo
python -m src.routing.evacuation --demo
python -m src.report.generate --offline
python -m src.detect.rppg --demo     # synthetic pulse, no camera
python -m src.detect.rppg --webcam   # live, needs opencv-python
python -m src.detect.fusion --demo   # the p_alive ladder

python -m src.hazard.test_hazard     # 10 checks, no framework
python -m src.detect.test_detect     # 14 checks, no weights needed
python -m src.test_pipeline          # 21 checks, ingest through report
python data/mock/check_mocks.py      # fixtures obey the contracts
```

There is no browser UI at the moment: `run_demo.py` starts the API and the endpoints
return JSON. `data/mock/zones.json` is generated by the zone engine from
`cell_inputs.earthquake.json` rather than hand-written, so the fixture cannot drift from
the code. Regenerate it with the command above and diff.

Every module runs standalone and fails loudly on missing data rather than
substituting zeros. A zero hazard score and an unmeasured cell are opposite claims.

## Status

The spec is frozen; modules land per §10 of the build plan.

- [x] `data/mock/*.json`, the four contracts
- [x] hazard `X`, config-driven over all seven disasters
- [x] zone engine: noisy-OR `U`, green suitability `G`, hysteresis, exposure score
- [x] detection stack: POS rPPG, thermal blobs, RGB-T merge, log-odds fusion
- [x] FastAPI backend serving all six endpoints
- [x] ingest: rainfall accumulation, terrain, infrastructure, layer ageing
- [x] priority ranker: survivability decay, expected lives, greedy dispatch
- [x] routing: risk-weighted Dijkstra, capacity-aware evacuation flow
- [x] report generator with an offline template
- [ ] dashboard
- [ ] YOLO and YAMNet weights (training on the team's own machines)
- [ ] real IMD, Bhuvan, Sentinel and WorldPop exports replacing the synthetic scene
