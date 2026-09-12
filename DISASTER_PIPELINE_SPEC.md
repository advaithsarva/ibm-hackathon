# Disaster Response Intelligence Pipeline — Build Spec

> **Hand this file to Claude Code as the single source of truth.**
> Read §0 (Constraints) and §2 (JSON Contracts) before writing any code.
> Anything marked **[SHIP]** gets built today. Anything marked **[SPEC]** goes in the
> architecture doc and the pitch, not the repo.

---

## 0. Hard Constraints — read first

| Constraint | Value | Consequence |
|---|---|---|
| Time budget | **4 hours total**, 4 developers | No training. No fine-tuning. Inference-only or classical. |
| GPU | **4 GB VRAM**, single machine | Models load **sequentially**, never concurrently. Budget ≤ 2.5 GB resident. |
| Rest of compute | CPU only | rPPG, audio, routing, scoring all run CPU-side. |
| Network at demo time | **Assume it fails** | All data pre-downloaded to `./data/`. No live API calls on stage. |
| Final deliverable | One dashboard | Map + zones + detections + priority table + routes + report. |

**Three rules that decide whether this ships:**

1. Freeze the JSON contracts in §2 at **minute 15**. Parallel work is impossible without them.
2. The dashboard developer builds against **mock JSON from minute 30**. This is the insurance policy against integration failure.
3. **Scope lock at 2:00.** After that, integration only. No new features, no matter how small.

**Vertical slice:** pick **ONE** disaster to build end-to-end. Recommended: **Earthquake**
(USGS feed is auth-free and instant, xBD damage weights exist, the rubble → vital-signs story
is the strongest). Second choice: **Flood** (Sen1Floods11 ships pre-labeled masks).
All seven disasters are configured in `configs/disasters/*.yaml` — the engine is shared, only
the hazard formula `X` changes.

---

## 1. System Architecture

### 1.1 Three-level ingestion (from source PDF)

```
LEVEL 1: ATMOSPHERIC & METEOROLOGICAL
  IMD daily/hourly rainfall · CWC river gauges · GloFAS forecasts
  IMD satellite/radar · SST · CAPE / atmospheric soundings
        │
LEVEL 2: GEOSPATIAL & TERRAIN
  ISRO/Bhuvan DEM & slopes · Sentinel-1 SAR · Sentinel-2 optical
  SRTM / CartoDEM · MODIS NDVI · soil moisture
        │
LEVEL 3: INFRASTRUCTURE & SOCIO-ECONOMIC
  OpenStreetMap roads/bridges/hospitals · WorldPop · Meta HRSL density
        │
        ▼
  [CLEANING & SYNCHRONIZATION]      ← common grid, common CRS, common timestamp
        ▼
  [PREPROCESSING & FEATURE ENGINEERING]
        ▼
  [ML INFERENCE ENGINE]
        ▼
  [NORMALIZED THRESHOLD MAPPING → X ∈ [0,1]]
        ▼
  [ZONE ENGINE → RED / BLUE / GREEN]          ← §4, the core novelty
        ▼
  [RISK & IMPACT MAP GENERATION]
   ├── Affected Areas
   ├── Critical Infrastructure Exposure
   └── Population Risk & Rescue Priority Score
        ▼
  [DETECTION & VITALS FUSION]                  ← §5
        ▼
  [EVACUATION PRIORITY RANKER]                 ← §6
        ▼
  [ROUTE OPTIMIZER]                            ← §7
        ▼
  [RESCUE PLAN GENERATOR & DASHBOARD]
   ├── Interactive GIS Map
   ├── Alert Dashboard
   └── Report Generator
```

### 1.2 Grid convention

Everything resolves to a common grid. Fix this early or nothing joins.

- **CRS:** EPSG:4326 for storage/display, EPSG:3857 or local UTM for distance math.
- **Cell size:** 100 m for urban earthquake/flood; 1 km for drought/cyclone regional.
- **`cell_id` format:** `"{lat:.4f}_{lon:.4f}"` of the cell centroid. String, stable, joinable.
- All rasters resampled to the grid with `rasterio.warp.reproject` (bilinear for continuous,
  nearest for categorical).

### 1.3 Repo layout

```
disaster-pipeline/
├── configs/
│   ├── global.yaml                # grid size, CRS, thresholds, weights
│   └── disasters/
│       ├── earthquake.yaml        # ← the one we ship
│       ├── flood.yaml
│       ├── cyclone.yaml
│       ├── landslide.yaml
│       ├── drought.yaml
│       ├── wildfire.yaml
│       └── lightning.yaml
├── data/                          # PRE-DOWNLOADED. never fetched live at demo.
│   ├── raw/
│   ├── cache/
│   └── mock/                      # mock JSON for the dashboard dev
├── src/
│   ├── ingest/                    # P1
│   │   ├── level1_meteo.py
│   │   ├── level2_geo.py
│   │   ├── level3_infra.py
│   │   └── sync.py                # regrid + align timestamps
│   ├── hazard/                    # P1
│   │   ├── formulas.py            # per-disaster X (§3)
│   │   ├── physics.py             # Manning, Holland, GMPE, FS, VCI, F_L (§3.2)
│   │   └── zones.py               # U, G, Z (§4)
│   ├── detect/                    # P2
│   │   ├── rgb_yolo.py
│   │   ├── thermal.py
│   │   ├── rppg.py                # POS algorithm, CPU, no weights
│   │   ├── audio_yamnet.py
│   │   └── fusion.py              # Bayesian log-odds (§5.3)
│   ├── priority/                  # P3
│   │   ├── survivability.py       # S_d(t) (§6.1)
│   │   ├── expected_lives.py      # Π_i (§6.2)
│   │   └── dispatch.py            # greedy knapsack (§6.3)
│   ├── routing/                   # P3
│   │   ├── graph.py               # OSMnx load + risk-weighted edges (§7.1)
│   │   └── evacuation.py          # min-cost flow (§7.2)
│   ├── report/
│   │   └── generate.py            # Claude API call, NOT a local LLM
│   └── api/
│       └── server.py              # FastAPI, serves the contracts in §2
├── dashboard/                     # P4 — builds against data/mock/ from minute 30
└── run_demo.py                    # one command, scripted scenario, offline
```

---

## 2. JSON Contracts — FREEZE AT MINUTE 15

These are the only interfaces between modules. Write them into `data/mock/` immediately with
plausible fake values so all four developers can work against them in parallel.

### 2.1 `GET /api/zones` — hazard engine output

```json
{
  "disaster": "earthquake",
  "generated_at": "2026-09-12T08:40:00Z",
  "grid_cells": [
    {
      "cell_id": "12.9716_77.5946",
      "centroid": [12.9716, 77.5946],
      "X": 0.82,
      "U": 0.21,
      "G": 0.10,
      "zone": "RED",
      "zone_reason": "X >= 0.75",
      "pop": 1450,
      "assets": [{"type": "hospital", "name": "City General", "value": 1.0}],
      "reachable": true,
      "components": {"pga_g": 0.35, "vs30": 280, "damage_prob": 0.61}
    }
  ]
}
```

### 2.2 `GET /api/detections` — detection service output

```json
{
  "cell_id": "12.9716_77.5946",
  "source": "drone_feed_03",
  "timestamp": "2026-09-12T08:41:12Z",
  "detections": [
    {"type": "rgb_person", "conf": 0.66, "bbox": [120, 88, 164, 190]},
    {"type": "thermal_hotspot", "conf": 0.71, "temp_c": 34.2},
    {"type": "rppg_pulse", "conf": 0.88, "bpm": 92, "snr_db": 6.1},
    {"type": "acoustic_distress", "conf": 0.54, "class": "Shout"}
  ],
  "p_alive": 0.94,
  "n_est": 6
}
```

### 2.3 `GET /api/priority` — ranked dispatch plan

```json
{
  "ranked": [
    {
      "rank": 1,
      "cell_id": "12.9716_77.5946",
      "pi": 4.71,
      "n_est": 6,
      "p_alive": 0.94,
      "survivability": 0.87,
      "eta_min": 18,
      "team": "NDRF-3",
      "capability_match": 1.0,
      "route": [[12.9716,77.5946],[12.9702,77.5988],[12.9688,77.6021]]
    }
  ],
  "unassigned": ["12.9801_77.6102"],
  "budget_used_team_hours": 11.5,
  "budget_total_team_hours": 14.0
}
```

### 2.4 `GET /api/evacuation` — flow assignment

```json
{
  "flows": [
    {"from_cell": "12.9716_77.5946", "to_shelter": "shelter_07",
     "people": 820, "path": [[12.97,77.59],[12.96,77.60]], "travel_min": 26}
  ],
  "shelters": [
    {"id": "shelter_07", "name": "Govt High School", "capacity": 1200,
     "assigned": 820, "utilization": 0.68, "location": [12.9601,77.6033]}
  ],
  "overflow": []
}
```

---

## 3. Hazard Layer — Normalized Score `X ∈ [0,1]`

### 3.1 Per-disaster normalization (from source PDF, keep verbatim)

All seven share the same tiering: **Low `< 0.40` · Medium `0.40–0.75` · High `≥ 0.75`**

| # | Disaster | Normalized hazard score `X` |
|---|---|---|
| 1 | **Floods** | `X = min(1.0, 0.5·(R₂₄/300) + 0.5·(Q/Q_max))` |
| 2 | **Cyclones** | `X = min(1.0, W/250)` &nbsp;*(W = max sustained wind, km/h)* |
| 3 | **Droughts** | `X = 1.0 − SMI` &nbsp;*(SMI = soil moisture index, normalized)* |
| 4 | **Earthquakes** | `X = min(1.0, PGA/4.0)` &nbsp;*(PGA in m/s²)* |
| 5 | **Forest fires** | `X = min(1.0, FWI/50)` |
| 6 | **Landslides** | `X = 0.6·(R₃d/200) + 0.4·(θ/60)` &nbsp;*(θ = slope °)* |
| 7 | **Lightning** | `X = min(1.0, CAPE/4500)` |

Store each as a YAML block so the engine is disaster-agnostic:

```yaml
# configs/disasters/earthquake.yaml
name: earthquake
hazard:
  formula: "min(1.0, pga_ms2 / 4.0)"
  inputs: [pga_ms2]
  tau_survivability_hours: 40
thresholds: {low: 0.40, high: 0.75}
uncertainty:
  tau_data_hours: 1.0
  sigma_max: 0.25
models:
  primary: gnn_or_catboost
  damage: xbd_baseline
```

### 3.2 Governing physics (compute the inputs to `X`)

**1. Floods — Manning's flow rate**

```
Q = (1/n) · A · R_h^(2/3) · S^(1/2)
Depth(x,y) = H_stage − Z_DEM(x,y)
```
`n` channel roughness · `A` cross-section area (m²) · `R_h = A / wetted perimeter` · `S` friction slope.
Flags whether discharge exceeds bankfull capacity.
*Fields:* `precip_24h_mm` (>64.5 mm/day = heavy rain warning), `precip_5d_cum` (soil saturation),
`river_gauge_m`, `sar_vv_db` (water < −18 dB — works through cloud), `elevation_z`.

**2. Cyclones — Holland parametric wind model**

```
P(r) = P_c + (P_n − P_c)·exp[−(R_m/r)^B]
```
Reconstructs the radial pressure/wind field → storm surge height and structural wind load.
*Fields:* `central_pressure_hpa`, `max_wind_knots`, `rmw_km`, `sst_celsius` (>26.5 °C fuels intensification).

**3. Droughts — Vegetation Condition Index**

```
VCI = 100 × (NDVI − NDVI_min) / (NDVI_max − NDVI_min)
```
`VCI < 35%` = severe vegetation stress; warns of crop failure months ahead.
*Fields:* `nir_band` / `red_band` (Sentinel-2 B8/B4), `soil_moisture_anomaly`, `evapotranspiration`.

**4. Earthquakes — GMPE attenuation**

```
ln(PGA) = c₁ + c₂·M − c₃·ln(√(R² + h²))
```
`PGA > 0.2 g` ⇒ high risk of masonry damage and bridge collapse.
*Fields:* `magnitude_mw`, `focal_depth_km` (≤20 km = higher surface damage), `pga_g`,
`vs30_shear_velocity` (low vs30 = soft soil amplification).

**5. Landslides — infinite slope Factor of Safety**

```
FS = [c' + (γz − γ_w·h_w)·cos²θ·tanφ'] / [γz·sinθ·cosθ]
```
`FS > 1.0` stable · **`FS ≤ 1.0` imminent failure**.
*Fields:* `slope_deg` (>30° elevated risk), `soil_cohesion_kpa`, `precip_3d_cum` (raises pore pressure).

**6. Lightning — flash rate parameterization**

```
F_L = a · CAPE^(1/2) · w_max²
```
*Fields:* `cape_j_kg` (>2500 J/kg = severe convection), `updraft_velocity_ms`, `stroke_current_ka` (10–100+ kA).
Produces 1–3 hour nowcasts.

### 3.3 Model assignment per disaster

| Disaster | Model (from PDF) | 4 GB substitute for today |
|---|---|---|
| Floods | XGBoost + U-Net (SAR water boundary) | XGBoost + SegFormer-B0 |
| Cyclones | PINN / 3D-ResNet | Holland model + GBDT |
| Droughts | TCN / SVR | Statistical VCI + gradient boosting |
| Earthquakes | GNN / CatBoost | GMPE + XGBoost + xBD damage weights |
| Landslides | XGBoost + geotechnical constraints | FS formula + XGBoost |
| Lightning | ConvLSTM / MLP | CAPE threshold + Random Forest |

> Train nothing. Where a `.pkl` doesn't exist, use the physics formula directly and label it
> "physics-based baseline, ML layer specified" in the slides. This is honest and judges respect it.

---

## 4. Zone Engine — RED / BLUE / GREEN  ⭐ core novelty

Three signals per cell `i`: hazard `X_i`, **uncertainty `U_i`**, safe-zone suitability `G_i`.

### 4.1 Epistemic uncertainty — noisy-OR

```
U_i = 1 − (1 − u_model)·(1 − u_stale)·(1 − u_cover)

u_model = min(1, σ_ens / σ_max)        ensemble / cross-source disagreement
u_stale = 1 − exp(−Δt / τ_data)        data age vs disaster-specific τ
u_cover = 1 − (A_sensed / A_cell)      sensor footprint coverage
```

`σ_ens`: XGBoost quantile spread, OR MC-dropout variance over 10 passes, OR — the cheap and
fully legitimate version — **the standard deviation across independent sources** (gauge vs SAR
vs forecast). Five lines of code. Use it.

`τ_data`: 1 h seismic · 6 h rainfall · 12 h SAR revisit.

### 4.2 Zone assignment

```
RED    if  X ≥ 0.75  OR  (X ≥ 0.40 AND U ≥ 0.50)
GREEN  if  X < 0.40  AND  U < 0.35  AND  G ≥ 0.60  AND  reachable
BLUE   otherwise
```

**Hysteresis (required — stops zone flicker):** a cell only changes class after crossing the
threshold by ±0.05 **and** holding for 2 consecutive cycles.

> **Pitch line, say it three times:** *"A cell is blue not because it's mildly dangerous, but
> because we don't know enough to call it safe. BLUE = 'send a drone here next.' We turn
> uncertainty into a dispatchable task queue instead of a footnote."*

### 4.3 Green-zone suitability

```
G_i = w₁·(1 − max_{t≤T} X_i(t))          ← forecast horizon, NOT current X
    + w₂·ẑ_i                              ← normalized elevation above flood stage
    + w₃·min(1, Cap_i / PopInflow_i)      ← shelter capacity vs expected arrivals
    + w₄·Access_i                         ← road connectivity
    − w₅·d̂_road_i                        ← normalized travel distance

w = [0.35, 0.20, 0.20, 0.15, 0.10]
```

The `max over t` matters: a green zone that floods in 6 hours is a death trap.

### 4.4 Exposure / rescue priority score (from PDF — feeds §6)

```
RescuePriority_i = X_i × ( w₁·PopDensity_i/MaxPop
                         + w₂·CriticalAssetValue_i/MaxAsset
                         + w₃·IsolationFactor_i/MaxIsolation )
```

- `PopDensity_i` — WorldPop / Meta HRSL
- `CriticalAssetValue_i` — OSM, weighting hospitals, evacuation centres, bridges higher
- `IsolationFactor_i` — whether road networks are severed, from graph routing

Use this as the exposure term feeding `N_i` in §6.2.

---

## 5. Detection & Vitals Stack

### 5.1 Sensing layers

| Layer | Method | Status | Compute |
|---|---|---|---|
| RGB person detection | YOLO11n/s, aerial-tuned | **[SHIP]** | ~0.6 GB VRAM |
| Thermal hotspot | YOLO on 3-ch thermal + adaptive blob threshold | **[SHIP]** | shares GPU |
| RGB–T fusion | Late NMS merge over aligned pairs | **[SHIP]** | free |
| **rPPG heartbeat** | **POS / CHROM — classical, zero training** | **[SHIP]** | **CPU only** |
| Acoustic distress | YAMNet → scream / shout / crying classes | **[SHIP]** | CPU only |
| Radar vital signs | UWB / mmWave FMCW through rubble | [SPEC] | hardware |
| Wi-Fi CSI respiration | Channel-state-info breathing detection | [SPEC] | commodity router |

**rPPG is the highest-impact-per-minute feature in this entire build.** POS
(Plane-Orthogonal-to-Skin) extracts a pulse waveform from a face ROI in ~40 lines of numpy.
Live-demo it on a teammate's webcam: green box, BPM overlay, "VITALS CONFIRMED."
Zero VRAM, zero training, visually undeniable.
**Caveat it honestly on screen: triage hint, not a medical reading.**

### 5.2 FINDER-class reference systems (slides, not code)

| System | Principle | Typical range |
|---|---|---|
| NASA JPL / DHS **FINDER** | Microwave Doppler detects micro-motion of heartbeat & breathing | ~9 m rubble, ~6 m concrete |
| UWB through-wall radar (Camero Xaver, Novelda X4) | Impulse radar phase shift from chest wall | 10–20 m |
| mmWave FMCW (TI IWR6843, 60 GHz) | Range-Doppler bin phase → HR + RR | 5–8 m, ~$300 board |
| Delsar LD3 seismic-acoustic | Geophone + mic array; tapping/voice through debris | multi-metre |
| Wi-Fi CSI sensing | Respiration from commodity router CSI amplitude | through-wall |
| Meta Disaster Maps / Google Crisis | Aggregated device-density displacement | population-scale |
| DJI M30T / thermal sUAS | Radiometric IR + zoom | aerial sweep |

**Pitch line:** *"FINDER-class radar is the ground-truth sensor. Our contribution is the fusion
layer — radar, thermal, camera and audio combine into one calibrated probability instead of four
separate operator judgments."*

### 5.3 Evidence fusion — Bayesian log-odds *(implement this, ~15 lines)*

```
logit(P_alive_i) = logit(p₀) + Σ_k z_k · log(Λ_k)
P_alive_i = sigmoid(logit)
```

`z_k ∈ {0,1}` = sensor `k` fired. Prior `p₀` = local population density × collapse probability.

| Sensor `k` | `Λ_k` | `log Λ_k` |
|---|---|---|
| Radar vital sign | 25 | 3.22 |
| Acoustic distress call | 12 | 2.48 |
| rPPG pulse locked | 8 | 2.08 |
| RGB person detection | 5 | 1.61 |
| Thermal hotspot (~37 °C) | 3 | 1.10 |
| Phone ping / crowd density | 2 | 0.69 |

Why it wins the Q&A: a thermal blob alone is weak (could be an engine). Thermal **+** rPPG **+**
audio compounds to `P > 0.95`. You have a principled answer to *"how do you avoid false positives?"*

---

## 6. Rescue Prioritization

### 6.1 Survivability decay — the golden hours

```
S_d(t) = exp(−t / τ_d)
```

| Disaster | `τ_d` | Rationale |
|---|---|---|
| Earthquake (entrapment) | 40 h | crush syndrome, dehydration |
| Landslide | 24 h | burial asphyxiation |
| Cyclone | 12 h | exposure |
| Flood | 8 h | hypothermia, exhaustion |
| Forest fire | 2 h | smoke inhalation |

### 6.2 Expected lives saved — the objective function

```
Π_i = N_i · P_alive_i · S_d(t_now + ETA_i) · κ_i
```

`N_i` estimated people in cell (from §4.4 exposure) · `ETA_i` from the risk-aware router (§7) ·
`κ_i ∈ [0,1]` capability match (does the dispatched team carry the right equipment for this hazard?).

> **The killer insight:** `ETA_i` sits *inside* the decay term. So the system automatically
> deprioritizes a large but unreachable cluster in favour of a smaller reachable one. That is a
> real triage decision no naive risk map makes. Demo it explicitly.

### 6.3 Dispatch as a knapsack

```
maximize  Σ_i Π_i · x_i
subject to Σ_i c_i · x_i ≤ B,    x_i ∈ {0,1}
```

`c_i` team-hours required · `B` total available team-hours.
**Greedy on `Π_i / c_i`** gives a `(1 − 1/e)` approximation and runs in milliseconds. Ship the greedy.

---

## 7. Routing & Evacuation

### 7.1 Risk-aware edge cost

```
c(e) = (L_e / v_e) · (1 + α·X_e + β·U_e) · [1 + 0.15·(f_e / cap_e)^4]

c(e) = ∞   if  X_e ≥ 0.75  OR  bridge_status_e == "down"
```

The bracketed term is the **BPR congestion function** — it stops the optimizer routing 50,000
people down one road. `α ≈ 3`, `β ≈ 1` (penalize uncertain roads, but less than known-dangerous ones).

### 7.2 Evacuation as min-cost flow, not shortest path

Shortest path alone sends everyone to the nearest shelter and overflows it.

```
minimize  Σ_ij c_ij · f_ij
subject to  Σ_j f_ij = Pop_i          (everyone leaves)
            Σ_i f_ij ≤ Cap_j          (shelters respect capacity)
            f_ij ≥ 0
```

`networkx.min_cost_flow` on a bipartite RED-cell → GREEN-zone graph. One function call.
**Capacity-aware evacuation is a genuine differentiator and takes ~20 minutes to implement.**

### 7.3 Downstream operational modules (from PDF)

- **Evacuation Priority Ranker** — processes cells in the top tier (`X ≥ 0.75`) to sequence
  regional evacuation queues and dispatch orders for state disaster response forces.
- **Route Optimizer** — safe alternative travel vectors around blocked infrastructure and active
  hazard zones via graph routing (NetworkX Dijkstra / A*, flooded roads = infinite weight).
- **Rescue Plan Generator & Dashboard**
  - *Interactive GIS Map* — real-time spatial polygons, hazard heatmaps, dynamic infrastructure state
  - *Alert Dashboard* — high-priority automated triggers and status panels for field commanders
  - *Report Generator* — executive summaries, resource allocation metrics, logistical blueprints

---

## 8. Model Registry — what fits in 4 GB

| Task | Model | Size / VRAM | Source |
|---|---|---|---|
| Person detection (aerial) | `YOLO11n` / `YOLOv8n` | 6 MB / ~0.6 GB | Ultralytics (`pip install ultralytics`) |
| Aerial SAR persons | YOLO trained on HERIDAL / SARD | same | Hugging Face Hub |
| Thermal persons | YOLO trained on FLIR ADAS / LLVIP | same | Hugging Face Hub |
| Flood water segmentation | `SegFormer-B0` or U-Net (Sen1Floods11) | ~15 M / ~1.3 GB | Hugging Face |
| Building damage | xView2 / xBD baseline weights | ~1.5 GB | xview2.org |
| **Heartbeat (rPPG)** | **POS / CHROM — classical** | **0** | `github.com/ubicomplab/rPPG-Toolbox` |
| Distress audio | YAMNet | 4 MB, CPU | TF Hub / `tensorflow-hub` |
| Report generation | **Claude API** (recommended) | 0 | `api.anthropic.com/v1/messages` |
| Report fallback (offline) | `Qwen2.5-3B-Instruct-Q4_K_M` | 2.3 GB | Hugging Face GGUF |
| Routing | OSMnx + NetworkX | 0, CPU | `pip install osmnx networkx` |

**VRAM budget:** YOLO (0.6) + SegFormer (1.3) = **1.9 GB resident**. Headroom for the rest.
If OOM: export to ONNX INT8, or `torch.cuda.empty_cache()` between stages.

> **Do not run a local LLM as the primary path.** It eats the entire VRAM budget for the feature
> judges care least about. Call the Claude API for the executive-summary generator — one fetch,
> far better output. Keep Qwen-3B-Q4 as the airplane-mode fallback only.

---

## 9. Datasets

> **URLs are from general knowledge and may have moved. Verify each one in the first 20 minutes,
> during the parallel download phase — do not discover a dead link at hour 3.**

### 9.1 Auth-free, instant — use these today

| Dataset | What you get | Link |
|---|---|---|
| **USGS Earthquake GeoJSON** ⭐ | Live global quakes, mag/depth/location. Zero friction. | `earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php` |
| **USGS ShakeMap** | PGA / PGV / MMI grids per event | `earthquake.usgs.gov/data/shakemap/` |
| **OpenStreetMap** via OSMnx/Overpass ⭐ | Roads, bridges, hospitals, schools — one line of Python | `osmnx` package / `overpass-api.de` |
| **NASA FIRMS** | Active fire/thermal anomalies, MODIS+VIIRS (free key, 2 min) | `firms.modaps.eosdis.nasa.gov/api/` |
| **NOAA IBTrACS** | Global cyclone best-track CSV | `ncei.noaa.gov/products/international-best-track-archive` |
| **Sen1Floods11** ⭐ | Sentinel-1 SAR flood chips **with labeled masks** | `github.com/cloudtostreet/Sen1Floods11` |
| **WorldPop** | 100 m population density GeoTIFF | `hub.worldpop.org` |
| **NASA COOLR / Global Landslide Catalog** | Landslide event inventory | `gpm.nasa.gov/landslides/` |
| **SRTM 30 m DEM** | Elevation, slope, aspect | `earthexplorer.usgs.gov` / `opentopography.org` |
| **GHSL (JRC)** | Built-up surface + population grids | `ghsl.jrc.ec.europa.eu` |
| **xView2 / xBD** | Pre/post disaster building damage, labeled | `xview2.org` |

### 9.2 Free but needs registration — skip today, cite in slides

| Dataset | Use | Link |
|---|---|---|
| Copernicus **GloFAS** | River discharge forecasts | `global-flood.emergency.copernicus.eu` |
| Copernicus **EDO** | Drought indicators, SPI, soil moisture | `edo.jrc.ec.europa.eu` |
| **Copernicus Data Space** | Sentinel-1 SAR + Sentinel-2 optical | `dataspace.copernicus.eu` |
| NASA **LP DAAC** | MODIS NDVI, land products | `lpdaac.usgs.gov` |
| **IMD** | India rainfall, satellite, radar | `mausam.imd.gov.in` / `imdpune.gov.in` |
| **CWC / India-WRIS** | River gauge levels | `indiawris.gov.in` |
| **ISRO Bhuvan** | CartoDEM, landslide hazard maps | `bhuvan.nrsc.gov.in` |
| **ISRO MOSDAC** | Cyclone satellite products | `mosdac.gov.in` |
| **IITM Damini** | India lightning network | `damini.iitm.ac.in` |
| **Meta HRSL** | High-res settlement layer | via HDX `data.humdata.org` |

### 9.3 Detection / vision datasets

| Dataset | Use | Link |
|---|---|---|
| **HERIDAL** | Aerial search-and-rescue persons in wilderness | `ipsar.fesb.unist.hr/HERIDAL` |
| **SARD** | Search-and-rescue aerial person detection | Hugging Face / Kaggle |
| **LLVIP** | Aligned visible–infrared pedestrian pairs ⭐ for RGB-T fusion | `github.com/bupt-ai-cz/LLVIP` |
| **FLIR ADAS** | Thermal object detection | `flir.com/oem/adas/adas-dataset-form/` |
| **VisDrone** | Drone-view detection benchmark | `github.com/VisDrone/VisDrone-Dataset` |
| **UBFC-rPPG / PURE** | rPPG validation (pulse ground truth) | search Hugging Face / original labs |
| **AudioSet** (via YAMNet) | Scream / shout / crying classes | `research.google.com/audioset/` |

### 9.4 Download rule

```bash
# hour 0, run in parallel across all 4 machines
mkdir -p data/raw data/cache data/mock
# each dev downloads their own layer; commit nothing large, share via USB/LAN
```

Everything lands in `./data/raw/`. `run_demo.py` reads **only** from disk.
Add a `--live` flag that you will never use on stage.

---

## 10. Build Order — 4 hours, 4 people

| Time | P1 — Hazard / Zones | P2 — Detection | P3 — Routing / Priority | P4 — Dashboard |
|---|---|---|---|---|
| **0:00–0:30** | **All four together:** lock scope to ONE disaster · download data + weights in parallel · **write the §2 JSON contracts into `data/mock/`** · `git init`, one folder per person | | | |
| **0:30–2:00** | Grid + `X` + `U` + `G` + zone rule + hysteresis → `zones.geojson` | YOLO RGB + thermal + **rPPG webcam** + YAMNet + log-odds fusion | OSMnx graph, risk-weighted Dijkstra, min-cost flow, `Π_i`, greedy knapsack | Leaflet/deck.gl map, zone layers, priority table, detection panel — **reading `data/mock/` only** |
| **2:00–3:00** | **INTEGRATE.** Swap mocks for real outputs one module at a time. Scope is frozen. No new features. | | | |
| **3:00–3:30** | Seed the scripted demo scenario. Freeze code. Tag the commit. | | | |
| **3:30–4:00** | Rehearse **3× out loud**. Build 5 slides. | | | |

### Definition of done — the demo runs if all of these are true

- [ ] `python run_demo.py` works with **Wi-Fi switched off**
- [ ] Map renders RED / BLUE / GREEN polygons
- [ ] Clicking a BLUE cell shows *why* it's blue (the `U` breakdown: stale / coverage / model)
- [ ] Live webcam shows an rPPG BPM reading
- [ ] `P_alive` bar visibly climbs as sensors stack up
- [ ] Priority table shows a lower-population cell outranking a higher one **because of ETA**
- [ ] Evacuate button draws flow lines and fills shelters to capacity, with one overflow reroute
- [ ] Report Generator emits a one-page incident brief
- [ ] Config-swap to a second disaster loads without crashing (even if outputs are rough)

---

## 11. Demo Script — 3 minutes

1. **(20 s)** Map loads. Zones paint themselves. *"Zone X, 40 minutes after a magnitude 6.8."*
2. **(30 s)** Click a BLUE cell. *"Blue isn't 'medium danger' — blue is 'we don't know.' Our
   uncertainty model flagged this cell because SAR revisit is 11 hours stale. It's now #1 in the
   recon queue."* ← **the winning moment**
3. **(40 s)** Detection feed, teammate on webcam. Thermal box → rPPG BPM locks → probability bar
   climbs 0.31 → 0.94. *"Four weak sensors, one calibrated answer."*
4. **(40 s)** Priority table. *"Cell 7 has more people but a 90-minute ETA. Cell 3 has fewer but we
   reach it in 18. The model routes to Cell 3 — because survivability decays and ETA sits inside
   the decay term."*
5. **(30 s)** Hit Evacuate. Flow lines to green zones, shelters fill, one overflows and reroutes.
6. **(20 s)** Report Generator → one-page brief. *"Config-swap to flood, cyclone, landslide — same
   engine, different X."*

---

## 12. Instructions to Claude Code

1. Read §0 and §2 first. **Generate `data/mock/*.json` matching §2 before writing any logic.**
2. Build `src/hazard/formulas.py` as a **config-driven evaluator** — the seven formulas in §3.1
   come from YAML, not from seven hardcoded functions.
3. `src/hazard/zones.py` implements §4 exactly, including hysteresis. Do not simplify `U`.
4. `src/detect/rppg.py` — implement **POS**, pure numpy + OpenCV, no model weights, CPU only.
5. `src/detect/fusion.py` — the log-odds table in §5.3 lives in a config dict, not inline constants.
6. `src/routing/` — OSMnx for the graph, NetworkX `min_cost_flow` for evacuation. Not a for-loop.
7. Never load two GPU models at once. Wrap in a context manager that frees VRAM on exit.
8. Every module must run standalone: `python -m src.hazard.zones --config configs/disasters/earthquake.yaml`
9. Fail loud and early on missing data files — never silently fall back to zeros.
10. Prioritize in this order: **working demo path > correctness > completeness > polish.**

---

*Companion architecture notes: normalized hazard tiers, ingestion levels, physics models,
rescue-priority formula, and downstream module definitions are carried forward verbatim from the
original pipeline PDF (§1.1, §3.1, §3.2, §4.4, §7.3).*
