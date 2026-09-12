"""IMD 0.25-degree gridded daily rainfall reader (real data, not a fixture).

The IMD .grd files are headerless little-endian float32: 129 latitudes from 6.5N to
38.5N and 135 longitudes from 66.5E to 100.0E, longitude varying fastest, one record per
day. Sea and missing cells carry -999.0.

There is no header, so nothing in the file tells you if you have the shape backwards.
The orientation is verified against geography instead: valid cells must peak across the
widest part of the subcontinent and vanish south of Kanyakumari.

    python -m src.ingest.imd_grid --grd data/raw/imd/rain_ind0.25_26_09_12.grd
    python -m src.ingest.imd_grid --grd <file> --cells -o data/cache/cell_inputs.imd.json
"""
import argparse
import datetime as dt
import json
import pathlib

import numpy as np

from src.grid import cell_id
from src.hazard.formulas import REPO_ROOT

NLAT, NLON = 129, 135
LAT0, LON0, STEP = 6.5, 66.5, 0.25
MISSING = -999.0
HEAVY_RAIN_MM = 64.5


def read_grd(path, day=0):
    """One day of the grid as a (129, 135) masked-by-NaN array in mm."""
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"IMD grid not found: {p}")

    raw = np.fromfile(p, dtype="<f4")
    per_day = NLAT * NLON
    if raw.size % per_day:
        raise ValueError(
            f"{p}: {raw.size} values is not a multiple of {per_day} "
            f"({NLAT} lat x {NLON} lon). Wrong product or a truncated download.")

    days = raw.size // per_day
    if not 0 <= day < days:
        raise IndexError(f"{p} holds {days} day(s); asked for index {day}")

    grid = raw[day * per_day:(day + 1) * per_day].reshape(NLAT, NLON).astype(float)
    grid[grid == MISSING] = np.nan
    if np.isnan(grid).all():
        raise ValueError(f"{p}: every cell is missing on day {day}")
    return grid


def axes():
    return (LAT0 + STEP * np.arange(NLAT), LON0 + STEP * np.arange(NLON))


def verify_orientation(grid):
    """Confirm the array is (lat, lon) and not transposed.

    A transposed read still reshapes cleanly and still produces plausible rainfall
    numbers, so it fails silently and puts every cell in the wrong place. India's land
    mask is asymmetric enough to catch it: the southern rows are ocean and the middle of
    the country is far wider than the far north.
    """
    valid = ~np.isnan(grid)
    lats, _ = axes()

    south = valid[lats < 8.0].sum()
    middle = valid[(lats >= 20.0) & (lats <= 28.0)].sum()
    far_north = valid[lats > 35.0].sum()

    if south > middle * 0.05:
        raise ValueError(f"orientation check failed: {south} land cells below 8N, where "
                         f"there is only ocean. The array is probably transposed.")
    if middle <= far_north:
        raise ValueError(f"orientation check failed: {middle} land cells across 20-28N "
                         f"versus {far_north} above 35N. India is wider in the middle.")
    return True


def to_records(grid, min_mm=0.0):
    """Grid -> [{cell_id, lat, lon, r24_mm}] for every land cell above min_mm."""
    lats, lons = axes()
    ys, xs = np.where(~np.isnan(grid) & (grid >= min_mm))
    return [{
        "cell_id": cell_id(float(lats[y]), float(lons[x])),
        "lat": round(float(lats[y]), 4),
        "lon": round(float(lons[x]), 4),
        "r24_mm": round(float(grid[y, x]), 2),
    } for y, x in zip(ys, xs)]


def summary(grid):
    valid = grid[~np.isnan(grid)]
    lats, lons = axes()
    y, x = np.unravel_index(np.nanargmax(grid), grid.shape)
    return {
        "land_cells": int(valid.size),
        "mean_mm": round(float(valid.mean()), 2),
        "max_mm": round(float(valid.max()), 2),
        "peak_at": [round(float(lats[y]), 2), round(float(lons[x]), 2)],
        "heavy_rain_cells": int((valid > HEAVY_RAIN_MM).sum()),
        "over_100mm": int((valid > 100).sum()),
    }


def to_cell_inputs(grid, top_n=40, q_max=900.0, observed_at=None, sensed_fraction=0.97):
    """Real rainfall -> cell inputs for the flood config.

    Only rainfall is real here. Discharge is derived from rainfall through a flat rating
    coefficient because we have no CWC gauge export, and terrain comes from a latitude
    proxy rather than a DEM. Both are labelled in each record so nobody mistakes them for
    measurements: the honest claim is "real IMD rainfall, modelled everything else".
    """
    observed_at = observed_at or dt.datetime.now(dt.timezone.utc)
    records = sorted(to_records(grid), key=lambda r: -r["r24_mm"])[:top_n]
    if not records:
        raise ValueError("no land cells with rainfall in this grid")

    age_h = 1.5                       # IMD publishes the daily grid on a next-morning cycle
    cells = []
    for r in records:
        rain_discharge = r["r24_mm"] * 2.4
        # Northern cells sit higher; a stand-in until a real DEM lands.
        elevation_norm = round(min(1.0, max(0.0, (r["lat"] - 8.0) / 30.0)), 3)
        x_now = min(1.0, 0.5 * (r["r24_mm"] / 300.0) + 0.5 * (rain_discharge / q_max))

        cells.append({
            "cell_id": r["cell_id"],
            "centroid": [r["lat"], r["lon"]],
            "hazard_inputs": {"r24_mm": r["r24_mm"], "q_cumecs": rain_discharge,
                              "q_max": q_max},
            "sensors": {
                "sigma_ens": 0.0,     # one source, so no ensemble spread to measure
                "data_age_hours": age_h,
                "area_sensed_fraction": sensed_fraction,
            },
            "green_inputs": {
                "forecast_max_x": round(min(1.0, x_now * 1.1), 3),
                "elevation_norm": elevation_norm,
                "shelter_capacity": 1200, "pop_inflow": 2000,
                "access": 0.8, "travel_distance_norm": 0.3,
            },
            "pop": 0,                 # WorldPop not joined yet; exposure stays zero
            "assets": [],
            "reachable": True,
            "components": {
                "r24_mm": r["r24_mm"],
                "source": "IMD 0.25 deg gridded daily rainfall",
                "measured": ["r24_mm"],
                "modelled": ["q_cumecs (rainfall x 2.4)", "elevation_norm (latitude proxy)",
                             "pop (not joined)"],
            },
        })
    return cells


def _cli():
    ap = argparse.ArgumentParser(description="Read an IMD 0.25 degree rainfall grid.")
    ap.add_argument("--grd", required=True)
    ap.add_argument("--day", type=int, default=0)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--cells", action="store_true", help="emit zone-engine cell inputs")
    ap.add_argument("-o", "--out")
    args = ap.parse_args()

    grid = read_grd(args.grd, args.day)
    verify_orientation(grid)
    s = summary(grid)

    if not args.cells:
        print(f"{s['land_cells']} land cells, mean {s['mean_mm']} mm, "
              f"max {s['max_mm']} mm at {s['peak_at']}")
        print(f"{s['heavy_rain_cells']} cells over IMD's {HEAVY_RAIN_MM} mm heavy-rain "
              f"threshold, {s['over_100mm']} over 100 mm\n")
        print(f"{'cell_id':<20} {'lat':>7} {'lon':>7} {'mm':>8}")
        for r in sorted(to_records(grid), key=lambda r: -r["r24_mm"])[:args.top]:
            print(f"{r['cell_id']:<20} {r['lat']:>7.2f} {r['lon']:>7.2f} "
                  f"{r['r24_mm']:>8.1f}")
        return

    cells = to_cell_inputs(grid, top_n=args.top)
    text = json.dumps(cells, indent=2)
    if args.out:
        path = pathlib.Path(args.out)
        if not path.is_absolute():
            path = REPO_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
        print(f"{path.relative_to(REPO_ROOT)}: {len(cells)} cells from real IMD rainfall "
              f"(max {s['max_mm']} mm)")
    else:
        print(text)


if __name__ == "__main__":
    _cli()
