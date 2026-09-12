"""Level 2: geospatial and terrain ingest (spec section 1.1).

Turns a DEM into the terrain variables the hazard formulas need — slope for landslides,
height above the flood stage for inundation and for green-zone suitability — and turns
Sentinel-1 VV backscatter into a water mask.

rasterio is imported lazily: the array functions below are plain numpy and are tested
without it, so a laptop with no geospatial stack can still run and check the maths.

    python -m src.ingest.level2_geo --demo
"""
import argparse
import math

import numpy as np

WATER_VV_DB = -18.0       # Sentinel-1 VV below this reads as open water (spec section 3.2)
STEEP_SLOPE_DEG = 30.0    # landslide risk rises sharply past this


def read_raster(path, band=1):
    """(array, transform, crs). Needs rasterio; raises with the install line if absent."""
    import pathlib
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"raster not found: {p}")
    try:
        import rasterio
    except ImportError:
        raise ImportError("reading rasters needs rasterio: pip install rasterio")
    with rasterio.open(str(p)) as src:
        return src.read(band).astype(float), src.transform, src.crs


def slope_degrees(dem, cell_size_m):
    """Horn's method: slope from a 3x3 neighbourhood, in degrees.

    The standard finite-difference gradient. Edges are handled by numpy's edge padding,
    which flattens the border by one cell rather than producing NaNs that then have to
    be special-cased downstream.
    """
    dem = np.asarray(dem, dtype=float)
    if dem.ndim != 2:
        raise ValueError(f"DEM must be 2-D, got shape {dem.shape}")
    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be > 0")

    padded = np.pad(dem, 1, mode="edge")
    # Horn kernels over the eight neighbours.
    dz_dx = ((padded[:-2, 2:] + 2 * padded[1:-1, 2:] + padded[2:, 2:])
             - (padded[:-2, :-2] + 2 * padded[1:-1, :-2] + padded[2:, :-2])) / (8 * cell_size_m)
    dz_dy = ((padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:])
             - (padded[:-2, :-2] + 2 * padded[:-2, 1:-1] + padded[:-2, 2:])) / (8 * cell_size_m)
    return np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))


def height_above_stage(dem, stage_m):
    """Metres above the flood stage. Negative means inundated."""
    return np.asarray(dem, dtype=float) - stage_m


def inundation_depth(dem, stage_m):
    """Water depth per cell, floored at zero (spec section 3.2)."""
    return np.maximum(0.0, stage_m - np.asarray(dem, dtype=float))


def normalize_elevation(dem):
    """DEM -> [0,1] for the green-suitability elevation term.

    Min-max over the scene, so it answers "high ground relative to here", which is what
    matters when choosing a shelter, not absolute altitude.
    """
    dem = np.asarray(dem, dtype=float)
    finite = dem[np.isfinite(dem)]
    if finite.size == 0:
        raise ValueError("DEM has no finite values")
    lo, hi = float(finite.min()), float(finite.max())
    if hi - lo < 1e-9:
        return np.full_like(dem, 0.5)     # flat scene: no cell is higher than another
    return np.clip((dem - lo) / (hi - lo), 0.0, 1.0)


def water_mask(vv_db, threshold=WATER_VV_DB):
    """Sentinel-1 VV backscatter -> boolean water.

    SAR is the reason this works during a monsoon: it sees through the cloud that blinds
    optical imagery for the whole event.
    """
    return np.asarray(vv_db, dtype=float) <= threshold


def flooded_fraction(vv_db, threshold=WATER_VV_DB):
    """Share of a cell reading as water, for the coverage term in U."""
    mask = water_mask(vv_db, threshold)
    return float(mask.sum() / mask.size) if mask.size else 0.0


def block_reduce(array, factor, how="mean"):
    """Downsample by an integer factor, for regridding a fine raster onto the grid.

    Trailing rows and columns that do not fill a whole block are dropped rather than
    partially averaged, which would quietly bias the edge cells.
    """
    a = np.asarray(array, dtype=float)
    if factor < 1:
        raise ValueError("factor must be >= 1")
    if factor == 1:
        return a
    h, w = (a.shape[0] // factor) * factor, (a.shape[1] // factor) * factor
    if h == 0 or w == 0:
        raise ValueError(f"factor {factor} is larger than the array {a.shape}")
    blocks = a[:h, :w].reshape(h // factor, factor, w // factor, factor)
    if how == "mean":
        return blocks.mean(axis=(1, 3))
    if how == "max":
        return blocks.max(axis=(1, 3))
    if how == "min":
        return blocks.min(axis=(1, 3))
    raise ValueError(f"unknown reduction {how!r}")


def _demo():
    """A synthetic hillside so the maths can be checked without a real DEM."""
    y, x = np.mgrid[0:20, 0:20]
    dem = 100.0 + 0.5 * y + 0.1 * x          # a plane tilted mostly north-south
    slope = slope_degrees(dem, cell_size_m=30.0)
    expected = math.degrees(math.atan(math.hypot(0.5, 0.1) / 30))
    # Interior only: edge padding flattens the border by one cell by design, so the
    # full-array mean sits below the true plane slope.
    print(f"synthetic plane, rise 0.5 m north and 0.1 m east per 30 m cell")
    print(f"  slope        {slope[1:-1, 1:-1].mean():.4f} deg interior "
          f"(exact {expected:.4f})")
    print(f"  normalized z {normalize_elevation(dem).min():.2f} to "
          f"{normalize_elevation(dem).max():.2f}")

    stage = 105.0
    depth = inundation_depth(dem, stage)
    print(f"  stage {stage} m floods {(depth > 0).sum()}/{depth.size} cells, "
          f"max depth {depth.max():.2f} m")

    vv = np.full((10, 10), -8.0)
    vv[:4, :] = -22.0
    print(f"  SAR water fraction {flooded_fraction(vv):.2f} (expected 0.40)")


def _cli():
    ap = argparse.ArgumentParser(description="Level 2 terrain ingest.")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--dem", help="GeoTIFF elevation raster")
    ap.add_argument("--cell-size", type=float, default=30.0)
    ap.add_argument("--stage", type=float, help="flood stage in metres")
    args = ap.parse_args()

    if args.demo:
        _demo()
        return
    if not args.dem:
        ap.error("pass --demo or --dem")

    dem, _transform, crs = read_raster(args.dem)
    slope = slope_degrees(dem, args.cell_size)
    print(f"DEM {dem.shape} crs={crs}")
    print(f"  elevation {np.nanmin(dem):.1f} to {np.nanmax(dem):.1f} m")
    print(f"  slope     mean {np.nanmean(slope):.2f} deg, "
          f"{(slope > STEEP_SLOPE_DEG).sum()} cells over {STEEP_SLOPE_DEG} deg")
    if args.stage is not None:
        depth = inundation_depth(dem, args.stage)
        print(f"  stage {args.stage} m floods {(depth > 0).sum()}/{depth.size} cells")


if __name__ == "__main__":
    _cli()
