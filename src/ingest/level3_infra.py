"""Level 3: infrastructure and socio-economic ingest (spec section 1.1).

Answers PS-1's second question, who and what is likely to be affected: population per
cell from WorldPop, and critical assets from OpenStreetMap weighted by how much their
loss costs during a flood.

OSMnx and rasterio are imported lazily. The weighting and binning functions are plain
Python and are tested without either.

    python -m src.ingest.level3_infra --demo
"""
import argparse
import collections

from src.grid import cell_id

# What a lost asset costs a district mid-monsoon. Hospitals top the list because losing
# one converts survivable injuries into fatalities; bridges follow because a severed
# route isolates every cell behind it (that is the isolation term in section 4.4).
ASSET_WEIGHTS = {
    "hospital": 1.0,
    "clinic": 0.6,
    "bridge": 0.9,
    "shelter": 0.85,
    "school": 0.7,          # doubles as an evacuation centre in Indian district plans
    "police": 0.6,
    "fire_station": 0.7,
    "water_treatment": 0.65,
    "power_substation": 0.75,
    "road_primary": 0.5,
}

# OSM tag -> our asset type. Anything unmatched is ignored rather than counted at a
# default weight, which would inflate exposure with car parks and bus shelters.
OSM_TAGS = {
    ("amenity", "hospital"): "hospital",
    ("amenity", "clinic"): "clinic",
    ("amenity", "doctors"): "clinic",
    ("amenity", "school"): "school",
    ("amenity", "college"): "school",
    ("amenity", "police"): "police",
    ("amenity", "fire_station"): "fire_station",
    ("amenity", "shelter"): "shelter",
    ("man_made", "water_works"): "water_treatment",
    ("power", "substation"): "power_substation",
    ("bridge", "yes"): "bridge",
    ("highway", "primary"): "road_primary",
    ("highway", "trunk"): "road_primary",
}


def asset_value(assets):
    """Total weighted value of the assets in a cell."""
    return round(sum(ASSET_WEIGHTS.get(a["type"], 0.0) for a in assets), 4)


def classify_osm_feature(tags):
    """OSM tag dict -> our asset type, or None when it is not critical infrastructure."""
    for (key, value), asset_type in OSM_TAGS.items():
        if tags.get(key) == value:
            return asset_type
    return None


def bin_to_grid(features, cell_size_deg):
    """Point features -> {cell_id: [assets]}.

    features: iterable of (lat, lon, type, name). Snapping is done on the cell centroid
    so the key matches what the zone engine produces for the same location.
    """
    if cell_size_deg <= 0:
        raise ValueError("cell_size_deg must be > 0")
    binned = collections.defaultdict(list)
    for lat, lon, asset_type, name in features:
        clat = (round(lat / cell_size_deg) * cell_size_deg)
        clon = (round(lon / cell_size_deg) * cell_size_deg)
        binned[cell_id(clat, clon)].append({
            "type": asset_type,
            "name": name,
            "value": ASSET_WEIGHTS.get(asset_type, 0.0),
        })
    return dict(binned)


def fetch_osm_assets(place, cell_size_deg=0.001):
    """Download critical infrastructure for a place name via OSMnx.

    Network call, so it belongs in the download phase and never in the demo path.
    """
    try:
        import osmnx as ox
    except ImportError:
        raise ImportError("OSM ingest needs osmnx: pip install osmnx")

    tags = {"amenity": True, "man_made": True, "power": True, "bridge": True}
    gdf = ox.features_from_place(place, tags)
    features = []
    for _, row in gdf.iterrows():
        asset_type = classify_osm_feature(row.to_dict())
        if asset_type is None:
            continue
        geom = row.geometry
        point = geom.centroid if geom.geom_type != "Point" else geom
        features.append((point.y, point.x, asset_type, row.get("name") or asset_type))
    return bin_to_grid(features, cell_size_deg)


def population_per_cell(pop_raster, factor):
    """WorldPop counts summed into grid cells.

    Population is a count, so blocks are summed, never averaged. Averaging here would
    silently divide a district's population by the block size.
    """
    import numpy as np
    a = np.asarray(pop_raster, dtype=float)
    a = np.nan_to_num(a, nan=0.0)
    if factor < 1:
        raise ValueError("factor must be >= 1")
    if factor == 1:
        return a
    h, w = (a.shape[0] // factor) * factor, (a.shape[1] // factor) * factor
    if h == 0 or w == 0:
        raise ValueError(f"factor {factor} is larger than the raster {a.shape}")
    return a[:h, :w].reshape(h // factor, factor, w // factor, factor).sum(axis=(1, 3))


def isolation_factor(cell, reachable_neighbours, total_neighbours):
    """Share of a cell's road links that are severed (section 4.4).

    1.0 means completely cut off. This is what makes an unreachable cluster rank
    differently from a reachable one of the same size.
    """
    if total_neighbours <= 0:
        return 1.0
    return round(1.0 - (reachable_neighbours / total_neighbours), 4)


def _demo():
    features = [
        (12.9716, 77.5946, "hospital", "City General"),
        (12.9717, 77.5947, "school", "Corporation School 4"),
        (12.9750, 77.5890, "bridge", "Canal Rd Overpass"),
        (12.9601, 77.6033, "shelter", "Govt High School"),
        (12.9601, 77.6034, "clinic", "Ward 12 PHC"),
    ]
    binned = bin_to_grid(features, cell_size_deg=0.001)
    print(f"{len(features)} features -> {len(binned)} cells")
    for cid, assets in sorted(binned.items()):
        kinds = ", ".join(a["type"] for a in assets)
        print(f"  {cid}  value={asset_value(assets):.2f}  {kinds}")

    print(f"\nisolation, 0 of 4 links open: {isolation_factor(None, 0, 4)}")
    print(f"isolation, 3 of 4 links open: {isolation_factor(None, 3, 4)}")


def _cli():
    ap = argparse.ArgumentParser(description="Level 3 infrastructure ingest.")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--place", help='e.g. "Bengaluru, India" (downloads from OSM)')
    ap.add_argument("--cell-size-deg", type=float, default=0.001)
    args = ap.parse_args()

    if args.demo:
        _demo()
        return
    if not args.place:
        ap.error("pass --demo or --place")

    binned = fetch_osm_assets(args.place, args.cell_size_deg)
    total = sum(len(v) for v in binned.values())
    print(f"{total} critical assets across {len(binned)} cells in {args.place}")
    counts = collections.Counter(a["type"] for v in binned.values() for a in v)
    for kind, n in counts.most_common():
        print(f"  {kind:<18} {n}")


if __name__ == "__main__":
    _cli()
