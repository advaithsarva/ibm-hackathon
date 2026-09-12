"""Grid conventions (§1.2). Everything joins on cell_id, so it lives in one place."""

CELL_ID_PRECISION = 4


def cell_id(lat, lon):
    """Stable, joinable string key for a cell centroid."""
    return f"{lat:.{CELL_ID_PRECISION}f}_{lon:.{CELL_ID_PRECISION}f}"


def centroid(cid):
    """Inverse of cell_id. Raises on a malformed key rather than guessing."""
    try:
        lat, lon = cid.split("_")
        return float(lat), float(lon)
    except ValueError:
        raise ValueError(f"malformed cell_id {cid!r}, expected '{{lat:.4f}}_{{lon:.4f}}'")
