"""Risk-aware road graph (spec section 7.1).

    c(e) = (L_e / v_e) * (1 + alpha*X_e + beta*U_e) * [1 + 0.15*(f_e/cap_e)^4]
    c(e) = infinity  if  X_e >= 0.75  or  bridge_status_e == "down"

Three things are happening in that cost. Travel time is the base. The middle term
penalises routing through hazard and, more weakly, through cells we are unsure about.
The bracket is the BPR congestion function, which stops the optimiser sending fifty
thousand people down one road because it happened to be shortest.

Answers PS-1 question 4: the safest and fastest route, where safest and fastest are
traded against each other explicitly rather than by picking one.

    python -m src.routing.graph --demo
"""
import argparse
import heapq
import math

from src.hazard.zones import load_global

INF = float("inf")


def edge_cost(length_m, speed_kmh, hazard_x, uncertainty_u, flow=0.0, capacity=None,
              cfg=None, bridge_down=False):
    """Seconds of effective travel time, or infinity when the edge is impassable."""
    cfg = cfg or load_global()["routing"]
    alpha = cfg["alpha_hazard"]
    beta = cfg["beta_uncertainty"]
    impassable = cfg["impassable_if_hazard_gte"]

    if bridge_down or hazard_x >= impassable:
        return INF
    if length_m < 0:
        raise ValueError("edge length cannot be negative")
    if speed_kmh <= 0:
        raise ValueError("speed must be > 0")

    base_s = length_m / (speed_kmh / 3.6)
    risk = 1.0 + alpha * hazard_x + beta * uncertainty_u

    congestion = 1.0
    if capacity:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        congestion = 1.0 + cfg["bpr_coefficient"] * (flow / capacity) ** cfg["bpr_exponent"]

    return base_s * risk * congestion


def build_graph(edges, zones_by_cell, cfg=None):
    """edges: [{u, v, length_m, speed_kmh, capacity, bridge_down, cell_id}] -> adjacency.

    Each edge inherits X and U from the cell it passes through. An edge whose cell is
    not in the zone map raises: routing through terrain nobody scored is exactly the
    silent failure the spec forbids.
    """
    cfg = cfg or load_global()["routing"]
    adjacency = {}
    for e in edges:
        cid = e.get("cell_id")
        if cid not in zones_by_cell:
            raise ValueError(
                f"edge {e['u']}->{e['v']} lies in cell {cid!r}, which has no hazard score. "
                f"Run the zone engine over the full road extent first."
            )
        cell = zones_by_cell[cid]
        cost = edge_cost(e["length_m"], e["speed_kmh"], cell["X"], cell["U"],
                         e.get("flow", 0.0), e.get("capacity"), cfg,
                         e.get("bridge_down", False))
        adjacency.setdefault(e["u"], []).append((e["v"], cost, e))
        if not e.get("oneway"):
            adjacency.setdefault(e["v"], []).append((e["u"], cost, e))
        adjacency.setdefault(e["v"], adjacency.get(e["v"], []))
    return adjacency


def shortest_path(adjacency, source, target):
    """Dijkstra over the risk-weighted costs -> (path, seconds).

    Returns ([], inf) when the target is unreachable, which is a real answer during a
    flood and must not be confused with a zero-cost route.
    """
    if source not in adjacency:
        raise KeyError(f"unknown source node {source!r}")
    dist = {source: 0.0}
    prev = {}
    seen = set()
    queue = [(0.0, source)]

    while queue:
        d, node = heapq.heappop(queue)
        if node in seen:
            continue
        seen.add(node)
        if node == target:
            break
        for neighbour, cost, _e in adjacency.get(node, []):
            if cost == INF:
                continue
            nd = d + cost
            if nd < dist.get(neighbour, INF):
                dist[neighbour] = nd
                prev[neighbour] = node
                heapq.heappush(queue, (nd, neighbour))

    if target not in dist:
        return [], INF

    path, node = [target], target
    while node in prev:
        node = prev[node]
        path.append(node)
    return path[::-1], dist[target]


def eta_minutes(adjacency, source, target):
    """Travel time in minutes, or None when there is no route."""
    _path, seconds = shortest_path(adjacency, source, target)
    return None if seconds == INF else round(seconds / 60.0, 1)


def reachable_from(adjacency, source):
    """Every node reachable at finite cost. Feeds the isolation factor in section 4.4."""
    seen, stack = {source}, [source]
    while stack:
        node = stack.pop()
        for neighbour, cost, _e in adjacency.get(node, []):
            if cost < INF and neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return seen


def load_osm_graph(place, network_type="drive"):
    """Download a real road network. Download phase only, never on stage."""
    try:
        import osmnx as ox
    except ImportError:
        raise ImportError("road graphs need osmnx: pip install osmnx networkx")
    g = ox.graph_from_place(place, network_type=network_type)
    edges = []
    for u, v, data in g.edges(data=True):
        speed = data.get("speed_kph") or 30.0
        edges.append({
            "u": u, "v": v,
            "length_m": float(data.get("length", 0.0)),
            "speed_kmh": float(speed),
            "oneway": bool(data.get("oneway", False)),
            "capacity": 1800.0,               # one lane, vehicles per hour
            "cell_id": None,                  # caller snaps these onto the grid
        })
    return edges


def _demo():
    cfg = load_global()["routing"]
    zones = {
        "A": {"X": 0.05, "U": 0.10},
        "B": {"X": 0.82, "U": 0.20},          # over the impassable threshold
        "C": {"X": 0.35, "U": 0.55},          # passable but uncertain
        "D": {"X": 0.10, "U": 0.10},
    }
    edges = [
        {"u": "depot", "v": "via_flood", "length_m": 1000, "speed_kmh": 40, "cell_id": "B"},
        {"u": "via_flood", "v": "site", "length_m": 1000, "speed_kmh": 40, "cell_id": "A"},
        {"u": "depot", "v": "via_unsure", "length_m": 1600, "speed_kmh": 40, "cell_id": "C"},
        {"u": "via_unsure", "v": "site", "length_m": 1600, "speed_kmh": 40, "cell_id": "D"},
    ]
    adjacency = build_graph(edges, zones, cfg)
    path, seconds = shortest_path(adjacency, "depot", "site")

    print("two routes from depot to site:")
    print(f"  through the flooded cell   2.0 km, X=0.82 -> impassable")
    print(f"  through the uncertain cell 3.2 km, X=0.35 U=0.55")
    print(f"\nchosen: {' -> '.join(path)}  ({seconds / 60:.1f} min effective)")
    print(f"straight-line time for the same 3.2 km at 40 km/h: "
          f"{3200 / (40 / 3.6) / 60:.1f} min")
    print("The detour is longer in distance and shorter in expected cost, which is the")
    print("trade the risk weighting exists to make.")


def _cli():
    ap = argparse.ArgumentParser(description="Risk-weighted road routing.")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        _demo()
    else:
        ap.print_help()


if __name__ == "__main__":
    _cli()
