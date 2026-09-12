"""Evacuation as min-cost flow (spec section 7.2).

    minimize  sum_ij c_ij * f_ij
    subject to  sum_j f_ij = Pop_i        everyone leaves
                sum_i f_ij <= Cap_j       shelters respect capacity
                f_ij >= 0

Shortest path alone sends everyone to the nearest shelter and overflows it. Capacity-
aware evacuation is the difference between a plan and a picture, and networkx does the
solving in one call.

    python -m src.routing.evacuation --demo
"""
import argparse
import json

from src.hazard.formulas import REPO_ROOT

SCALE = 1000          # networkx min_cost_flow needs integers; costs are scaled and rounded


def solve_flow(sources, shelters, costs):
    """sources: {cell_id: people}. shelters: {id: capacity}. costs: {(cell, shelter): min}.

    Returns (flows, overflow). A cell with no finite route to any shelter, or demand
    beyond total capacity, lands in overflow rather than being dropped or forced into a
    shelter that cannot hold it.
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("evacuation flow needs networkx: pip install networkx")

    total_demand = sum(sources.values())
    total_capacity = sum(shelters.values())

    routable = {c: {s for s in shelters if costs.get((c, s)) is not None}
                for c in sources}
    stranded = {c: p for c, p in sources.items() if not routable[c]}
    servable = {c: p for c, p in sources.items() if routable[c]}

    overflow = [{"from_cell": c, "people": p,
                 "reason": "no route to any shelter"} for c, p in stranded.items()]

    servable_demand = sum(servable.values())
    shortfall = max(0, servable_demand - total_capacity)

    # Node demands must sum to zero. SOURCE supplies everyone who can move; SINK absorbs
    # whoever fits in a shelter and UNPLACED absorbs the rest, so the two add back to it.
    g = nx.DiGraph()
    g.add_node("SOURCE", demand=-servable_demand)
    g.add_node("SINK", demand=(servable_demand - shortfall))

    for c, people in servable.items():
        g.add_edge("SOURCE", f"c:{c}", capacity=people, weight=0)
    for s, cap in shelters.items():
        g.add_edge(f"s:{s}", "SINK", capacity=cap, weight=0)
    for c in servable:
        for s in routable[c]:
            g.add_edge(f"c:{c}", f"s:{s}", capacity=servable[c],
                       weight=int(round(costs[(c, s)] * SCALE)))

    if shortfall > 0:
        # A relief valve so the solve stays feasible; whatever uses it is real overflow.
        g.add_node("UNPLACED", demand=shortfall)
        for c in servable:
            g.add_edge(f"c:{c}", "UNPLACED", capacity=servable[c],
                       weight=int(round(10_000 * SCALE)))

    result = nx.min_cost_flow(g)

    flows = []
    for c in servable:
        for s in routable[c]:
            moved = result[f"c:{c}"].get(f"s:{s}", 0)
            if moved > 0:
                flows.append({"from_cell": c, "to_shelter": s, "people": int(moved),
                              "travel_min": round(costs[(c, s)], 1)})
        unplaced = result[f"c:{c}"].get("UNPLACED", 0) if shortfall > 0 else 0
        if unplaced > 0:
            overflow.append({"from_cell": c, "people": int(unplaced),
                             "reason": "all reachable shelters at capacity"})

    return flows, overflow


def build_plan(zones, shelter_records, costs, evacuate_zones=("RED", "BLUE")):
    """Zone contract + shelters + travel times -> the section 2.4 /api/evacuation contract.

    BLUE is evacuated alongside RED by default. A cell we cannot call safe is not a cell
    to leave people in.
    """
    sources = {c["cell_id"]: c["pop"] for c in zones["grid_cells"]
               if c["zone"] in evacuate_zones and c["pop"] > 0}
    shelters = {s["id"]: s["capacity"] for s in shelter_records}

    flows, overflow = solve_flow(sources, shelters, costs)

    assigned = {s["id"]: 0 for s in shelter_records}
    for f in flows:
        assigned[f["to_shelter"]] += f["people"]

    return {
        "flows": sorted(flows, key=lambda f: -f["people"]),
        "shelters": [{
            "id": s["id"], "name": s["name"], "capacity": s["capacity"],
            "assigned": assigned[s["id"]],
            "utilization": round(assigned[s["id"]] / s["capacity"], 4) if s["capacity"] else 0.0,
            "location": s["location"],
        } for s in shelter_records],
        "overflow": overflow,
    }


def _demo():
    zones = {"grid_cells": [
        {"cell_id": "12.9716_77.5946", "zone": "RED", "pop": 1450},
        {"cell_id": "12.9750_77.5890", "zone": "RED", "pop": 2310},
        {"cell_id": "12.9702_77.5988", "zone": "RED", "pop": 980},
        {"cell_id": "12.9801_77.6102", "zone": "BLUE", "pop": 1670},
        {"cell_id": "12.9601_77.6033", "zone": "GREEN", "pop": 410},
    ]}
    shelters = [
        {"id": "shelter_07", "name": "Govt High School", "capacity": 2400,
         "location": [12.9601, 77.6033]},
        {"id": "shelter_11", "name": "Community Hall", "capacity": 2560,
         "location": [12.9555, 77.6150]},
    ]
    costs = {
        ("12.9716_77.5946", "shelter_07"): 26.0,
        ("12.9716_77.5946", "shelter_11"): 34.0,
        ("12.9750_77.5890", "shelter_07"): 31.0,
        ("12.9750_77.5890", "shelter_11"): 44.0,
        ("12.9702_77.5988", "shelter_07"): 19.0,
        ("12.9702_77.5988", "shelter_11"): 38.0,
        ("12.9801_77.6102", "shelter_11"): 22.0,
    }

    plan = build_plan(zones, shelters, costs)
    demand = sum(c["pop"] for c in zones["grid_cells"] if c["zone"] in ("RED", "BLUE"))
    capacity = sum(s["capacity"] for s in shelters)
    print(f"demand {demand} people, capacity {capacity} places\n")
    print(json.dumps(plan, indent=2))

    placed = sum(f["people"] for f in plan["flows"])
    spilled = sum(o["people"] for o in plan["overflow"])
    print(f"\nplaced {placed}, overflow {spilled}")
    print("Note shelter_07 is not filled by its nearest cells alone: the solver moves")
    print("people past it to shelter_11 so the total travel time stays minimal under")
    print("capacity, which shortest-path routing cannot do.")


def _cli():
    ap = argparse.ArgumentParser(description="Capacity-aware evacuation flow.")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--zones", help="a /api/zones contract file")
    ap.add_argument("-o", "--out")
    args = ap.parse_args()

    if args.demo:
        _demo()
        return
    if not args.zones:
        ap.error("pass --demo or --zones")

    import pathlib
    path = pathlib.Path(args.zones)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"zones file not found: {path}")
    raise SystemExit(
        "Real shelters and travel times are not wired yet. Build the road graph with "
        "src.routing.graph first, then pass its ETAs in as costs."
    )


if __name__ == "__main__":
    _cli()
