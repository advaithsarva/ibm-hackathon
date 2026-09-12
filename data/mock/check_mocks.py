"""Re-derive every mock fixture from the rules in the spec and assert it matches.

Run: python data/mock/check_mocks.py
Fails loudly if a hand-edited fixture drifts from the contracts in §2 or the zone rule in §4.2.
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
load = lambda n: json.loads((HERE / n).read_text())


def zone_of(c):
    """§4.2, verbatim."""
    if c["X"] >= 0.75 or (c["X"] >= 0.40 and c["U"] >= 0.50):
        return "RED"
    if c["X"] < 0.40 and c["U"] < 0.35 and c["G"] >= 0.60 and c["reachable"]:
        return "GREEN"
    return "BLUE"


def noisy_or(u):
    """§4.1: U = 1 - (1-u_model)(1-u_stale)(1-u_cover)."""
    return 1 - (1 - u["u_model"]) * (1 - u["u_stale"]) * (1 - u["u_cover"])


def main():
    zones = load("zones.json")
    cells = {c["cell_id"]: c for c in zones["grid_cells"]}
    assert len(cells) == len(zones["grid_cells"]), "duplicate cell_id in zones.json"

    for cid, c in cells.items():
        lat, lon = c["centroid"]
        assert cid == f"{lat:.4f}_{lon:.4f}", f"{cid}: cell_id does not match centroid (§1.2)"
        for k in ("X", "U", "G"):
            assert 0.0 <= c[k] <= 1.0, f"{cid}: {k}={c[k]} outside [0,1]"
        assert c["zone"] == zone_of(c), f"{cid}: labelled {c['zone']}, rule says {zone_of(c)}"
        # U is a summary of its parts; allow rounding slack, not contradiction.
        assert abs(noisy_or(c["uncertainty_components"]) - c["U"]) < 0.05, \
            f"{cid}: U={c['U']} disagrees with its noisy-OR components (§4.1)"

    # every cell_id referenced anywhere must exist in zones.json
    refs = [(d["cell_id"], "detections") for d in load("detections.json")]
    prio = load("priority.json")
    refs += [(r["cell_id"], "priority.ranked") for r in prio["ranked"]]
    refs += [(c, "priority.unassigned") for c in prio["unassigned"]]
    evac = load("evacuation.json")
    refs += [(f["from_cell"], "evacuation.flows") for f in evac["flows"]]
    refs += [(o["from_cell"], "evacuation.overflow") for o in evac["overflow"]]
    for cid, where in refs:
        assert cid in cells, f"{where} references unknown cell {cid}"

    assert [r["rank"] for r in prio["ranked"]] == list(range(1, len(prio["ranked"]) + 1)), \
        "priority.ranked is not 1..n"
    pis = [r["pi"] for r in prio["ranked"]]
    assert pis == sorted(pis, reverse=True), "priority.ranked is not sorted by pi descending"
    assert prio["budget_used_team_hours"] <= prio["budget_total_team_hours"], "dispatch over budget (§6.3)"

    # §7.2: shelters respect capacity, and assigned matches the inbound flows
    for s in evac["shelters"]:
        inbound = sum(f["people"] for f in evac["flows"] if f["to_shelter"] == s["id"])
        assert inbound == s["assigned"], f"{s['id']}: flows sum to {inbound}, assigned says {s['assigned']}"
        assert s["assigned"] <= s["capacity"], f"{s['id']}: over capacity"
        assert abs(s["assigned"] / s["capacity"] - s["utilization"]) < 0.01, f"{s['id']}: utilization wrong"

    print(f"ok: {len(cells)} cells, {len(prio['ranked'])} ranked, {len(evac['shelters'])} shelters")


if __name__ == "__main__":
    main()
