"""Dispatch as a knapsack (spec section 6.3).

    maximize  sum_i Pi_i * x_i    subject to  sum_i c_i * x_i <= B

Greedy on Pi/c gives a (1 - 1/e) approximation and runs in milliseconds. The spec says
ship the greedy, and it is right: the inputs carry more uncertainty than the 37% worst
case the approximation admits, so an exact solver would be false precision that costs
time we do not have.

Answers PS-1 question 3: which locations should emergency teams respond to first.

    python -m src.priority.dispatch --demo
"""
import argparse
import json

from src.hazard.formulas import REPO_ROOT, load_config
from src.priority.expected_lives import score_cells


def greedy_knapsack(scored, budget_team_hours, cost_key="cost_team_hours"):
    """Take cells in descending Pi per team-hour until the budget runs out.

    Continues past the first cell that does not fit rather than stopping: a cheap
    high-value cell later in the list should still be taken.
    """
    if budget_team_hours < 0:
        raise ValueError("budget cannot be negative")

    ranked = sorted(scored,
                    key=lambda c: c["pi"] / max(c.get(cost_key, 1.0), 1e-9),
                    reverse=True)
    chosen, unassigned, spent = [], [], 0.0
    for c in ranked:
        cost = c.get(cost_key, 1.0)
        if cost <= 0:
            raise ValueError(f"{c['cell_id']}: cost must be > 0")
        if spent + cost <= budget_team_hours:
            chosen.append(c)
            spent += cost
        else:
            unassigned.append(c)

    chosen.sort(key=lambda c: c["pi"], reverse=True)
    return chosen, unassigned, round(spent, 3)


def assign_teams(chosen, teams):
    """Give each chosen cell the best free team, nearest first among equals.

    teams: [{"id", "kind", "hours"}]. A cell with no team left stays unassigned rather
    than being handed to one that cannot do the job.
    """
    available = sorted(teams, key=lambda t: t["hours"], reverse=True)
    used, assigned, leftover = set(), [], []

    for c in sorted(chosen, key=lambda c: c["pi"], reverse=True):
        team = next((t for t in available
                     if t["id"] not in used and t["hours"] >= c.get("cost_team_hours", 1.0)),
                    None)
        if team is None:
            leftover.append(c)
            continue
        used.add(team["id"])
        assigned.append({**c, "team": team["id"], "team_kind": team["kind"]})

    return assigned, leftover


def build_plan(cells, disaster_cfg, budget_team_hours, teams=None, elapsed_hours=0.0):
    """Cells with detections -> the section 2.3 /api/priority contract."""
    scored = score_cells(cells, disaster_cfg, elapsed_hours)
    chosen, over_budget, spent = greedy_knapsack(scored, budget_team_hours)

    if teams:
        chosen, no_team = assign_teams(chosen, teams)
        over_budget = over_budget + no_team

    ranked = []
    for i, c in enumerate(sorted(chosen, key=lambda c: c["pi"], reverse=True), 1):
        ranked.append({
            "rank": i,
            "cell_id": c["cell_id"],
            "pi": c["pi"],
            "n_est": c["n_est"],
            "p_alive": c["p_alive"],
            "survivability": c["survivability"],
            "eta_min": c["eta_min"],
            "team": c.get("team"),
            "capability_match": c["capability_match"],
            "route": c.get("route", []),
        })

    return {
        "ranked": ranked,
        "unassigned": [c["cell_id"] for c in sorted(over_budget,
                                                    key=lambda c: c["pi"], reverse=True)],
        "budget_used_team_hours": spent,
        "budget_total_team_hours": budget_team_hours,
    }


def _demo():
    cfg = load_config("configs/disasters/flood.yaml")
    cells = [
        {"cell_id": "12.9716_77.5946", "n_est": 6, "p_alive": 0.94, "eta_min": 18,
         "cost_team_hours": 3.5, "team_kind": "boat",
         "route": [[12.9716, 77.5946], [12.9702, 77.5988]]},
        {"cell_id": "12.9702_77.5988", "n_est": 4, "p_alive": 0.79, "eta_min": 12,
         "cost_team_hours": 2.0, "team_kind": "swift_water"},
        {"cell_id": "12.9750_77.5890", "n_est": 14, "p_alive": 0.72, "eta_min": 90,
         "cost_team_hours": 6.0, "team_kind": "boat"},
        {"cell_id": "12.9688_77.6021", "n_est": 3, "p_alive": 0.22, "eta_min": 25,
         "cost_team_hours": 4.0, "team_kind": "medical"},
    ]
    teams = [{"id": "NDRF-1", "kind": "swift_water", "hours": 8.0},
             {"id": "NDRF-3", "kind": "boat", "hours": 8.0},
             {"id": "SDRF-2", "kind": "boat", "hours": 6.0}]

    plan = build_plan(cells, cfg, budget_team_hours=14.0, teams=teams)
    print(json.dumps(plan, indent=2))
    print(f"\nbudget {plan['budget_used_team_hours']}/{plan['budget_total_team_hours']} "
          f"team-hours, {len(plan['ranked'])} dispatched, "
          f"{len(plan['unassigned'])} left")


def _cli():
    ap = argparse.ArgumentParser(description="Greedy dispatch planner.")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--config", default="configs/disasters/flood.yaml")
    ap.add_argument("--cells", help="JSON list of cells with n_est, p_alive, eta_min")
    ap.add_argument("--budget", type=float, default=14.0)
    ap.add_argument("-o", "--out")
    args = ap.parse_args()

    if args.demo:
        _demo()
        return
    if not args.cells:
        ap.error("pass --demo or --cells")

    import pathlib
    path = pathlib.Path(args.cells)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"cells file not found: {path}")

    plan = build_plan(json.loads(path.read_text()), load_config(args.config), args.budget)
    text = json.dumps(plan, indent=2)
    if args.out:
        out = pathlib.Path(args.out)
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.write_text(text + "\n")
        print(f"{out.relative_to(REPO_ROOT)}: {len(plan['ranked'])} ranked")
    else:
        print(text)


if __name__ == "__main__":
    _cli()
