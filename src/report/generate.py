"""Incident brief generator (spec section 1.3, PS-1 rescue/relief plan).

Calls the Claude API. Not a local LLM: that would eat the whole GPU budget for the
feature judges care least about, and the output would be worse.

The offline path is not a fallback model, it is a template. Every number in the brief
comes from the contracts, so the deterministic version says the same things in plainer
prose. On stage with the network down, that is what runs.

    python -m src.report.generate --offline
    ANTHROPIC_API_KEY=... python -m src.report.generate
"""
import argparse
import datetime as dt
import json
import os
import pathlib

from src.hazard.formulas import REPO_ROOT

MODEL = "claude-sonnet-5"
MOCK = REPO_ROOT / "data" / "mock"


def load_contracts(mock_dir=MOCK):
    """Read the four contracts. Missing files raise rather than degrading the brief."""
    out = {}
    for name in ("zones", "detections", "priority", "evacuation"):
        path = pathlib.Path(mock_dir) / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"{name}.json not found at {path}")
        out[name] = json.loads(path.read_text())
    return out


def summarize(contracts):
    """The facts a brief is built from. Computed once, used by both paths."""
    cells = contracts["zones"]["grid_cells"]
    by_zone = {z: [c for c in cells if c["zone"] == z] for z in ("RED", "BLUE", "GREEN")}
    priority = contracts["priority"]
    evac = contracts["evacuation"]

    return {
        "disaster": contracts["zones"]["disaster"],
        "generated_at": contracts["zones"]["generated_at"],
        "cells_total": len(cells),
        "red": len(by_zone["RED"]),
        "blue": len(by_zone["BLUE"]),
        "green": len(by_zone["GREEN"]),
        "at_risk_population": sum(c["pop"] for c in by_zone["RED"] + by_zone["BLUE"]),
        "critical_assets": [
            {"cell_id": c["cell_id"], "assets": [a["name"] for a in c["assets"]]}
            for c in by_zone["RED"] if c.get("assets")
        ],
        "recon_queue": [
            {"cell_id": c["cell_id"], "reason": c["zone_reason"], "pop": c["pop"]}
            for c in sorted(by_zone["BLUE"], key=lambda c: -c["pop"])
        ],
        "dispatch": priority["ranked"],
        "unassigned": priority["unassigned"],
        "budget": f"{priority['budget_used_team_hours']}/"
                  f"{priority['budget_total_team_hours']} team-hours",
        "shelters": evac["shelters"],
        "overflow": evac["overflow"],
        "evacuating": sum(f["people"] for f in evac["flows"]),
    }


def offline_brief(s):
    """Deterministic one-page brief. Every figure traces to a contract field."""
    lines = [
        f"INCIDENT BRIEF — {s['disaster'].upper()}",
        f"Generated {s['generated_at']}",
        "",
        "SITUATION",
        f"  {s['cells_total']} cells assessed: {s['red']} RED, {s['blue']} BLUE, "
        f"{s['green']} GREEN.",
        f"  {s['at_risk_population']:,} people in RED or BLUE cells.",
        "",
        "IMMEDIATE DISPATCH",
    ]
    for r in s["dispatch"]:
        team = r.get("team") or "unassigned"
        lines.append(f"  {r['rank']}. {r['cell_id']}  {team}  ETA {r['eta_min']} min  "
                     f"{r['n_est']} est. survivors, P(alive) {r['p_alive']:.2f}")
    lines.append(f"  Budget {s['budget']}.")
    if s["unassigned"]:
        lines.append(f"  Beyond budget: {', '.join(s['unassigned'])}.")

    lines += ["", "RECON QUEUE (uncertain, not yet safe to declare)"]
    for c in s["recon_queue"]:
        lines.append(f"  {c['cell_id']}  {c['pop']:,} people  — {c['reason']}")
    if not s["recon_queue"]:
        lines.append("  None. Every cell is classified with confidence.")

    lines += ["", "CRITICAL INFRASTRUCTURE IN RED ZONES"]
    for c in s["critical_assets"]:
        lines.append(f"  {c['cell_id']}: {', '.join(c['assets'])}")
    if not s["critical_assets"]:
        lines.append("  None recorded in RED cells.")

    lines += ["", "EVACUATION"]
    lines.append(f"  {s['evacuating']:,} people routed to shelter.")
    for sh in s["shelters"]:
        lines.append(f"  {sh['name']}: {sh['assigned']:,}/{sh['capacity']:,} "
                     f"({sh['utilization']:.0%})")
    for o in s["overflow"]:
        lines.append(f"  OVERFLOW {o['people']:,} from {o['from_cell']}: {o['reason']}")

    lines += ["", "Figures are model output, not ground truth. Confirm before committing "
              "teams."]
    return "\n".join(lines)


PROMPT = """You are drafting an incident brief for a district emergency operations centre
during an active {disaster} event. Your reader is an operations director who will commit
teams based on this page.

Write at most one page, in this order: situation, immediate dispatch, recon queue,
critical infrastructure, evacuation. Plain prose and short lists. No preamble, no
markdown headers beyond the section names.

Rules:
- Use only the figures in the data below. Do not add, round away, or estimate any number.
- BLUE zones are not "moderate risk". They are cells the system cannot yet call safe,
  and they are a recon queue. Say so plainly if any exist.
- If the dispatch plan leaves cells unassigned, say which and why the budget ran out.
- Close with one line noting these are model outputs requiring confirmation.

DATA
{data}"""


def claude_brief(s, api_key=None, model=MODEL, timeout=60):
    """Ask Claude for the brief. Raises on any failure so the caller chooses the fallback."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise PermissionError("ANTHROPIC_API_KEY is not set")

    import urllib.error
    import urllib.request

    body = json.dumps({
        "model": model,
        "max_tokens": 1500,
        "messages": [{"role": "user",
                      "content": PROMPT.format(disaster=s["disaster"],
                                               data=json.dumps(s, indent=2))}],
    }).encode()

    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body,
        headers={"content-type": "application/json", "x-api-key": key,
                 "anthropic-version": "2023-06-01"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as e:
        raise ConnectionError(f"Claude API returned {e.code}: {e.read()[:200].decode()}")
    except (urllib.error.URLError, TimeoutError) as e:
        raise ConnectionError(f"could not reach the Claude API: {e}")

    return "".join(block.get("text", "") for block in payload.get("content", [])).strip()


def generate(mock_dir=MOCK, offline=False, api_key=None):
    """Returns (brief_text, source) where source is 'claude' or 'template'."""
    s = summarize(load_contracts(mock_dir))
    if offline:
        return offline_brief(s), "template"
    try:
        return claude_brief(s, api_key), "claude"
    except (PermissionError, ConnectionError) as e:
        return f"{offline_brief(s)}\n\n[Claude unavailable: {e}]", "template"


def _cli():
    ap = argparse.ArgumentParser(description="Generate the incident brief.")
    ap.add_argument("--offline", action="store_true", help="template only, no API call")
    ap.add_argument("--mock-dir", default=str(MOCK))
    ap.add_argument("-o", "--out")
    args = ap.parse_args()

    text, source = generate(args.mock_dir, args.offline)
    if args.out:
        path = pathlib.Path(args.out)
        if not path.is_absolute():
            path = REPO_ROOT / path
        path.write_text(text + "\n", encoding="utf-8")
        print(f"{path.relative_to(REPO_ROOT)} written from {source} "
              f"at {dt.datetime.now().strftime('%H:%M:%S')}")
    else:
        print(text)


if __name__ == "__main__":
    _cli()
