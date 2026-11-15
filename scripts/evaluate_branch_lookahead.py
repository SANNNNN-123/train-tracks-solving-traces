"""Offline check: would a 1-ply propagation lookahead choose better branch values?

For every branch decision in the run-03 medium rollout, apply each of the two
candidate values to a copy of the board, run the existing deduction engine to a
fixpoint (or contradiction), and compare the chosen value with the reference.
No model and no solution are used by the probe; the reference is only the label.
"""

import json
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "modal")
sys.path.insert(0, "scripts")
from action_policy import legal_actions  # noqa: E402

TILES = ("XX", "NS", "EW", "NE", "NW", "SE", "SW")
BIT = {name: 1 << i for i, name in enumerate(TILES)}
REF_INDEX = {name: i for i, name in enumerate(TILES)}


def mask(candidates):
    return sum(BIT[c] for c in candidates)


def probe(state, cell, value, limit=400):
    """Return (contradicted, deductions, detail) after assuming value at cell."""
    board = [row[:] for row in state["domains"]]
    r, c = cell["row"] - 1, cell["column"] - 1
    board[r][c] = BIT[value]
    local = {
        "row_clues": state["row_clues"],
        "column_clues": state["column_clues"],
        "givens": state["givens"],
        "domains": board,
        "branch_stack": list(state["branch_stack"]) + [
            {"to_step": -1, "cell": cell, "value": value}
        ],
    }
    deductions = 0
    for _ in range(limit):
        actions = legal_actions(local, pending=False)
        if any(a["type"] == "contradiction" for a in actions):
            return True, deductions, "contradiction"
        deduces = [a for a in actions if a["type"] == "deduce"]
        if not deduces:
            return False, deductions, "stall"
        for action in deduces:
            rr, cc = action["cell"]["row"] - 1, action["cell"]["column"] - 1
            new = mask(action["candidates"])
            if new & local["domains"][rr][cc] != new:
                return True, deductions, "invalid_probe"
            local["domains"][rr][cc] = new
            deductions += 1
    return False, deductions, "limit"


def score(contradicted, deductions):
    # Prefer a probe that does not contradict; then prefer more forced progress.
    return (0 if contradicted else 1, deductions)


def main():
    data = json.load(open("output/eval/run-03-legal-action-menu/medium/rollout.json"))
    rows = []
    for puzzle in data["puzzles"]:
        ref = puzzle["reference"]
        for event in puzzle["history"]:
            action = event["action"]
            if action["type"] != "branch":
                continue
            cell = action["cell"]
            truth = ref[cell["row"] - 1][cell["column"] - 1]
            dom = event["state"]["domains"][cell["row"] - 1][cell["column"] - 1]
            candidates = [t for t in TILES if dom & BIT[t]]
            clean = truth in candidates
            chosen = action["value"]
            results = {v: probe(event["state"], cell, v) for v in candidates}
            best = max(candidates, key=lambda v: score(*results[v][:2]))
            # tie-break toward the model's own choice when the probe cannot tell
            top = [v for v in candidates if score(*results[v][:2]) == score(*results[best][:2])]
            if chosen in top:
                best_tie = chosen
            else:
                best_tie = best
            rows.append(dict(puzzle=puzzle["puzzle_id"], solved=puzzle["solved"],
                             clean=clean, chosen=chosen, truth=truth, best=best,
                             best_tie=best_tie, candidates=candidates,
                             results=results, model_ok=chosen == truth,
                             probe_ok=best == truth, probe_tie_ok=best_tie == truth))

    def report(name, subset):
        n = len(subset)
        if not n:
            return
        print(f"{name}: n={n}")
        print(f"  model accuracy          {sum(r['model_ok'] for r in subset)}/{n} = {sum(r['model_ok'] for r in subset)/n:.1%}")
        print(f"  probe accuracy          {sum(r['probe_ok'] for r in subset)}/{n} = {sum(r['probe_ok'] for r in subset)/n:.1%}")
        print(f"  probe+tiebreak accuracy {sum(r['probe_tie_ok'] for r in subset)}/{n} = {sum(r['probe_tie_ok'] for r in subset)/n:.1%}")
        # how often the probe is decisive
        decisive = [r for r in subset if score(*r['results'][r['candidates'][0]][:2]) != score(*r['results'][r['candidates'][1]][:2])]
        print(f"  probe decisive          {len(decisive)}/{n} = {len(decisive)/n:.1%} of decisions")
        if decisive:
            print(f"    on decisive, probe acc {sum(r['probe_ok'] for r in decisive)}/{len(decisive)} = {sum(r['probe_ok'] for r in decisive)/len(decisive):.1%}")
        # breakdown by whether both contradict / one / neither
        kinds = Counter()
        for r in subset:
            c = [r['results'][v][0] for v in r['candidates']]
            kinds[tuple(c)] += 1
        print(f"  contradiction pattern (value a,b): {dict(kinds)}")
        print()

    report("ALL branch decisions", rows)
    report("CLEAN (reference still in domain)", [r for r in rows if r['clean']])
    report("SOLVED puzzles", [r for r in rows if r['solved']])
    report("FAILED puzzles", [r for r in rows if not r['solved']])
    report("FAILED + CLEAN", [r for r in rows if not r['solved'] and r['clean']])

    # Per-pair probe accuracy on clean rows
    print("Per-pair (clean rows):")
    by = defaultdict(list)
    for r in rows:
        if r["clean"]:
            by[tuple(sorted(r["candidates"]))].append(r)
    for pair, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        n = len(rs)
        print(f"  {pair}: n={n} model {sum(r['model_ok'] for r in rs)/n:.0%} probe {sum(r['probe_ok'] for r in rs)/n:.0%}")

    # Show a few probe failures for inspection
    print("\nExample clean probe failures:")
    for r in [r for r in rows if r["clean"] and not r["probe_ok"]][:8]:
        print(" ", r["puzzle"], r["candidates"], "chosen", r["chosen"], "truth", r["truth"],
              "probe", r["best"], {v: r["results"][v] for v in r["candidates"]})


if __name__ == "__main__":
    main()
