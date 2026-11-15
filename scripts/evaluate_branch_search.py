"""Offline check: bounded DFS over branch values using only the rule engine.

A value is rejected only when propagation plus a bounded search proves it
impossible. If the current board still contains the reference solution, the
true value must remain possible, so this filter is safe: it can only remove
wrong branches, never the correct one.

Usage: python scripts/evaluate_branch_search.py [depth]
"""

import json
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "modal")
sys.path.insert(0, "scripts")
from action_policy import legal_actions  # noqa: E402
from trace_logic import BIT, branch_cell, choices, completed_error  # noqa: E402

TILES = ("XX", "NS", "EW", "NE", "NW", "SE", "SW")


def clue_from_state(state):
    return {
        "rows": state["row_clues"],
        "columns": state["column_clues"],
        "given_shapes": [(g["row"] - 1, g["column"] - 1, g["value"]) for g in state["givens"]],
    }


def mask(candidates):
    return sum(BIT[c] for c in candidates)


def propagate(board, state, stack):
    """Apply all currently legal deductions to a fixpoint.

    Returns (contradicted, board, deductions). Stale actions are skipped; they
    were computed before an earlier action in the same round changed the cell.
    """
    board = [row[:] for row in board]
    deductions = 0
    for _ in range(400):
        local = {"row_clues": state["row_clues"], "column_clues": state["column_clues"],
                 "givens": state["givens"], "domains": board, "branch_stack": stack}
        actions = legal_actions(local, pending=False)
        if any(a["type"] == "contradiction" for a in actions):
            return True, board, deductions
        deduces = [a for a in actions if a["type"] == "deduce"]
        if not deduces:
            return False, board, deductions
        applied = 0
        for action in deduces:
            r, c = action["cell"]["row"] - 1, action["cell"]["column"] - 1
            new = mask(action["candidates"])
            if new == board[r][c] or new & board[r][c] != new:
                continue  # stale or already applied
            board[r][c] = new
            deductions += 1
            applied += 1
        if not applied:
            return False, board, deductions
    return False, board, deductions


def possible(board, state, stack, depth):
    """True if the board can reach a locally consistent state within `depth` guesses."""
    contradicted, board, _ = propagate(board, state, stack)
    if contradicted:
        return False
    clue = clue_from_state(state)
    if branch_cell(board) is None:
        return completed_error(board, clue) is None
    if depth <= 0:
        return True  # unknown, conservatively possible
    r, c = branch_cell(board)
    cell = {"row": r + 1, "column": c + 1}
    for value in choices(board[r][c]):
        child = [row[:] for row in board]
        child[r][c] = BIT[value]
        child_stack = list(stack) + [{"to_step": -1, "cell": cell, "value": value}]
        if possible(child, state, child_stack, depth - 1):
            return True
    return False


def main():
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    data = json.load(open("output/eval/run-03-legal-action-menu/medium/rollout.json"))
    rows = []
    for puzzle in data["puzzles"]:
        ref = puzzle["reference"]
        H, W = len(ref), len(ref[0])
        for event in puzzle["history"]:
            action = event["action"]
            if action["type"] != "branch":
                continue
            state = event["state"]
            board = state["domains"]
            fully_clean = all(
                (board[r][c] & BIT[ref[r][c]]) for r in range(H) for c in range(W)
            )
            cell = action["cell"]
            r, c = cell["row"] - 1, cell["column"] - 1
            truth = ref[r][c]
            candidates = [t for t in TILES if board[r][c] & BIT[t]]
            feasible = {}
            for value in candidates:
                child = [row[:] for row in board]
                child[r][c] = BIT[value]
                child_stack = list(state["branch_stack"]) + [
                    {"to_step": -1, "cell": cell, "value": value}]
                feasible[value] = possible(child, state, child_stack, depth)
            survivors = [v for v in candidates if feasible[v]]
            chosen = action["value"]
            # safe choice: keep the model's value if it survives, else a survivor.
            if chosen in survivors or not survivors:
                pick = chosen
            else:
                pick = survivors[0]
            rows.append(dict(puzzle=puzzle["puzzle_id"], solved=puzzle["solved"],
                             clean=fully_clean, candidates=candidates, truth=truth,
                             chosen=chosen, survivors=survivors, pick=pick,
                             model_ok=chosen == truth, pick_ok=pick == truth,
                             true_survives=truth in survivors,
                             both=len(survivors) == 2, none=len(survivors) == 0))

    def report(name, subset):
        n = len(subset)
        if not n:
            return
        print(f"{name}: n={n}")
        print(f"  model accuracy              {sum(r['model_ok'] for r in subset)}/{n} = {sum(r['model_ok'] for r in subset)/n:.1%}")
        print(f"  safe-filter accuracy        {sum(r['pick_ok'] for r in subset)}/{n} = {sum(r['pick_ok'] for r in subset)/n:.1%}")
        print(f"  true value survived filter  {sum(r['true_survives'] for r in subset)}/{n} = {sum(r['true_survives'] for r in subset)/n:.1%}")
        print(f"  both values survived        {sum(r['both'] for r in subset)}/{n} = {sum(r['both'] for r in subset)/n:.1%}")
        print(f"  no value survived           {sum(r['none'] for r in subset)}/{n}")
        print()

    report("ALL", rows)
    report("FULLY CLEAN (board contains reference everywhere)", [r for r in rows if r["clean"]])
    report("SOLVED", [r for r in rows if r["solved"]])
    report("FAILED", [r for r in rows if not r["solved"]])
    report("FAILED + FULLY CLEAN", [r for r in rows if not r["solved"] and r["clean"]])

    print("Per-pair (fully clean):")
    by = defaultdict(list)
    for r in rows:
        if r["clean"]:
            by[tuple(sorted(r["candidates"]))].append(r)
    for pair, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        n = len(rs)
        print(f"  {pair}: n={n} model {sum(r['model_ok'] for r in rs)/n:.0%} filter {sum(r['pick_ok'] for r in rs)/n:.0%} both {sum(r['both'] for r in rs)/n:.0%}")


if __name__ == "__main__":
    main()
