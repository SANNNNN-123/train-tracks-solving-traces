"""CPU self-check for the Kaggle evaluation harness.

Two independent checks, both without a GPU:

1. **Oracle rollout** — an oracle policy that reads the audited solution drives
   the real rollout loop.  This exercises board updates for deductions, branches
   and ``finish``, completion detection and the solved metric, and proves the
   depth-2 filter never overrides a correct branch on a clean board.
2. **Run-03 replay** — the recorded medium rollout is re-driven step by step
   through the same harness.  It reproduces run-03 exactly (14/32 solved, 123
   undo actions, zero mismatches), which pins down the branch stack and the
   ``pending`` undo path that the oracle path never triggers.

Usage (from the ``kaggle/`` directory):

    python selfcheck.py                 # all 32 medium + 32 dev puzzles + replay
    python selfcheck.py --limit 6 --small-limit 4   # quick pass
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from action_policy import choice_id, completion, development_ids  # noqa: E402
from evaluate_kaggle import rollout, select_medium_sample  # noqa: E402


def oracle_policy(reference):
    """Play the audited solution; asserts the rule engine never contradicts it."""

    def policy(state, actions):
        kinds = {action["type"] for action in actions}
        if "undo" in kinds:
            action = next(a for a in actions if a["type"] == "undo")
        elif "contradiction" in kinds:
            raise AssertionError(f"clean board produced a contradiction: {actions}")
        elif "finish" in kinds:
            action = {"type": "finish"}
        elif "branch" in kinds:
            branches = [a for a in actions if a["type"] == "branch"]
            cell = branches[0]["cell"]
            truth = reference[cell["row"] - 1][cell["column"] - 1]
            action = next((a for a in branches if a["value"] == truth), branches[0])
        else:
            action = next(a for a in actions if a["type"] == "deduce")
            cell = action["cell"]
            truth = reference[cell["row"] - 1][cell["column"] - 1]
            if truth not in action["candidates"]:
                raise AssertionError(f"unsound deduction at {cell}: {action}")
        return completion(choice_id(action, actions))

    return policy


def solve(record, depth, args):
    reference = [[tile or "XX" for tile in row] for row in record["solution"]]
    result = rollout(oracle_policy(reference), record, args)
    branches = [event for event in result["history"]
                if event.get("branch_search")]
    return result, branches


def replay_policy(history):
    """Return the recorded raw completions, in order."""
    cursor = {"index": 0}

    def policy(state, actions):
        index = cursor["index"]
        if index >= len(history):
            raise AssertionError("harness ran past the recorded rollout")
        cursor["index"] += 1
        return history[index]["raw"]

    return policy


def check_replay(path, records, args):
    """Re-drive the recorded run-03 rollouts; the harness must reproduce them.

    This is the regression test for the branch stack, the pending undo path and
    the solved metric: 123 of the recorded undo actions live in these traces.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    mismatches = []
    undone = 0
    for puzzle in data["puzzles"]:
        record = records[puzzle["puzzle_id"]]
        expected = puzzle["reference"]
        actual = [[tile or "XX" for tile in row] for row in record["solution"]]
        if actual != expected:
            mismatches.append(f"{puzzle['puzzle_id']}: source solution differs from rollout")
            continue
        result = rollout(replay_policy(puzzle["history"]), record, args)
        undone += sum(event["action"]["type"] == "undo" for event in puzzle["history"])
        if (result["solved"] != puzzle["solved"]
                or result["failure"] != puzzle["failure"]
                or result["actions"] != len(puzzle["history"])):
            mismatches.append(
                f"{puzzle['puzzle_id']}: expected solved={puzzle['solved']} "
                f"failure={puzzle['failure']} steps={len(puzzle['history'])}; got "
                f"solved={result['solved']} failure={result['failure']} "
                f"steps={result['actions']}")
    solved = sum(puzzle["solved"] for puzzle in data["puzzles"])
    print(f"{'replay run-03':18s} depth 0: solved {solved}/{len(data['puzzles'])}  "
          f"undos {undone}  mismatches {len(mismatches)}")
    return mismatches


def report(name, pairs, depth):
    solved = sum(result["solved"] for result, _ in pairs)
    overrides = sum(result["search_overrides"] for result, _ in pairs)
    branches = sum(len(branches) for _, branches in pairs)
    both = sum(1 for _, branches in pairs for event in branches
               if len(event["branch_search"]["survivors"] or []) == 2)
    seconds = sum(result["seconds"] for result, _ in pairs)
    print(f"{name:18s} depth {depth}: solved {solved}/{len(pairs)}  "
          f"branches {branches}  both-survive {both}  "
          f"overrides {overrides}  {seconds:.1f}s")
    return solved, overrides


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--medium", type=Path,
                        default=HERE / "data/train-puzzles-medium.jsonl")
    parser.add_argument("--small", type=Path,
                        default=HERE / "data/train-puzzles-small.jsonl")
    parser.add_argument("--limit", type=int, default=32,
                        help="medium puzzles, 0 means all")
    parser.add_argument("--small-limit", type=int, default=32,
                        help="dev-holdout puzzles, 0 means all")
    parser.add_argument("--max-actions", type=int, default=256)
    parser.add_argument("--replay", type=Path,
                        default=HERE.parent / "output/eval/run-03-legal-action-menu/medium/rollout.json",
                        help="Recorded run-03 rollout to re-drive; skipped when missing")
    args = parser.parse_args()

    run_args = SimpleNamespace(
        max_actions=args.max_actions,
        max_new_tokens=32,
        branch_search_depth=0,
        branch_search_budget=4000,
        compact=True,
    )

    medium = [json.loads(line) for line in args.medium.open(encoding="utf-8")]
    medium = [record for record in medium if record.get("split") == "test_medium"]
    medium_records = {record["puzzle_id"]: record for record in medium}
    if args.limit:
        medium = select_medium_sample(medium, args.limit)

    small = [json.loads(line) for line in args.small.open(encoding="utf-8")]
    dev_ids = development_ids(record["puzzle_id"] for record in small)
    dev = sorted((record for record in small if record["puzzle_id"] in dev_ids),
                 key=lambda record: record["puzzle_id"])
    if args.small_limit:
        dev = dev[:args.small_limit]

    failures = []
    started = time.monotonic()
    for name, puzzles in (("medium", medium), ("small dev", dev)):
        for depth in (0, 2):
            run_args.branch_search_depth = depth
            pairs = [solve(record, depth, run_args) for record in puzzles]
            solved, overrides = report(name, pairs, depth)
            if solved != len(pairs):
                bad = [result["puzzle_id"] for result, _ in pairs if not result["solved"]]
                failures.append(f"{name} depth {depth}: failed {bad}")
            if depth > 0 and overrides:
                failures.append(f"{name} depth {depth}: {overrides} overrides on clean boards")

    if args.replay and args.replay.exists():
        run_args.branch_search_depth = 0
        failures.extend(check_replay(args.replay, medium_records, run_args))
    else:
        print("replay: skipped (recorded rollout not found)")

    elapsed = time.monotonic() - started
    if failures:
        print("\nSELF-CHECK FAILED")
        for line in failures:
            print(" -", line)
        return 1
    print(f"\nSELF-CHECK PASSED in {elapsed:.1f}s "
          "(oracle solves every puzzle; filter never overrides)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
