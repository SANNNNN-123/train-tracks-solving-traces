"""Collect the policy's own branch states and label them from the solution.

Run-05 trained on *teacher* states — boards the oracle reaches — and cut dev loss
58% without changing a single branch decision on the puzzles that fail. The model
is already right on the boards the teacher visits, and wrong on the boards it
creates itself. This module records those boards (DAgger).

Two things are stored per branch decision:

* the exact board the policy saw, so the training prompt is byte-identical to the
  one the evaluator would build;
* whether the audited value was still in the menu. A poisoned board can have
  eliminated the true tile; those states cannot be labelled and are counted
  separately rather than silently dropped or, worse, trained with the other value.

The collector wraps any policy callable, so the same code runs against an adapter
on Kaggle and against the CPU oracle used by the tests.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from data_prep import build_examples
from evaluate_kaggle import load_policy, rollout
from medium_traces import reference_board, write_jsonl


def branch_action(actions):
    """Return the branch action offered by this menu, or ``None``."""
    return next((action for action in actions if action["type"] == "branch"), None)


def collect(policy, record, args, log=None):
    """Run one puzzle under ``policy`` and record every branch state it sees.

    Returns ``(states, result)`` where each state is a JSON-ready dict.  States are
    keyed by the step index the harness used, so a refused/invalid reply still
    lines up with the event it produced.
    """
    truth_board = reference_board(record)
    states = []
    counter = {"step": 0}

    def recorder(state, actions):
        step = counter["step"]
        counter["step"] += 1
        branch = branch_action(actions)
        if branch is not None:
            cell = branch["cell"]
            truth = truth_board[cell["row"] - 1][cell["column"] - 1]
            offered = [action["value"] for action in actions
                       if action["type"] == "branch" and action["cell"] == cell]
            states.append({
                "puzzle_id": record["puzzle_id"],
                "step": step,
                "cell": cell,
                "truth": truth,
                "offered": offered,
                "recoverable": truth in offered,
                "state": copy.deepcopy(state),
            })
        return policy(state, actions)

    result = rollout(recorder, record, args)

    # Fill in what the policy actually committed at each of those states.
    events = {event["step"]: event for event in result["history"]}
    for item in states:
        event = events.get(item["step"]) or {}
        action = event.get("action") or {}
        item["model"] = action.get("value") if action.get("type") == "branch" else None
        item["override"] = bool((event.get("branch_search") or {}).get("override"))

    if log:
        correct = sum(item["model"] == item["truth"] for item in states)
        log(f"  {record['puzzle_id']:18s} branches {len(states):2d} "
            f"correct {correct:2d} unrecoverable "
            f"{sum(not item['recoverable'] for item in states):2d} "
            f"actions {result['actions']:3d} "
            f"{'solved' if result['solved'] else 'failed'}")
    return states, result


def dedupe(states):
    """Drop repeated boards: a revisit after undo is the same training prompt."""
    seen, unique, repeated = set(), [], 0
    for item in states:
        key = (item["puzzle_id"], json.dumps(item["state"], sort_keys=True))
        if key in seen:
            repeated += 1
            continue
        seen.add(key)
        unique.append(item)
    return unique, repeated


def build_from_states(states, *, branch_repeat: int = 1, reverse_orders: bool = True):
    """Turn recorded states into ``user``/``completion`` rows.

    Every row's target is the audited value, so a wrong model choice becomes a
    corrected positive.  States where the truth was no longer offered are counted
    and skipped: there is nothing to teach on a board that cannot reach the
    solution.
    """
    unique, repeated = dedupe(states)
    rows = []
    stats = {
        "states": len(states),
        "repeated_boards": repeated,
        "unique_states": len(unique),
        "model_correct": sum(item["model"] == item["truth"] for item in unique),
        "model_wrong": sum(item["model"] != item["truth"] for item in unique),
        "unrecoverable": sum(not item["recoverable"] for item in unique),
        "puzzles": len({item["puzzle_id"] for item in unique}),
    }
    for item in unique:
        if not item["recoverable"]:
            continue
        rows.append({
            "puzzle_id": item["puzzle_id"],
            "step_index": item["step"],
            "state": item["state"],
            "action": {"type": "branch", "cell": item["cell"], "value": item["truth"]},
            "expert_policy_label": True,
        })

    examples, build_stats = build_examples(
        rows, None, branch_repeat=branch_repeat,
        reverse_branch_orders=reverse_orders,
        corrected_wrong_branches=False,
        max_deductions_per_puzzle=0)
    stats["labelled_rows"] = len(rows)
    stats["examples"] = len(examples)
    stats["by_kind"] = build_stats["by_kind"]
    stats["estimated_tokens"] = int(
        sum(len(example["user"]) + len(example["completion"]) for example in examples) / 3.6)
    return examples, stats


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=None,
                        help="train-puzzles-medium.jsonl; with --build it is only used for a split check")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--only", default=None,
                        help="Comma-separated ids to roll out")
    parser.add_argument("--limit", type=int, default=0,
                        help="First N of --only, for a smoke run")
    parser.add_argument("--max-actions", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--branch-search-depth", type=int, default=2)
    parser.add_argument("--branch-search-budget", type=int, default=4000)
    parser.add_argument("--output", type=Path, default=None,
                        help="JSONL of recorded states (collection mode)")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--build", action="store_true",
                        help="Build examples from --states instead of running the model")
    parser.add_argument("--states", default=None,
                        help="Comma-separated state JSONL files for --build")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--role", choices=("train", "dev"), default="train")
    parser.add_argument("--branch-repeat", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.build:
        if not args.states or not args.output_dir:
            raise SystemExit("--build needs --states and --output-dir")
        states = []
        for name in [part.strip() for part in args.states.split(",") if part.strip()]:
            path = Path(name)
            with path.open(encoding="utf-8") as handle:
                chunk = [json.loads(line) for line in handle]
            print(f"{path.name}: {len(chunk)} states")
            states.extend(chunk)
        if args.only:
            wanted = {part.strip() for part in args.only.split(",") if part.strip()}
            states = [item for item in states if item["puzzle_id"] in wanted]
            print(f"filtered to {len(states)} states in {len(wanted)} puzzles")
        examples, stats = build_from_states(
            states, branch_repeat=1 if args.role == "dev" else args.branch_repeat,
            reverse_orders=args.role != "dev")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = args.output_dir / f"{args.role}-examples.jsonl"
        write_jsonl(out, examples)
        report = {"role": args.role, "states": len(states), **stats, "file": str(out)}
        (args.output_dir / f"{args.role}-report.json").write_text(
            json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({key: report[key] for key in
                          ("role", "states", "unique_states", "model_correct",
                           "model_wrong", "unrecoverable", "labelled_rows",
                           "examples", "estimated_tokens")}, indent=1))
        print(f"wrote {out}")
        return

    if not args.adapter or not args.data or not args.output:
        raise SystemExit("collection needs --adapter, --data and --output")
    records = [json.loads(line) for line in args.data.open(encoding="utf-8")]
    by_id = {record["puzzle_id"]: record for record in records}
    wanted = [part.strip() for part in (args.only or "").split(",") if part.strip()]
    if not wanted:
        raise SystemExit("--only is required for collection")
    missing = [pid for pid in wanted if pid not in by_id]
    if missing:
        raise SystemExit(f"--only: not in this data file: {missing}")
    if args.limit:
        wanted = wanted[:args.limit]

    policy = load_policy(args)
    rollout_args = argparse.Namespace(
        max_actions=args.max_actions, branch_search_depth=args.branch_search_depth,
        branch_search_budget=args.branch_search_budget,
        max_new_tokens=args.max_new_tokens, compact=True)

    all_states = []
    for index, puzzle_id in enumerate(wanted, 1):
        states, _ = collect(policy, by_id[puzzle_id], rollout_args,
                            log=lambda message, i=index: print(
                                f"[{i}/{len(wanted)}]{message}", flush=True))
        all_states.extend(states)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for item in all_states:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = {
        "output": str(args.output),
        "puzzles": len(wanted),
        "states": len(all_states),
        "model_correct": sum(item["model"] == item["truth"] for item in all_states),
        "unrecoverable": sum(not item["recoverable"] for item in all_states),
        "overrides": sum(item["override"] for item in all_states),
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
