"""Build branch-state SFT examples from the audited medium solutions.

Medium boards are 9x9..10x10 and the run-04 adapter guesses the wrong branch on
roughly half of the hard ones, so this module turns the audited solutions into
supervised branch decisions for the *training* part of the medium pool.

Two properties matter:

* **Cheap.**  Medium rollouts are generated on CPU by an oracle that plays the
  audited solution, so a full trace costs under a second.  Only branch states
  plus a small sample of deductions are kept (~30 examples per puzzle instead of
  the ~170-step trace), which puts a complete 240-puzzle training set at
  **7,488 rows / ~6.9M tokens** (measured, 2.7 min on a laptop CPU).
* **Leak-free.**  ``build_pool`` generates examples for whatever records it is
  given and never chooses ids.  The one place ids are chosen is ``split_pool``,
  and it reuses the evaluator's stratified sampler so the evaluation ids are
  exactly the 32 run-04 scored, with a second held-out block left untouched and
  only the remainder trained on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from action_policy import choice_id, completion
from data_prep import build_examples
from evaluate_kaggle import rollout, select_medium_sample

# The 32 puzzles run-04 evaluated at 256 actions.  They are the evaluation set
# and must never appear in training; the constant only drives the assertion in
# `split_pool`, the selection itself comes from the evaluator's sampler.
RUN04_EVAL_IDS = [
    "01012026", "02022026", "03032026", "03092026", "04062026", "05022026",
    "05082026", "06052026", "08102026", "10012026", "12062026", "13072026",
    "14072026", "17042026", "18102026", "19042026", "19062026", "19102026",
    "20012026", "20062026", "22012026", "22022026", "24032026", "24102026",
    "25022026", "27012026", "28052026", "28102026", "29052026", "29082026",
    "30032026", "30082026",
]


def reference_board(record) -> list[list[str]]:
    return [[tile or "XX" for tile in row] for row in record["solution"]]


def oracle_action(actions, reference):
    """Play the audited solution: undo when pending, else take the true value."""
    kinds = {action["type"] for action in actions}
    if "undo" in kinds:
        return next(action for action in actions if action["type"] == "undo")
    if "contradiction" in kinds:
        raise AssertionError(f"clean board produced a contradiction: {actions}")
    if "finish" in kinds:
        return {"type": "finish"}
    if "branch" in kinds:
        branches = [action for action in actions if action["type"] == "branch"]
        cell = branches[0]["cell"]
        truth = reference[cell["row"] - 1][cell["column"] - 1]
        return next((a for a in branches if a["value"] == truth), branches[0])
    return next(action for action in actions if action["type"] == "deduce")


def oracle_rows(record, max_actions: int = 512):
    """Replay the audited solution and return ``(rows, result)``.

    Each row is ``{puzzle_id, step_index, state, action, expert_policy_label}``,
    the format ``data_prep.build_examples`` consumes.
    """
    reference = reference_board(record)
    rows = []

    def policy(state, actions):
        action = oracle_action(actions, reference)
        rows.append({
            "puzzle_id": record["puzzle_id"],
            "step_index": len(rows),
            "state": state,
            "action": action,
            "expert_policy_label": True,
        })
        return completion(choice_id(action, actions))

    args = type("TraceArgs", (), {
        "max_actions": max_actions,
        "branch_search_depth": 0,
        "branch_search_budget": 1,
        "max_new_tokens": 16,
        "compact": True,
    })()
    result = rollout(policy, record, args)
    return rows, result


def puzzle_examples(record, *, max_deductions_per_puzzle: int = 8,
                    branch_repeat: int = 1, max_actions: int = 512):
    """Return ``(examples, stats, result)`` for one puzzle."""
    rows, result = oracle_rows(record, max_actions=max_actions)
    if not result["solved"]:
        raise AssertionError(
            f"{record['puzzle_id']}: oracle rollout failed ({result['failure']})")
    examples, stats = build_examples(
        rows, {record["puzzle_id"]: reference_board(record)},
        branch_repeat=branch_repeat,
        max_deductions_per_puzzle=max_deductions_per_puzzle)
    for example in examples:
        example["puzzle_id"] = record["puzzle_id"]
    return examples, stats, result


def build_pool(records, *, max_deductions_per_puzzle: int = 8,
               branch_repeat: int = 1, max_actions: int = 512, log=None):
    """Build examples for a list of puzzle records, one puzzle at a time.

    Returns ``(examples, report)`` where the report carries per-puzzle counts so
    a collapsed generation (e.g. an oracle failure) is visible immediately.
    """
    examples = []
    report = {"puzzles": 0, "oracle_actions": 0, "branch_examples": 0,
              "deduce_examples": 0, "by_kind": {}, "puzzles_with_short_trace": []}
    for index, record in enumerate(records, 1):
        built, stats, result = puzzle_examples(
            record, max_deductions_per_puzzle=max_deductions_per_puzzle,
            branch_repeat=branch_repeat, max_actions=max_actions)
        examples.extend(built)
        report["puzzles"] += 1
        report["oracle_actions"] += result["actions"]
        report["branch_examples"] += sum(
            1 for example in built
            if example["kind"] in {"branch", "branch_corrected"})
        report["deduce_examples"] += sum(
            1 for example in built if example["kind"] == "deduce")
        for kind, count in stats["by_kind"].items():
            report["by_kind"][kind] = report["by_kind"].get(kind, 0) + count
        if result["actions"] < 80:
            report["puzzles_with_short_trace"].append(record["puzzle_id"])
        if log and (index % 20 == 0 or index == len(records)):
            log(f"generated {index}/{len(records)} puzzles, "
                f"{len(examples):,} examples")
    return examples, report


def write_jsonl(path, examples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    return len(examples)


def split_pool(records, eval_count: int = 32, fresh_count: int = 32):
    """Split the medium pool into ``(train, eval, fresh)`` record lists.

    ``eval`` is the block run-04 measured, so the comparison is like-for-like;
    ``fresh`` is a second untouched block used as development loss; ``train`` is
    the remainder.  The three are disjoint and cover the input.
    """
    records = sorted(records, key=lambda record: record["puzzle_id"])
    eval_records = select_medium_sample(records, eval_count)
    eval_ids = {record["puzzle_id"] for record in eval_records}
    expected = {f"{pid}-medium" for pid in RUN04_EVAL_IDS}
    if eval_ids != expected:
        extra = sorted(eval_ids - expected)
        missing = sorted(expected - eval_ids)
        raise AssertionError(
            f"medium split drifted: {len(eval_ids)} eval ids, "
            f"only-in-input {extra[:5]}, only-in-run04 {missing[:5]}")

    remaining = [record for record in records if record["puzzle_id"] not in eval_ids]
    fresh_records = select_medium_sample(remaining, fresh_count)
    fresh_ids = {record["puzzle_id"] for record in fresh_records}
    train_records = [record for record in remaining
                     if record["puzzle_id"] not in fresh_ids]
    if eval_ids & fresh_ids or len(train_records) + len(eval_ids) + len(fresh_ids) != len(records):
        raise AssertionError("medium split is not disjoint")
    return train_records, eval_records, fresh_records


def estimate_tokens(examples, chars_per_token: float = 3.6) -> int:
    """Rough token count without loading a tokenizer."""
    chars = sum(len(example["user"]) + len(example["completion"])
                for example in examples)
    return int(chars / chars_per_token)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="train-puzzles-medium.jsonl from the data dataset")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-deductions", type=int, default=16,
                        help="Deduction examples kept per training puzzle")
    parser.add_argument("--dev-deductions", type=int, default=4,
                        help="Deduction examples kept per held-out puzzle")
    parser.add_argument("--branch-repeat", type=int, default=1)
    parser.add_argument("--max-actions", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap each pool for a smoke run (the split check still runs)")
    args = parser.parse_args()

    records = [json.loads(line) for line in args.data.open(encoding="utf-8")]
    medium = [record for record in records if record.get("split") == "test_medium"]
    print(f"medium puzzles: {len(medium)} (of {len(records)} records)")

    train_records, eval_records, fresh_records = split_pool(medium)
    print(f"split: train {len(train_records)} | eval {len(eval_records)} | "
          f"fresh {len(fresh_records)}")
    if args.limit:
        train_records = train_records[:args.limit]
        fresh_records = fresh_records[:args.limit]
        eval_records = eval_records[:args.limit]
        print(f"limit applied: {args.limit} puzzles per pool")

    def log(message):
        print(message, flush=True)

    train_examples, train_report = build_pool(
        train_records, max_deductions_per_puzzle=args.max_deductions,
        branch_repeat=args.branch_repeat, max_actions=args.max_actions, log=log)
    dev_examples, dev_report = build_pool(
        fresh_records, max_deductions_per_puzzle=args.dev_deductions,
        branch_repeat=1, max_actions=args.max_actions, log=log)

    train_path = args.output_dir / "train-examples.jsonl"
    dev_path = args.output_dir / "dev-examples.jsonl"
    write_jsonl(train_path, train_examples)
    write_jsonl(dev_path, dev_examples)

    report = {
        "data": str(args.data),
        "max_actions": args.max_actions,
        "max_deductions_per_puzzle": args.max_deductions,
        "branch_repeat": args.branch_repeat,
        "split": {
            "train": len(train_records),
            "eval": len(eval_records),
            "fresh": len(fresh_records),
            "eval_ids": [record["puzzle_id"] for record in eval_records],
            "fresh_ids": [record["puzzle_id"] for record in fresh_records],
            "matches_run04": not args.limit,
        },
        "examples": {"train": len(train_examples), "dev": len(dev_examples)},
        "estimated_tokens": {
            "train": estimate_tokens(train_examples),
            "dev": estimate_tokens(dev_examples),
        },
        "train_report": train_report,
        "dev_report": dev_report,
        "files": {"train": str(train_path), "dev": str(dev_path)},
    }
    report_path = args.output_dir / "generation-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    print(f"\ntrain examples: {len(train_examples):,} "
          f"(~{report['estimated_tokens']['train'] / 1e6:.1f}M tokens)")
    print(f"dev examples:   {len(dev_examples):,}")
    print(f"branch examples: {train_report['branch_examples']:,} | "
          f"deduce: {train_report['deduce_examples']:,}")
    print(f"oracle actions: {train_report['oracle_actions']:,}")
    print(f"wrote {train_path}")
    print(f"wrote {dev_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
