"""Build the Kaggle training views, including branch-focused augmentation.

Two changes relative to the original Modal baseline:

1. **Every branch decision is trained in both menu orders.**  The evaluator
   sorts the legal menu, so a model can learn "pick a001" instead of reading the
   board.  Reversing the two branch entries removes that shortcut.
2. **Wrong-branch trace rows are reused as corrected positives.**  The original
   `split_examples` drops rows with ``expert_policy_label: false``.  Their state
   is still a valid pre-branch state, and the audited solution tells us the
   correct value.  Adding that corrected example gives the model the hard states
   where the teacher probe guessed wrong, without ever training a wrong target.

Branch examples are optionally repeated so they are not drowned by the ~8.6k
deduction examples.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from action_policy import (choice_id, completion, development_ids,
                           legal_actions, prompt, split_examples)

HERE = Path(__file__).resolve().parent
TILES = ("XX", "NS", "EW", "NE", "NW", "SE", "SW")


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def load_solutions(path: Path) -> dict[str, list[list[str]]]:
    solutions = {}
    for record in load_jsonl(path):
        solutions[record["puzzle_id"]] = [
            [tile or "XX" for tile in row] for row in record["solution"]
        ]
    return solutions


def _make_example(row, actions, target):
    return {
        "user": prompt(row["state"], actions),
        "completion": completion(target),
        "kind": row["action"]["type"],
        "puzzle_id": row["puzzle_id"],
        "target": target,
        "menu_size": len(actions),
    }


def _branch_pair(mask: int) -> str:
    return "|".join(name for index, name in enumerate(TILES) if mask & (1 << index))


def _corrected_branch_row(row, solution):
    cell = row["action"]["cell"]
    correct = solution[cell["row"] - 1][cell["column"] - 1]
    actions = legal_actions(row["state"], pending=False)
    branch_actions = [a for a in actions if a["type"] == "branch"]
    if not any(a["cell"] == cell and a["value"] == correct for a in branch_actions):
        return None, None
    corrected = {"type": "branch", "cell": cell, "value": correct}
    corrected_row = dict(row)
    corrected_row["action"] = corrected
    return corrected_row, actions


def build_examples(rows, solutions=None, *, branch_repeat: int = 2,
                   reverse_branch_orders: bool = True,
                   corrected_wrong_branches: bool = True,
                   max_deductions_per_puzzle: int = 48,
                   seed: int = 42):
    """Return ``(examples, stats)`` for a list of trace rows."""
    selected = split_examples(rows, max_deductions_per_puzzle)
    examples = []
    by_kind = Counter()
    pairs = Counter()
    values = Counter()
    corrected = 0
    reversed_orders = 0

    def add(row, actions, target):
        examples.append(_make_example(row, actions, target))

    for row in selected:
        pending = row["action"]["type"] == "undo"
        actions = legal_actions(row["state"], pending=pending)
        target = choice_id(row["action"], actions)
        add(row, actions, target)
        by_kind[row["action"]["type"]] += 1
        if row["action"]["type"] == "branch":
            cell = row["action"]["cell"]
            pairs[_branch_pair(row["state"]["domains"][cell["row"] - 1][cell["column"] - 1])] += 1
            values[row["action"]["value"]] += 1
            if reverse_branch_orders:
                rev = list(reversed(actions))
                add(row, rev, choice_id(row["action"], rev))
                reversed_orders += 1

    if corrected_wrong_branches and solutions is not None:
        for row in rows:
            if row["action"]["type"] != "branch" or row.get("expert_policy_label", True):
                continue
            solution = solutions.get(row["puzzle_id"])
            if solution is None:
                continue
            corrected_row, actions = _corrected_branch_row(row, solution)
            if corrected_row is None:
                continue
            target = choice_id(corrected_row["action"], actions)
            add(corrected_row, actions, target)
            by_kind["branch_corrected"] += 1
            cell = corrected_row["action"]["cell"]
            pairs[_branch_pair(corrected_row["state"]["domains"][cell["row"] - 1][cell["column"] - 1])] += 1
            values[corrected_row["action"]["value"]] += 1
            corrected += 1
            if reverse_branch_orders:
                rev = list(reversed(actions))
                add(corrected_row, rev, choice_id(corrected_row["action"], rev))
                reversed_orders += 1

    if branch_repeat > 1:
        branch_examples = [
            e for e in examples if e["kind"] in {"branch", "branch_corrected"}
        ]
        for _ in range(branch_repeat - 1):
            examples.extend(branch_examples)

    rng = random.Random(seed)
    rng.shuffle(examples)
    stats = {
        "total": len(examples),
        "base_selected": len(selected),
        "by_kind": dict(by_kind),
        "corrected_wrong_branches": corrected,
        "reversed_branch_orders": reversed_orders,
        "branch_repeat": branch_repeat,
        "branch_examples_after_repeat": sum(
            1 for e in examples if e["kind"] in {"branch", "branch_corrected"}),
        "branch_pair_counts": dict(pairs),
        "branch_value_counts": dict(values),
    }
    return examples, stats


def build_dev_examples(rows, max_deductions_per_puzzle: int = 48):
    """Clean dev view: base examples only, no augmentation."""
    selected = split_examples(rows, max_deductions_per_puzzle)
    examples = []
    for row in selected:
        actions = legal_actions(row["state"], pending=row["action"]["type"] == "undo")
        target = choice_id(row["action"], actions)
        examples.append(_make_example(row, actions, target))
    return examples


def split_rows(rows):
    dev_ids = development_ids(row["puzzle_id"] for row in rows)
    train_rows = [row for row in rows if row["puzzle_id"] not in dev_ids]
    dev_rows = [row for row in rows if row["puzzle_id"] in dev_ids]
    return train_rows, dev_rows, len(dev_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "data/train.jsonl")
    parser.add_argument("--source", type=Path,
                        default=HERE / "data/train-puzzles-small.jsonl")
    parser.add_argument("--branch-repeat", type=int, default=2)
    parser.add_argument("--no-branch-augment", action="store_true")
    parser.add_argument("--write", type=Path, default=None,
                        help="Optional JSONL output with user/completion fields")
    args = parser.parse_args()

    rows = load_jsonl(args.data)
    solutions = load_solutions(args.source)
    train_rows, dev_rows, dev_count = split_rows(rows)
    if args.no_branch_augment:
        examples, stats = build_examples(
            train_rows, solutions, branch_repeat=1,
            reverse_branch_orders=False, corrected_wrong_branches=False)
    else:
        examples, stats = build_examples(
            train_rows, solutions, branch_repeat=args.branch_repeat)
    dev = build_dev_examples(dev_rows)
    stats["puzzles_total"] = len({r["puzzle_id"] for r in rows})
    stats["puzzles_train"] = len({r["puzzle_id"] for r in train_rows})
    stats["puzzles_dev"] = dev_count
    stats["dev_examples"] = len(dev)
    print(json.dumps(stats, indent=2))
    if args.write:
        with args.write.open("w", encoding="utf-8", newline="\n") as handle:
            for example in examples:
                handle.write(json.dumps(example, ensure_ascii=False) + "\n")
        print(f"wrote {len(examples)} examples to {args.write}")


if __name__ == "__main__":
    main()
