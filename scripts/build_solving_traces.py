"""Generate checkable Train Tracks action traces from training puzzles only.

Run from the repository root:
    python scripts/dataset/build_solving_traces.py

The audited solution guides successful branch labels offline. Counts and port
rules alone justify every deduction. A wrong branch is recorded only when
local propagation actually reaches a contradiction; it is not a policy label.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_logic import (BIT, TILES, Step, apply_deduction, branch_cell,
                         choices, clue_from_record, completed_error,
                         initial_domains, next_step, snapshot)

DEFAULT_OUTPUT = ROOT / "data/train.jsonl"


def train_records() -> list[tuple[str, dict]]:
    selected = {}
    with (ROOT / "source/train-puzzles.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            puzzle_id = record["puzzle_id"]
            if record["split"] != "train" or puzzle_id in selected:
                raise ValueError(f"Invalid training record: {puzzle_id}")
            selected[puzzle_id] = record
    if len(selected) != 212:
        raise ValueError(f"Expected 212 training puzzles, found {len(selected)}")
    return sorted(selected.items())


def solution_token(record: dict, cell: tuple[int, int]) -> str:
    r, c = cell
    return record["solution"][r][c] or "XX"


def probe_wrong(board, clue, cell, value, max_steps=1000):
    """Return visible propagation steps ending in a contradiction, if found."""
    trial = snapshot(board)
    r, c = cell
    trial[r][c] = BIT[value]
    events = []
    for _ in range(max_steps):
        step = next_step(trial, clue)
        if step is None:
            return None
        events.append(step)
        if step.kind == "contradiction":
            return events
        apply_deduction(trial, step)
    return None


def build_trace(puzzle_id: str, record: dict) -> tuple[list[dict], dict]:
    clue = clue_from_record(record)
    board = initial_domains(clue)
    height, width = len(board), len(board[0])
    if len(record["solution"]) != height or any(
            len(line) != width for line in record["solution"]):
        raise ValueError(f"Wrong solution dimensions: {puzzle_id}")
    for r, line in enumerate(board):
        for c, mask in enumerate(line):
            if not mask & BIT[solution_token(record, (r, c))]:
                raise ValueError(f"Solution excluded initially: {puzzle_id} {(r, c)}")

    rows, stack = [], []
    probed = False

    def add(action: dict, *, expert=True):
        rows.append({
            "puzzle_id": puzzle_id,
            "step_index": len(rows),
            "state": {
                "row_clues": clue["rows"], "column_clues": clue["columns"],
                "givens": [{"row": r + 1, "column": c + 1, "value": shape}
                           for r, c, shape in clue["given_shapes"]],
                "domains": snapshot(board),
                "branch_stack": [dict(item) for item in stack],
            },
            "action": action,
            "expert_policy_label": expert,
        })

    for _ in range(height * width * 30):
        step = next_step(board, clue)
        if step:
            if step.kind != "deduce":
                raise ValueError(f"Contradiction on correct path: {puzzle_id}, {step}")
            add(step.action())
            apply_deduction(board, step)
            if step.cell and not board[step.cell[0]][step.cell[1]] & BIT[solution_token(record, step.cell)]:
                raise ValueError(f"Deduction excluded solution: {puzzle_id}, {step}")
            continue
        cell = branch_cell(board)
        if cell is None:
            if completed_error(board, clue):
                raise ValueError(f"Incomplete solution path: {puzzle_id}")
            if any(choices(board[r][c]) != [solution_token(record, (r, c))]
                   for r in range(height) for c in range(width)):
                raise ValueError(f"Teacher path differs from audited solution: {puzzle_id}")
            add({"type": "finish"})
            return rows, {"puzzle_id": puzzle_id,
                          "steps": len(rows),
                          "wrong_branch_probed": probed,
                          "successful_branches": sum(row["action"]["type"] == "branch"
                                                     and row["expert_policy_label"] for row in rows)}

        r, c = cell
        correct = solution_token(record, cell)
        if not board[r][c] & BIT[correct]:
            raise ValueError(f"Teacher branch excluded: {puzzle_id}, {cell}")
        # One genuine wrong-turn episode per puzzle keeps traces useful and small.
        if not probed:
            for wrong in choices(board[r][c]):
                if wrong == correct:
                    continue
                probe = probe_wrong(board, clue, cell, wrong)
                if probe is None:
                    continue
                checkpoint = snapshot(board)
                branch_index = len(rows)
                action = {"type": "branch", "cell": {"row": r + 1, "column": c + 1},
                          "value": wrong}
                add(action, expert=False)
                board[r][c] = BIT[wrong]
                stack.append({"to_step": branch_index, "cell": action["cell"],
                              "value": wrong})
                for event in probe:
                    add(event.action())
                    if event.kind == "deduce":
                        apply_deduction(board, event)
                add({"type": "undo", "to_step": branch_index})
                board = checkpoint
                stack.pop()
                probed = True
                break

        branch_index = len(rows)
        action = {"type": "branch", "cell": {"row": r + 1, "column": c + 1},
                  "value": correct}
        add(action)
        board[r][c] = BIT[correct]
        stack.append({"to_step": branch_index, "cell": action["cell"],
                      "value": correct})
    raise RuntimeError(f"Trace step limit exceeded: {puzzle_id}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0,
                        help="Generate the first N training puzzles; 0 means all")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    records = train_records()
    if args.limit:
        records = records[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = Counter()
    with args.output.open("w", encoding="utf-8") as stream:
        for index, (puzzle_id, record) in enumerate(records, 1):
            trace, stats = build_trace(puzzle_id, record)
            for row in trace:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                summary[row["action"]["type"]] += 1
            summary["puzzles"] += 1
            summary["wrong_branch_episodes"] += stats["wrong_branch_probed"]
            if index % 25 == 0 or index == len(records):
                print(f"{index}/{len(records)} training puzzles", flush=True)
    try:
        output_label = args.output.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        output_label = str(args.output)
    report = {"source_split": "train", "puzzles": summary.pop("puzzles"),
              "actions": dict(summary), "output": output_label}
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
