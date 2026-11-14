"""Replay and verify generated Train Tracks action traces.

Run: python scripts/dataset/check_solving_traces.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_solving_traces import DEFAULT_OUTPUT, train_records
from trace_logic import (BIT, apply_deduction, clue_from_record, completed_error,
                         initial_domains, next_step, snapshot)


def check_trace(puzzle_id: str, rows: list[dict], record: dict) -> Counter:
    clue = clue_from_record(record)
    board = initial_domains(clue)
    stack = []
    pending_conflict = False
    counts = Counter()
    finished = False
    for index, row in enumerate(rows):
        if row["puzzle_id"] != puzzle_id or row["step_index"] != index:
            raise ValueError(f"Bad row identity: {puzzle_id}, {index}")
        state = row["state"]
        givens = [{"row": r + 1, "column": c + 1, "value": shape}
                  for r, c, shape in clue["given_shapes"]]
        if (state["row_clues"] != clue["rows"] or
                state["column_clues"] != clue["columns"] or
                state["givens"] != givens or state["domains"] != board or
                state["branch_stack"] != [item[0] for item in stack]):
            raise ValueError(f"State does not match replay: {puzzle_id}, {index}")
        action = row["action"]
        kind = action["type"]
        counts[kind] += 1
        if pending_conflict and kind != "undo":
            raise ValueError(f"Contradiction must be followed by undo: {puzzle_id}")
        if kind == "deduce":
            step = next_step(board, clue)
            if step is None or step.action() != action:
                raise ValueError(f"Unjustified deduction: {puzzle_id}, {index}")
            apply_deduction(board, step)
        elif kind == "branch":
            if next_step(board, clue) is not None:
                raise ValueError(f"Branch taken before propagation: {puzzle_id}, {index}")
            r, c = action["cell"]["row"] - 1, action["cell"]["column"] - 1
            value = action["value"]
            if not (0 <= r < len(board) and 0 <= c < len(board[0])
                    and value in BIT and board[r][c].bit_count() > 1
                    and board[r][c] & BIT[value]):
                raise ValueError(f"Illegal branch: {puzzle_id}, {index}")
            if row["expert_policy_label"] != (
                    value == (record["solution"][r][c] or "XX")):
                raise ValueError(f"Wrong policy label: {puzzle_id}, {index}")
            checkpoint = snapshot(board)
            stack_item = {"to_step": index, "cell": action["cell"], "value": value}
            board[r][c] = BIT[value]
            stack.append((stack_item, checkpoint))
        elif kind == "contradiction":
            step = next_step(board, clue)
            if step is None or step.action() != action:
                raise ValueError(f"Unjustified contradiction: {puzzle_id}, {index}")
            pending_conflict = True
        elif kind == "undo":
            if not pending_conflict or not stack:
                raise ValueError(f"Unjustified undo: {puzzle_id}, {index}")
            branch, board = stack.pop()
            if action["to_step"] != branch["to_step"]:
                raise ValueError(f"Undo targets wrong branch: {puzzle_id}, {index}")
            pending_conflict = False
        elif kind == "finish":
            if index != len(rows) - 1 or pending_conflict or completed_error(board, clue):
                raise ValueError(f"Premature finish: {puzzle_id}, {index}")
            expected = [[BIT[value or "XX"] for value in line]
                        for line in record["solution"]]
            if board != expected:
                raise ValueError(f"Finished board differs from audit: {puzzle_id}")
            finished = True
        else:
            raise ValueError(f"Unknown action: {puzzle_id}, {kind}")
    if not finished:
        raise ValueError(f"Missing finish: {puzzle_id}")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-subset", action="store_true")
    args = parser.parse_args()
    records = dict(train_records())
    traces = {}
    with args.path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            puzzle_id = row["puzzle_id"]
            if puzzle_id not in records:
                raise ValueError(f"Nontraining puzzle in trace: {puzzle_id}")
            traces.setdefault(puzzle_id, []).append(row)
    if not args.allow_subset and set(traces) != set(records):
        raise ValueError("Trace file does not contain exactly the training puzzles")
    total = Counter()
    for puzzle_id, rows in traces.items():
        total.update(check_trace(puzzle_id, rows, records[puzzle_id]))
    print(json.dumps({"verified_training_puzzles": len(traces),
                      "actions": dict(total)}, indent=2))


if __name__ == "__main__":
    main()
