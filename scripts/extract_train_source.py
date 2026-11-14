"""Export only frozen training puzzles from the upstream audit records.

Run from this repository root:
    python scripts/extract_train_source.py --source-root ../Train_Track_Puzzle
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
                        default=ROOT.parent / "Train_Track_Puzzle")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "source/train-puzzles.jsonl")
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    with (source_root / "dataset-audit/puzzle-splits.csv").open(
            newline="", encoding="utf-8") as stream:
        train_ids = {row["puzzle_id"] for row in csv.DictReader(stream)
                     if row["split"] == "train"}
    records = {}
    with (source_root / "dataset-audit/validated-puzzles.jsonl").open(
            encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            puzzle_id = Path(record["path"]).stem
            if puzzle_id not in train_ids:
                continue
            if puzzle_id in records or record["status"] != "validated":
                raise ValueError(f"Duplicate or invalid training record: {puzzle_id}")
            records[puzzle_id] = {
                "puzzle_id": puzzle_id,
                "split": "train",
                "source_path": record["path"],
                "puzzle": record["puzzle"],
                "solution": record["solution"],
            }
    if set(records) != train_ids or len(records) != 212:
        raise ValueError("The exported training IDs differ from the frozen split")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for puzzle_id in sorted(records):
            stream.write(json.dumps(records[puzzle_id], separators=(",", ":")) + "\n")
    print(f"Exported {len(records)} training-only source puzzles to {args.output}")


if __name__ == "__main__":
    main()
