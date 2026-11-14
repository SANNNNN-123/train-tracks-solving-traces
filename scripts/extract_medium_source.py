"""Export the frozen medium test puzzles for local trace generation.

The output is held-out test material. Never add it to the training JSONL.
"""

import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
                        default=ROOT.parent / "Train_Track_Puzzle")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "output/medium/source.jsonl")
    args = parser.parse_args()
    old_root = args.source_root.resolve()
    with (old_root / "dataset-audit/puzzle-splits.csv").open(
            newline="", encoding="utf-8") as stream:
        medium_ids = {row["puzzle_id"] for row in csv.DictReader(stream)
                      if row["split"] == "test_medium"}
    if len(medium_ids) != 304:
        raise ValueError(f"Expected 304 frozen medium puzzles, found {len(medium_ids)}")
    records = {}
    with (old_root / "dataset-audit/validated-puzzles.jsonl").open(
            encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            puzzle_id = Path(record["path"]).stem
            if puzzle_id not in medium_ids:
                continue
            if puzzle_id in records or record["status"] != "validated":
                raise ValueError(f"Duplicate or invalid medium puzzle: {puzzle_id}")
            records[puzzle_id] = {
                "puzzle_id": puzzle_id, "split": "test_medium",
                "source_path": record["path"],
                "puzzle": record["puzzle"], "solution": record["solution"],
            }
    if set(records) != medium_ids:
        raise ValueError("Medium records differ from the frozen split")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for puzzle_id in sorted(records):
            stream.write(json.dumps(records[puzzle_id], separators=(",", ":")) + "\n")
    os.replace(temporary, args.output)
    print(f"Exported {len(records)} held-out medium puzzles to {args.output}")


if __name__ == "__main__":
    main()
