"""Generate held-out medium puzzle traces without modifying train.jsonl."""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from build_solving_traces import build_trace

ROOT = Path(__file__).resolve().parents[1]


def medium_records(path):
    selected = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            puzzle_id = record["puzzle_id"]
            if record["split"] != "test_medium" or puzzle_id in selected:
                raise ValueError(f"Invalid medium record: {puzzle_id}")
            selected[puzzle_id] = record
    if len(selected) != 304:
        raise ValueError(f"Expected 304 medium puzzles, found {len(selected)}")
    return sorted(selected.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "output/medium/source.jsonl")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "output/medium/test_medium.jsonl")
    parser.add_argument("--limit", type=int, default=0,
                        help="First N medium puzzles for a quick check; 0 means all")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    records = medium_records(args.source)
    if args.limit:
        records = records[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    counts = Counter()
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for index, (puzzle_id, record) in enumerate(records, 1):
                rows, stats = build_trace(puzzle_id, record)
                for row in rows:
                    stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                    counts[row["action"]["type"]] += 1
                counts["wrong_branch_episodes"] += stats["wrong_branch_probed"]
                if index % 10 == 0 or index == len(records):
                    print(f"{index}/{len(records)} medium puzzles", flush=True)
        os.replace(temporary, args.output)
    finally:
        if temporary.exists():
            temporary.unlink()
    try:
        output_label = args.output.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        output_label = str(args.output)
    report = {
        "source_split": "test_medium", "training_eligible": False,
        "puzzles": len(records), "actions": {
            key: counts[key] for key in
            ("deduce", "branch", "contradiction", "undo", "finish")},
        "wrong_branch_episodes": counts["wrong_branch_episodes"],
        "output": output_label,
    }
    summary = args.output.with_suffix(".summary.json")
    summary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
