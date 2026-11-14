"""Replay the held-out medium traces against their audited solutions."""

import argparse
import json
from collections import Counter
from pathlib import Path

from build_medium_traces import medium_records
from check_solving_traces import check_trace

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "output/medium/source.jsonl")
    parser.add_argument("--input", type=Path,
                        default=ROOT / "output/medium/test_medium.jsonl")
    parser.add_argument("--allow-subset", action="store_true")
    args = parser.parse_args()
    records = dict(medium_records(args.source))
    total, seen = Counter(), set()
    current_id, current_rows = None, []

    def verify():
        if current_id is None:
            return
        if current_id not in records or current_id in seen:
            raise ValueError(f"Unknown or repeated medium puzzle: {current_id}")
        total.update(check_trace(current_id, current_rows, records[current_id]))
        seen.add(current_id)
        if len(seen) % 10 == 0:
            print(f"verified {len(seen)} medium puzzles", flush=True)

    with args.input.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            puzzle_id = row["puzzle_id"]
            if current_id is not None and puzzle_id != current_id:
                verify()
                current_rows = []
            current_id = puzzle_id
            current_rows.append(row)
    verify()
    if not args.allow_subset and seen != set(records):
        raise ValueError("Trace file does not contain exactly the frozen medium puzzles")
    print(json.dumps({"verified_medium_puzzles": len(seen),
                      "actions": dict(total)}, indent=2))


if __name__ == "__main__":
    main()
