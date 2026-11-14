# Offline solving traces: format and verification

The [project plan](LEARNING_TO_SOLVE.md) explains how these traces fit into
training an action policy. The [README](../README.md) is the dataset card.

## Reproduce

Run from the root of this repository with Python 3.11 or newer:

```bash
python scripts/build_solving_traces.py
python scripts/check_solving_traces.py
python scripts/render_solving_traces.py
```

`source/train-puzzles.jsonl` contains the training-only clues, givens, and
audited solutions needed by these scripts. To recreate that source file from
a sibling checkout of the original project:

```bash
python scripts/extract_train_source.py --source-root ../Train_Track_Puzzle
```

The generator writes `data/train.jsonl` and `data/train.summary.json`. The
renderer writes the standalone [solving-traces.html](../solving-traces.html).
The replay checker independently walks the JSONL actions and verifies every
deduction, branch, contradiction, undo, and solved board.

## Contents

There are **212 training puzzles** and **24,565 actions**:

| Action | Count | Meaning |
| --- | ---: | --- |
| `deduce` | 23,225 | Remove candidate tiles ruled out by a visible constraint |
| `branch` | 794 | Try one tile at an ambiguous cell |
| `contradiction` | 167 | Detect impossible clue bounds, ports, or a disconnected completed path |
| `undo` | 167 | Restore the board before the wrong branch |
| `finish` | 212 | Complete a verified single-track solution |

One wrong-branch episode appears in 167 puzzles. The other 45 puzzles still
have successful traces; the generator simply found no wrong branch that
reached a contradiction under its local propagation limit. Each wrong branch
is followed by a genuine checked contradiction and an undo.

## JSONL row

```json
{
  "puzzle_id": "01042026-small",
  "step_index": 0,
  "state": {
    "row_clues": [3, 4, 4, 3, 1, 3, 1],
    "column_clues": [4, 4, 3, 4, 3, 1],
    "givens": [
      {"row": 2, "column": 6, "value": "EW"},
      {"row": 4, "column": 1, "value": "NS"},
      {"row": 7, "column": 3, "value": "NS"}
    ],
    "domains": [
      [33, 101, 101, 101, 101, 65],
      [43, 127, 127, 127, 127, 4],
      [43, 127, 127, 127, 127, 83],
      [2, 127, 127, 127, 127, 83],
      [43, 127, 127, 127, 127, 83],
      [43, 127, 127, 127, 127, 83],
      [9, 29, 2, 29, 29, 17]
    ],
    "branch_stack": []
  },
  "action": {
    "type": "deduce",
    "rule": "clue_satisfied",
    "anchor": {"row": 7, "clue": 1},
    "cell": {"row": 7, "column": 1},
    "candidates": ["XX"]
  },
  "expert_policy_label": true
}
```

This is an actual dataset row: row 7's clue is 1, and its given track at
column 3 already satisfies it, so row 7 column 1 must be empty. Coordinates
are one-based, while `step_index` is zero-based.

`state.domains` is a grid of integer bitmasks. The bit order is:

| Bit | Candidate |
| ---: | --- |
| 0 | `XX` (empty) |
| 1 | `NS` |
| 2 | `EW` |
| 3 | `NE` |
| 4 | `NW` |
| 5 | `SE` |
| 6 | `SW` |

For example, mask `127` is all seven choices (`?7` in the viewer). Mask `33`
is `XX` or `SE` (`?2`). The candidate count means *not yet ruled out by the
recorded constraints*; it does not mean every candidate completes the puzzle.

`deduce` actions replace one cell's domain with the listed `candidates`.
Supported deduction rules are `clue_satisfied`,
`all_remaining_cells_required`, and `port_reciprocity`. `branch` sets one
candidate; the active choices appear in `branch_stack`. `undo.to_step` points
to the zero-based branch action to restore. A `contradiction` reports the
offending rule and anchor without changing the board.

`expert_policy_label: false` occurs only on a deliberately wrong trial
branch. A future positive action policy should exclude those branch rows.
The deductions made while exploring a wrong branch remain sound for that
partial state, and the later contradiction and undo are valid actions.

## Scope

The published `data/train.jsonl` contains only the original project's
**train** puzzle IDs. The successful
branch tile is selected using an audited training solution; the input state
contains only the current clues, givens, candidate domains, and branch stack.
Optional held-out medium traces are generated in `data/train_medium.jsonl`,
with source solutions in `source/train-puzzles-medium.jsonl`. They remain separate from training and excluded by the Hugging Face
uploader. They are for inspection and replay verification only.
This is process-supervision data, not an evaluation of a trained solver.
