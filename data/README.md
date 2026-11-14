---
pretty_name: Train Tracks Solving Traces
language:
- en
tags:
- train-tracks
- puzzle-solving
- process-supervision
- backtracking
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
---

# Train Tracks solving traces

Action-level traces for training a model to choose the next justified action
on a partial Train Tracks puzzle. The train split contains **24,565 actions
from 212 training puzzles**, with one trace per puzzle. There are no validation
or test puzzles in this dataset.

| Action | Rows |
| --- | ---: |
| Deduce | 23,225 |
| Branch | 794 |
| Contradiction | 167 |
| Undo | 167 |
| Finish | 212 |

## Data format

`data/train.jsonl` has one JSON object per action. Each object contains
`puzzle_id`, zero-based `step_index`, `state`, `action`, and
`expert_policy_label`. `state` describes the board before the action:
`row_clues`, `column_clues`, `givens`, `domains`, and `branch_stack`.
Coordinates in givens and actions are one-based.

Each domain is an integer bitmask of possible tiles. Bits 0 through 6 mean
`XX` (empty), `NS`, `EW`, `NE`, `NW`, `SE`, and `SW`. For example, `127` allows
all seven tiles and `33` allows `XX` or `SE`.

`action.type` is `deduce`, `branch`, `contradiction`, `undo`, or `finish`.
Deduction actions use a rule and anchor to narrow one cell's candidates.
The supported deduction rules are `clue_satisfied`,
`all_remaining_cells_required`, and `port_reciprocity`.

**Wrong trial branches have `expert_policy_label: false`.** Exclude those
rows from positive action-imitation targets. Subsequent deductions,
contradictions, and undos on those partial boards remain valid actions.

## Use and limits

```python
from datasets import load_dataset

traces = load_dataset("zuhairsan/train-tracks-solving-traces", split="train")
```

The successful branch in each trace was selected offline using an audited
training solution. These traces do not show a model solving puzzles on its
own. Evaluate whole-puzzle solving on separate held-out puzzles, not on this
training split.

The source puzzles were derived from PuzzleMadness Train Tracks puzzles from
January through October 2026. This dataset contains derived clues, given
pieces, and action traces, without source puzzle images. It does not assert
ownership of the original puzzles or grant a license over their content.
