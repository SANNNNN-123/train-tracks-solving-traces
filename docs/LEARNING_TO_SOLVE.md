# Learning to solve Train Tracks puzzles

## Goal

Train a model to choose **one justified action at a time** on a partial Train
Tracks board, then measure whether its own sequence of actions solves a new
puzzle. The board has row and column track counts, given rail pieces, two
boundary endpoints, and a single connected nonbranching path.

This repository is the **offline training-trace dataset** for that goal. The
trace format and commands are in [SOLVING_TRACES.md](SOLVING_TRACES.md); the
[HTML viewer](../solving-traces.html) shows every training puzzle step by step.

## Why this dataset exists

The original `Train_Track_Puzzle` project tried three related approaches:

| Experiment | Evaluation result | What it measured |
| --- | --- | --- |
| LiquidAI/d1-2B counts and ports decision model | 4/46 validation puzzles solved at best | Final tile choices during a model-owned whole-puzzle rollout |
| Qwen3-VL-2B anchored deduction SFT | 91/241 exact one-step validation actions (37.8%) | Apply a deduction **when the rule and anchor are supplied** |
| Qwen3-VL-2B deduction-plus-tile SFT | 0/46 puzzles solved; 1,182/2,472 cells correct (47.8%) | Final tile choices during a model-owned whole-puzzle rollout |

In the Qwen whole-puzzle rollout, the first wrong tile came by the **5th
decision (median)**; 37 of 46 puzzles had an error in the first 10 decisions.
Later prompts then contained the model's earlier mistakes. The tile training
examples had correct earlier moves, and the deduction task supplied the rule
and anchor. A low teacher-forced loss did not show that the model could choose
rules, recover from errors, or complete a puzzle.

## What is complete

1. **Offline traces:** 24,565 actions from 212 training puzzles. They include
   count and port deductions, successful branches, 167 wrong-branch episodes
   with a checkable contradiction and undo, and a final solved board per puzzle.
2. **Replay verification:** `python scripts/check_solving_traces.py` verifies
   every action and the final board. `source/train-puzzles.jsonl` contains only
   training puzzles, so the dataset is self-contained without held-out
   solutions or puzzle images.
3. **Review tool:** `solving-traces.html` lets a person inspect the evolving
   candidate domains, rails, branches, contradictions, and undos.

The audited training solution guides a successful branch **offline**. Wrong
trial branches have `expert_policy_label: false`; they are examples of
recoverable failures, not positive branch labels.

## Next work

1. **Measure the one-step baseline.** On the original project's frozen
   validation puzzles, measure final-tile accuracy with correct histories and
   classify the first error in model-owned rollouts. This separates basic
   tile-choice errors from error accumulation.
2. **Prepare an action-policy training view.** Render each candidate domain as
   readable tile choices and train on structured `deduce`, successful
   `branch`, `contradiction`, `undo`, and `finish` actions. Exclude the 167
   `expert_policy_label: false` branch rows from positive imitation targets.
   Balance action types so the 23,225 deduction rows do not overwhelm the
   recovery examples.
3. **Build an interactive evaluator.** The model chooses a rule, cell, and
   action. Code checks it against visible count, port, and path constraints,
   applies legal actions, and reports contradictions. The checker must not
   choose the tile or call the offline solution at evaluation time.
4. **Train on model-created states.** Collect partial boards from the model's
   own rollouts, label recoverable next actions offline, and retrain. This
   addresses the gap between correct teacher histories and the states reached
   by the model itself ([DAgger](https://users.cs.utah.edu/~dsbrown/readings/dagger.pdf)).

Report action validity, forced-action accuracy, contradiction and undo
accuracy, the first wrong step, and **whole puzzles solved**. Keep the
original project's held-out test splits untouched while developing on its
frozen validation split. Step-level feedback is motivated by
[process-supervision research](https://arxiv.org/abs/2305.20050), but results
from other reasoning tasks do not guarantee puzzle-solving gains here.
