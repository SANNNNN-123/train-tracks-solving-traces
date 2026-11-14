# Trace action policy on Modal

Run these commands **from the `Train_Track_Solving_Traces` repository root**.
The scripts use only files in this repository; no path to the original project
is needed. They do not start automatically.

```bash
python -m unittest discover -s modal -p 'test_*.py'
uv run modal run --detach modal/train.py --mode full
uv run modal run --detach modal/evaluate.py --run-id RUN_ID --limit-puzzles 8
```

For a medium-puzzle pilot, evaluate 32 puzzles, balanced across the four board
shapes (8 each):

```bash
uv run modal run --detach modal/evaluate.py --run-id RUN_ID --dataset medium --limit-puzzles 32
```

Use the `run_id` printed by training. For medium evaluation,
`--limit-puzzles 0` evaluates all 304 puzzles; a positive limit selects a
deterministic sample balanced across board shapes. For small evaluation,
`--limit-puzzles 0` evaluates all 32 internal development puzzles.
`--max-actions 256` caps a rollout. Results and the adapter are stored in the
Modal volume `train-tracks-solving-traces`.
The scripts use the existing Modal secret `hf-train-tracks` for `HF_TOKEN`.

`train.py` fits a LoRA adapter on Qwen2.5-1.5B-Instruct. It uses a deterministic
180 puzzle training / 32 puzzle development split **within the 212 training
puzzles**. The prompt spells out each cell's candidate tile names and the model
learns to select a choice ID from the rule checker's legal-action menu. The
selected menu entry supplies the actual action to the evaluator. Wrong trial
branches (`expert_policy_label: false`) are excluded from positive targets.
Every puzzle's starting action is kept, and up to 48 deduction examples per
puzzle are sampled round-robin across deduction rules so port deductions cannot
dominate the retained sample.

`evaluate.py` starts each held-out puzzle from clues and givens. The checker
enumerates justified deductions, contradictions, undo, finish, and fallback
branch choices; the model selects a menu ID at each step. This constrains the
model to supported actions while leaving it responsible for choosing a useful
move. The stored source solution is used **only after the rollout** to score
the final board. The JSON output records the board, menu size, and each chosen
action. This is an internal development benchmark, not the original project's
frozen 46 puzzle validation set. A later external test should import the frozen
puzzle inputs without including their answers in training or policy prompts.

`--dataset medium` uses puzzles marked `test_medium` in
`source/train-puzzles-medium.jsonl`. The 32-puzzle pilot is for an initial
estimate; keep the other 272 for a later full evaluation if you plan to tune
the model based on pilot results. Clues and givens initialize each rollout;
the stored solution is consulted only after the rollout for scoring. Do not
train on `data/train_medium.jsonl` when reporting this held-out test.
