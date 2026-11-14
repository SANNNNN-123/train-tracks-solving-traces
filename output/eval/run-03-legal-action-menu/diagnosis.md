# Run 03 evaluation diagnosis

## Results

- 31 of 32 internal development puzzles solved (96.9%). All 31 solved boards
  passed the path and clue checks.
- 1,733 of 1,762 cells match the reference (98.4%).
- The remaining puzzle, `09012026-small`, reached the evaluator's 256-action
  cap. All 256 chosen actions were accepted by the checker.

## Why `09012026-small` hit the cap

- The rollout made 241 deductions, 9 branches, 3 contradictions, and 3 undos.
- Six branch guesses were still active when the cap stopped the rollout.
- The prediction at the cap had 34 correct singleton tiles, 14 incorrect
  singleton tiles, and 15 unresolved cells.
- The recorded teacher trace solves this puzzle in 160 actions, with 150
  deductions, 7 branches, 1 contradiction, and 1 undo.

This is a search-path problem, not an invalid-action problem. The menu keeps
every move legal, but the model explored a deeper and less useful branch path
than the recorded teacher trace. The action cap is above the teacher trace
length, so raising it alone may not solve the puzzle reliably.

## Next experiment

Run the evaluator on this puzzle alone with a higher cap and inspect its branch
choices. If it keeps nesting guesses, add a branch-depth limit or branch
heuristic and train on successful and failed model-generated branch states.

The complete board comparison is in `predicted-vs-actual.html`; the full action
history is in `rollout.json`.
