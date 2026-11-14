# Run 01 diagnosis

## What failed

- The model solved 0 of 32 internal development puzzles.
- It proposed `port_reciprocity` on all 32 puzzles. The first proposal failed
  validation on 31 puzzles; the remaining puzzle failed on its second proposal.
- The 32 rejected deductions comprise 26 `unsupported_rule` cases (the claimed
  port rule removes no candidates at that cell) and 6 `wrong_candidates` cases
  (the proposed candidate list does not match the tiles supported by that rule).
- The reported 106/1,762 correct cells are exactly the 106 given tiles present
  before the model acted. The model did not resolve any additional cell to a
  single tile.
- The recorded first trace action uses a row or column count rule on 26 of the
  same 32 puzzles. A valid action need not match the recorded one, but these
  model proposals were independently checked and invalid.

## Why low development loss did not predict success

- The full training set selected by `split_examples` has 8,640 deduction
  targets; 4,938 (57.2%) use `port_reciprocity`.
- Randomly capping deductions at 48 per training puzzle retained the initial
  action for only 73 of 180 training puzzles. It retained 18 of 32 initial
  development actions. The one-step evaluation begins at exactly these
  underrepresented initial states.
- The prompt presents candidate domains as integer bitmasks. The model must
  decode tile sets and apply count and port rules while also generating the
  rule name, location, and full remaining candidate list.
- Development loss (1.296 before, 0.174 after) measures teacher-forced token
  prediction over selected trace rows. It does not measure whether the model's
  first self-chosen action is legal.

## Checks and next experiment

- The rollout's initial domains matched the trace and source initial domains
  for 31 puzzles. The one apparent mismatch was a recorded-history mutation
  after a valid first action; `modal/evaluate.py` now snapshots domains when
  recording states. This logging bug did not change the state sent to the
  model or cause the invalid deductions.
- The local action-policy tests pass, including validation of every positive
  trace action. The checker accepts actions other than the recorded action when
  they follow a visible rule.
- Next experiment: present domains as tile names, preserve all initial-board
  examples, balance count-rule and port-rule examples, and measure exact
  first-action validity before another whole-puzzle run. Continue to report
  whole puzzles solved separately from development loss.

The board comparison is in `predicted-vs-actual.html`; the full rollout is in
`rollout.json`.
