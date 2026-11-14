# Why 18 of 32 medium puzzles failed (run-03)

Analysis of `medium/rollout.json` (32 `test_medium` puzzles, 256-action cap) and the
matching teacher traces in `data/train_medium.jsonl`. Every number below is
reproducible with the analysis scripts named at the end.

## Headline

All 18 failures are **action-limit failures**, not invalid-action failures: the
checker accepted every one of the 4,608 actions in those rollouts, and there were
zero no-op deductions. The model can still deduce. The gap is **branch-value
selection and the search cost that follows from wrong guesses**.

| | Solved (14) | Failed (18) |
|---|---:|---:|
| Actions used (mean) | 195.9 | 256 |
| Teacher trace length (mean) | 183.4 | 198.9 |
| Actions discarded by undo (mean) | 33.4 (15.4%) | 122.0 (47.7%) |
| Actions that survive (mean) | 162.5 | 134.0 |
| Wrong branch guesses (mean) | 1.5 | 4.4 |
| Branch accuracy | 73.8% | 54.5% |
| Singleton cells at stop (mean) | 86.9 | 67.7 |

The failed rollouts do **more** total deduction work than the solved ones (236 vs
185 deduction actions, 596 vs 451 candidate removals), but almost half of it is
thrown away when a wrong branch is undone. Productive work ends up at 134
actions versus the ~199 the teacher needs for the same puzzles.

## Root cause chain

1. **Branches are forced, not premature.** `legal_actions()` only offers a
   branch when no deduction, contradiction, or finish action exists, and it offers
   exactly the two values of one cell chosen by `branch_cell()` (fewest
   candidates, then top-left). Every one of the 256 branch steps in this run had
   `legal_action_count == 2`. The model is not choosing to branch too early; the
   checker has nothing else to offer.

2. **The only decision that can leave the true solution is the branch value.**
   Deductions are rule-checked and sound relative to the current assumptions.
   The model's deduction behaviour is close to the teacher's: 2.46 vs 2.46
   candidates removed per deduction on solved puzzles, and 2.65 vs 2.44 on
   failed ones, with zero no-op deductions. So the generalization gap is
   concentrated in the branch decision.

3. **The model picks the branch value from a fixed tile prior, not from the
   board.** It chose menu option `a001` 205/256 times (80%). Because branch
   options are sorted by value string, `a001` is usually `XX` (the alphabetically
   last tile), and `XX` was chosen 160 times. It never chose `SE` in 160
   `{XX, SE}` branches and never chose `NE` in 44 `{NS, NE}` branches. On the
   failed set this is only 54.5% accurate, and that is almost exactly what a
   context-free training prior scores:

   | | Model | Training tile prior | Per-pair oracle prior |
   |---|---:|---:|---:|
   | Solved branches | 73.8% | 73.8% | 73.8% |
   | Failed branches | 54.5% | 55.1% | 59.7% |

   By candidate pair on the failed set:

   | Pair | Model accuracy | What the model picked |
   |---|---:|---|
   | `{XX, SE}` | 71/115 = 62% | `XX` every time |
   | `{EW, SW}` | 18/35 = 51% | `EW` 34/35 times |
   | `{NS, NE}` | 7/26 = 27% | `NS` every time (`NE` is correct 15/26) |

   The training set has only 538 branch examples out of 9,642 (5.6%), so the
   model learned the marginal "which tile is usually the successful branch"
   statistics of the 212 small puzzles, not how to read the clues at the branch
   cell.

4. **A wrong guess is expensive, and backtracking is chronological.** After a
   wrong value, the model deduces forced consequences until a contradiction.
   The 80 undos that removed a value disagreeing with the reference discarded
   44.6 actions on average (median 34.5). The other 43 undos removed a value that
   *was* correct but had led to a contradiction under an earlier wrong guess;
   those still discarded 33.9 actions on average (median 20). `legal_actions()` then
   forces an undo of the *most recent* branch, and `evaluate.py` commits the
   opposite value by clearing the chosen bit:

   ```python
   board = snapshot(branch["checkpoint"])
   board[r][c] &= ~BIT[branch["value"]]
   ```

   That flip is sound only relative to the remaining assumptions. When the real
   culprit was an earlier guess, the flip can remove the true tile and the board
   no longer contains the unique solution. The model must exhaust the whole
   subtree under the wrong ancestor before it can undo it. That is why the
   failures burn 60-210 actions after the first wrong branch.

5. **Larger boards amplify this.** Failed puzzles need more branch decisions
   (teacher branch count 9.0 vs 5.9 on solved), and 10x10 has both more
   ambiguous cells and longer traces. Shape is a risk multiplier, not the cause:
   9x9 also failed (2/8), and the solved/failed split is explained by branch
   accuracy, not by shape alone.

## Two kinds of failure

At the cap, 12 of the 18 boards no longer contain the true solution (a wrong
branch is still active or a correct tile was committed away). The other 6 are
clean and merely unfinished.

| Failure type | Puzzles | What it means |
|---|---|---|
| Clean but unfinished | 01012026, 05082026, 10012026, 20062026, 22012026, 28052026 | Board still contains the solution and no wrong branch is active. A higher cap or modest efficiency gain should finish these. |
| Trapped in a wrong subtree | 02022026, 03032026, 03092026, 05022026, 06052026, 13072026, 14072026, 18102026, 19042026, 24032026, 25022026, 30082026 | Board has lost the solution; the model must backtrack past a wrong ancestor. A bigger cap is not reliable here. |

Six of the 12 trapped puzzles still have their **first wrong branch active** at
the cap (03092026, 13072026, 14072026, 24032026, 25022026, 30082026).

## Is the 256 cap too low?

Not mainly. Teacher traces for **17 of the 18** failed puzzles fit under 256
(mean 198.9; only `01012026` at 270 exceeds it). Solved rollouts took about
1.03-1.07x the teacher length. If the failed puzzles branched as efficiently as
the solved ones, about 16 of 18 would fit under 256 (only `01012026` and
`24032026` would be tight).

At the observed waste rate (47.7% discarded), the cap needed would be roughly
**360 actions (median; range ~290-630)**. Raising the cap alone would let some
rollouts wander longer in a doomed subtree rather than fix the search.

## Per-puzzle evidence

`lost` = actions inside a branch that was later undone; `useful` = 256 - lost;
`wrong` = branch values that disagree with the reference solution; `poison@end`
= cells whose domain no longer contains the reference tile at the cap.

| puzzle | shape | solved | model | teacher | lost | useful | wrong | poison@end | first wrong |
|---|---|:--:|--:|--:|--:|--:|--:|--:|---|
| 01012026 | 10x10 | no | 256 | 270 | 93 | 163 | 2 | 0 | step 106, depth 1 |
| 02022026 | 9x10 | no | 256 | 184 | 108 | 148 | 6 | 2 | step 81, depth 1 |
| 03032026 | 10x10 | no | 256 | 192 | 107 | 149 | 4 | 6 | step 79, depth 0 |
| 03092026 | 9x9 | no | 256 | 175 | 128 | 128 | 6 | 29 | step 36, depth 2 |
| 04062026 | 9x10 | yes | 212 | 173 | 47 | 165 | 4 | 0 | step 52, depth 0 |
| 05022026 | 10x10 | no | 256 | 230 | 99 | 157 | 3 | 17 | step 40, depth 0 |
| 05082026 | 10x9 | no | 256 | 151 | 122 | 134 | 4 | 0 | step 90, depth 0 |
| 06052026 | 9x10 | no | 256 | 187 | 98 | 158 | 3 | 16 | step 93, depth 4 |
| 08102026 | 9x9 | yes | 238 | 183 | 87 | 151 | 3 | 0 | step 49, depth 0 |
| 10012026 | 10x9 | no | 256 | 171 | 123 | 133 | 2 | 0 | step 55, depth 4 |
| 12062026 | 10x9 | yes | 220 | 206 | 44 | 176 | 1 | 0 | step 84, depth 2 |
| 13072026 | 10x9 | no | 256 | 176 | 156 | 100 | 5 | 4 | step 90, depth 5 |
| 14072026 | 10x10 | no | 256 | 205 | 155 | 101 | 9 | 16 | step 55, depth 0 |
| 17042026 | 9x9 | yes | 182 | 177 | 14 | 168 | 1 | 0 | step 159, depth 4 |
| 18102026 | 9x9 | no | 256 | 180 | 115 | 141 | 7 | 11 | step 79, depth 1 |
| 19042026 | 10x9 | no | 256 | 208 | 129 | 127 | 4 | 8 | step 76, depth 2 |
| 19062026 | 9x10 | yes | 226 | 197 | 69 | 157 | 2 | 0 | step 60, depth 2 |
| 19102026 | 10x10 | yes | 233 | 184 | 55 | 178 | 2 | 0 | step 77, depth 2 |
| 20012026 | 9x9 | yes | 152 | 178 | 0 | 152 | 0 | 0 | - |
| 20062026 | 9x10 | no | 256 | 236 | 95 | 161 | 2 | 0 | step 58, depth 1 |
| 22012026 | 10x10 | no | 256 | 174 | 109 | 147 | 2 | 0 | step 93, depth 1 |
| 22022026 | 9x10 | yes | 207 | 200 | 14 | 193 | 1 | 0 | step 165, depth 4 |
| 24032026 | 10x9 | no | 256 | 252 | 152 | 104 | 2 | 16 | step 35, depth 3 |
| 24102026 | 10x9 | yes | 201 | 196 | 42 | 159 | 1 | 0 | step 108, depth 2 |
| 25022026 | 10x10 | no | 256 | 206 | 172 | 84 | 6 | 6 | step 63, depth 1 |
| 27012026 | 9x9 | yes | 159 | 165 | 0 | 159 | 0 | 0 | - |
| 28052026 | 10x10 | no | 256 | 207 | 97 | 159 | 6 | 0 | step 83, depth 2 |
| 28102026 | 9x10 | yes | 173 | 173 | 7 | 166 | 1 | 0 | step 157, depth 2 |
| 29052026 | 9x9 | yes | 202 | 203 | 53 | 149 | 3 | 0 | step 57, depth 1 |
| 29082026 | 10x9 | yes | 203 | 198 | 36 | 167 | 2 | 0 | step 60, depth 4 |
| 30032026 | 9x9 | yes | 135 | 135 | 0 | 135 | 0 | 0 | - |
| 30082026 | 9x10 | no | 256 | 177 | 138 | 118 | 7 | 18 | step 60, depth 1 |

Small-run comparison (internal holdout from the 212 training puzzles): 31/32
solved, 33.0% wrong branches, 19.7% of actions discarded. So the same branch
policy is forgivable on short small puzzles and decisive on medium ones.

## What I would change, in priority order

1. **Improve branch-value selection first.** It is the only decision that can
   take the board off the solution, and it currently behaves like a fixed prior.
   A cheap 1-ply lookahead (apply each of the two values, run the existing
   deduction engine, prefer the value with no immediate contradiction or more
   forced progress) is likely to recover most of the 18.

2. **Train a branch-value policy on model-reached states.** The current SFT data
   has only ~680 positive branch rows and excludes wrong branches entirely, so
   there is little signal about which of two values is better. Collect failed
   rollout branch states, label both values against the audited solution, and
   train a two-way choice. Randomize menu order or score both option logits so
   the `a001` position bias cannot form.

3. **Fix the backtracking heuristic.** The harness always undoes the most recent
   branch. Conflict-directed backjumping to the branch that actually caused the
   contradiction, or a depth limit, would stop one shallow wrong guess from
   consuming the whole budget.

4. **Then consider the cap.** Raise it for a small subset (e.g. 512-1024) and
   measure whether the 12 trapped puzzles recover by backjumping or merely
   wander. Do not raise it as the primary fix; the teacher traces already fit
   under 256 for 17/18.

## Reproduce

Scripts added under `scripts/`:

- `analyze_medium_failures.py` - per-puzzle action/type/progress summary
- `compare_teacher_medium.py` - teacher lengths and structure for the same 32
- `analyze_branches_medium.py` - branch correctness, branch close reasons
- `analyze_waste_medium.py` - lost/productive actions and counterfactual caps
- `analyze_subtrees_medium.py` - cost of an undone branch
- `analyze_branch_priors.py` - model vs prior baselines
- `analyze_poisoning.py`, `analyze_final_poison.py` - board consistency replay
- `analyze_first_wrong.py` - first wrong guess, depth, and cost
- `compare_small_medium.py` - small vs medium waste/branch comparison
- `build_failure_table.py` - the table above
