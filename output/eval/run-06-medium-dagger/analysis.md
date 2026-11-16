# run-06 (DAgger): collection results, training aborted

The notebook was run under papermill with `allow_errors=False`. Collection completed
(82 min); the section-8 guard then asserted `dev puzzles == 8`, dev had **7**
(`08032026` never stalled, so it offered no branch), and papermill stopped there.
Training never ran, so no adapter was produced.

The guard was too strict for that case: it now warns and continues, and only fails if
the training set itself collapses (`labelled_rows < 40`).

## Collected states (40 puzzles asked, 39 delivered)

| | Train | Dev |
| --- | --- | --- |
| Branch states | 334 | 51 |
| Model already correct on its own board | 244 (73%) | 38 (75%) |
| Model wrong | 90 (27%) | 13 (25%) |
| Unrecoverable: truth tile already eliminated | 51 (15%) | 10 (20%) |
| Teachable wrong decisions | **39** | 13 |
| Labelled rows | 283 | 41 |
| Rows after both menu orders | **566** | 41 |
| Tokens (estimate) | 383k | 25k |

Training rows verified independently: 566 rows = 283 states × 2 menu orders, every
label equal to the audited tile, every menu exactly 2 entries, 32 puzzles, no overlap
with dev.

## What the wrong decisions look like

| Model | Truth | Count | Note |
| --- | --- | --- | --- |
| `XX` | `SE` | 23 | exactly the eval failure signature, menu `{SE, XX}` |
| `NS` | `NE` | 7 | tile vs tile |
| `EW` | `SW` | 7 | tile vs tile |
| `NE` | `NS` | 3 | tile vs tile |
| `SW` | `EW` | 1 | tile vs tile |

55% of teachable errors are `XX` where a tile belongs, matching the 42/77 rate
measured on the eval probe's wrong branches. The corrections are on-target; there are
just 42 of them, spread over 39 puzzles (one puzzle, `25052026`, contributes 7).

## Why it was not expected to move much

* 73% of the model's own branch decisions were already right, versus 55% on the eval
  probe's states — the training puzzles are easier.
* 566 rows over 3 epochs is **53 optimizer steps** at effective batch 32, versus
  run-05's 234 steps on 7,488 rows.
* Only 39 of the 283 training rows carry a correction.

## If this is ever resumed

1. Re-run the notebook with the fixed section 8. If
   `/kaggle/working/dagger/dagger-shard*.jsonl` still exist, section 7 skips and the
   cost is training only (~25-40 min); otherwise collection is 82 min again.
2. Train sections 9-10, then 14 for the adapter zip. Sections 11-13 (probe and full
   evaluation) are not needed to obtain the adapter.
3. To make the signal real, collect all 240 training puzzles (~7 h on 2 GPUs): at the
   measured 42 teachable decisions per 39 puzzles that is roughly 300 corrections.

## Rerun result (measured): 27/32, one puzzle worse than run-04

The fixed notebook was re-run end to end, including the full 32-id evaluation at
depth 2 / 512 actions (the only run so far measured that way with one adapter).

| | |
| --- | --- |
| Training | 566 rows, **54 optimizer steps**, 16.8 min, 2 GPUs, 3.65 GiB |
| Dev loss (own-state distribution, 7 puzzles) | 0.0615 → 0.0596 (**−3%**) |
| Full 32 ids | **27/32** — lost `24032026`, gained none |
| Probe (10 ids) | 5/10 (run-04: 6/10) |
| Invalid actions | 0 |
| Branch accuracy, 10 probe ids | 96/173 (55%) → 92/165 (**56%**) |
| Branch accuracy on the 4 targets | 37/39/43/25% → 37/39/43/25% |
| Branch sequences identical to run-04 | 5/10 |
| `XX→tile` wrong decisions | 42 → 39 (of 77 → 73) |

**Where the puzzle went.** `24032026` solved for run-04 at 422 actions. Run-06 made
one different branch decision at step 302 (`r2c4`: run-04 `NS`, which is the audited
tile; run-06 `NE`), then needed 24 branches instead of 19, hit 22 contradictions
instead of 7, and exhausted 512 actions at 50/90 cells. A single flipped decision
that was correct before.

**Why nothing else moved.** 54 steps at 2e-4 barely touched a converged adapter: the
own-state dev loss fell 3%, branch accuracy moved 1 point, and the four targets are
cell-for-cell unchanged except `14072026` (+7 cells, still fails). Two solvers got
slower (`12062026` 176→444, `17042026` 168→294); the other 20 are within a few actions.

**Conclusion.** Run-06 is a 27/32 regression of run-04, not an improvement. The
DAgger data is right in kind and an order of magnitude too small in quantity: at 42
corrections the epoch cannot move the four targets, and the one decision it did change
cost a puzzle. Run-04 at 28/32 remains the checkpoint.

