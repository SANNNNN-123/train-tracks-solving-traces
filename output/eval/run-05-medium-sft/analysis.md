# run-05 (medium branch-state SFT): measured result

Probe only: the 10 medium ids that run-04 could not finish at 256 actions, re-run
with the new adapter at depth 2 / 512 actions / 16 new tokens. The full 22-id
evaluation was skipped because the four targets were still at the cap, so the score
cannot exceed 28/32 and a full run could only find losses.

## Training

| | |
| --- | --- |
| Data | 7,488 rows from 240 puzzles (3,408 branch, 3,840 deduction, 240 finish) |
| Schedule | 1 epoch, LR 1e-4, cosine, 234 optimizer steps, effective batch 32 |
| Runtime | 2.12 h on 2 T4s, fp16, 6.66 GiB peak |
| Dev loss (572 rows, 32 unseen puzzles) | 0.1157 → **0.0491** (−58%) |
| Per-token probability of the correct completion | 89.1% → 95.2% |

## Probe

| Puzzle | cells run-04 → run-05 | actions | result |
| --- | --- | --- | --- |
| 03092026 | 16/81 → **33/81** | 512 → 512 | still failing |
| 14072026 | 38/100 → 41/100 | 512 → 512 | still failing |
| 25022026 | 81/100 → **69/100** | 512 → 512 | still failing |
| 30082026 | 54/90 → **44/90** | 512 → 512 | still failing |
| 10012026 | 90/90 → 90/90 | 274 → 283 | solved |
| 13072026 | 90/90 → 90/90 | 434 → 466 | solved |
| 18102026 | 81/81 → 81/81 | 270 → 289 | solved |
| 19042026 | 90/90 → 90/90 | 407 → 389 | solved |
| 22012026 | 100/100 → 100/100 | 289 → **420** | solved |
| 24032026 | 90/90 → 90/90 | 422 → 435 | solved |

**6/10 solved — identical to run-04. 0 fixed, 0 lost, 0 invalid actions.**
Eval time 82 min (sum over puzzles). Expected full-set score: 28/32, unchanged.

## Why it did nothing

The loss moved; the policy did not:

* branch sequences are **identical on 7 of 10 puzzles**, and the other three differ
  by a single decision each (14072026 at r3c5, 25022026 at the 18th branch,
  30082026 at r2c6);
* correct branch decisions: **96/173 → 95/163** — one fewer, not better;
* deduction actions: **3,753 → 3,975 (+5.9%)**, so the epoch mostly changed the
  deduction order (22012026 needed 131 more actions for the same board).

The dev set explains it. A baseline loss of 0.116 per token over a ~7-token
completion is ~0.8 nats per example, so run-04 already put the correct menu id first
on essentially every teacher-state row. There was no decision error on that
distribution to fix, only margin. The decisions that fail are made on boards the
model creates itself after a wrong branch, and the oracle never generates those
boards.

Conclusion: teacher-state SFT cannot fix this failure mode, however much loss it
removes. See `kaggle/README.md` §4d and `train_tracks_medium_dagger.ipynb`.
