# Run 03 medium evaluation diagnosis

## Results

- Solved 14 of 32 puzzles (43.8%). All 14 completed boards were valid.
- Matched 2,330 of 2,888 cells (80.7%) across the final predictions.
- The other 18 puzzles reached the 256-action cap. Their actions were accepted by the rule checker; this run had no rejected or unsupported actions.
- Successful puzzles took 136–238 actions (mean 196). Every unsuccessful puzzle used all 256 actions.

## Results by board shape

| Board shape | Solved | Cell matches |
|---|---:|---:|
| 9x9 | 6/8 | 582/648 (89.8%) |
| 9x10 | 4/8 | 632/720 (87.8%) |
| 10x9 | 3/8 | 560/720 (77.8%) |
| 10x10 | 1/8 | 556/800 (69.5%) |

The larger shapes performed worse in this small sample, especially 10x10. This is a signal to investigate, not enough evidence to conclude that board size alone caused the drop.

## Diagnosis

The legal-action menu is doing its job: the model consistently chooses actions the checker accepts. The weakness on medium puzzles is rollout efficiency and search progress. The model often spends the full action budget branching, undoing, and making deductions without reaching a complete board. Raising the cap may let some runs finish, but this result does not show whether the remaining actions would lead to a solution.

Run 03's 31/32 small-puzzle result does not transfer to this medium sample: the medium pilot solved 14/32. The evaluation used 32 deterministically sampled `test_medium` puzzles, with 8 of each board shape, from a held-out set of 304. Treat this as an initial estimate, not a full-set score. If using these results to change training or inference, keep the other 272 puzzles for a later evaluation.

## Suggested next step

Inspect the 18 capped histories for repeated branch choices, branch depth, and whether deductions continue to reduce cell domains. Then run a higher action cap on a small subset of the failures. Compare how many finish and how many merely spend longer in an unproductive search path before changing the training data or policy.

The full per-action histories are in `rollout.json`; the interactive board comparison is in `predicted-vs-actual.html`.
