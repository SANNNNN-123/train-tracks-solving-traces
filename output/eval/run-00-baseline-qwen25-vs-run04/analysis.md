# run-00: base Qwen2.5-1.5B vs run-04 on the 32-puzzle small holdout

Same 32 small puzzles (deterministic dev holdout of the 212-puzzle small split,
1762 cells, 6x6 to 9x9), same prompt, menu, 256-action budget, 32 new tokens,
greedy decoding. Four arms: each policy with and without the depth-2 branch filter.

## Result

| Arm | Solved | Cells | Invalid actions | Median actions | Filter overrides |
| --- | --- | --- | --- | --- | --- |
| base, no filter | **0/32** | 312/1762 (17.7%) | **32** | 8 | 0 |
| base, depth 2 | **0/32** | 312/1762 (17.7%) | **32** | 8 | 0 |
| run-04, no filter | **31/32** | 1721/1762 (97.7%) | 0 | 115 | 0 |
| run-04, depth 2 | **31/32** | 1728/1762 (98.1%) | 0 | 107 | 25 |

Adapter effect: **+31 puzzles, +1409 cells, 32 → 0 invalid actions.**
Filter effect: base unchanged; run-04 +7 cells and 25 overrides, **0 flips**.

## How the base model fails

All 32 failures are `invalid_choice`, and each puzzle dies on its **first** malformed
reply (exactly one invalid event per puzzle). It is not a reasoning failure:

* it emits valid `{"choice":"a00N"}` for a median of **7 actions** (max 65);
* then it drops the JSON wrapper and replies `"a000"` — the right token, wrong
  serialization — and the rollout ends;
* 31 of the 32 terminal replies are `"a000"`, one is `"a005"`;
* by then it had 312/1762 cells (median 9 per puzzle, max 29), i.e. the moves before
  the slip were often legal and sometimes correct.

So zero-shot Qwen2.5-1.5B gets ~18% of the board, then loses the format. The adapter
never does (0 invalid in 32 puzzles).

The depth-2 filter cannot help either baseline arm: with a median of 8 actions the
model rarely reaches a branch, and the one arm that records depth 2 is identical
puzzle-for-puzzle to depth 0 (0 overrides).

## The single run-04 failure is budget-bound

`09012026-small` fails with `action_limit` at 256 actions: 22/63 cells at depth 0,
**29/63 at depth 2**. The recorded run-03 adapter failed the *same* puzzle, also at
256 actions (34/63). This is the only unsolved small puzzle for either adapter, and it
is a budget problem rather than a policy one — worth one 512-action retry.

## Comparison with the recorded run-03 small result

run-03 (different adapter, depth 0, 256 actions): 31/32 solved, 1733/1762 cells,
0 invalid. run-04 here: 31/32, 1721/1762 at depth 0 and 1728/1762 at depth 2. Same
solved count, ~5-12 cells (0.3-0.7%) fewer, and the same single failure.

## Takeaway

On the easy distribution the adapter **is** the policy: without it the model cannot
hold the output format for more than a few moves, so 0/32 is a format result, not a
solving result. The branch filter is a marginal refinement on small boards (25
interventions, +7 cells, no flips) — consistent with small puzzles stalling less often
than 10x10 ones.

## Presentation page

`comparison.html` (self-contained, 0.45 MB) shows both policies on the same puzzle,
side by side, with a shared step slider, per-step narration, the exact reply that
ends each base episode, and a 32-row evidence table. Rebuild with:

```bash
python scripts/render_baseline_vs_run04.py
```

