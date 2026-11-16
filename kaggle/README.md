# Kaggle package: branch-focused Train Tracks policy

This folder trains and evaluates the improvement to
`Qwen/Qwen2.5-1.5B-Instruct` on Kaggle **GPU T4 x2**, without Modal.

The shipped notebook is standalone: it carries the code and only needs the three
`.jsonl` data files uploaded as a dataset. The scripts here are the same logic
for CLI runs and for local CPU verification.

It contains two complementary changes over the run-03 baseline:

| Change | What it fixes |
| --- | --- |
| **Safe depth-2 branch search** at inference | The model picks a branch value blindly when propagation stalls. On clean boards it was right 68.2% of the time; the search raises that to **92.4%** and never rejects the true value. |
| **Branch-focused SFT** | Only 538 of 9,642 training rows (5.6%) were branch decisions, and the menu order leaked the answer (`a001` chosen 205/256 times). Training now uses both menu orders, upsamples branch rows, and turns the 142 wrong-branch trace rows into corrected positives. |

Why this is the right fix: the run-03 medium failures were **not** invalid
actions or bad deductions — they were wrong branch commitments that poisoned the
board. 12 of 18 failed boards no longer contained the reference solution when
the action cap was reached. Branch-value selection is the bottleneck, so both
halves of this package target it directly.

Baseline reference score: **14/32 medium puzzles** (run-03, no filter, baseline
SFT). The package evaluates the *same* 32-puzzle shape-stratified sample
(verified by puzzle id), so the numbers are directly comparable.

---

## Files

| Path | Purpose |
| --- | --- |
| `branch_search.py` | Sound bounded DFS over the two branch values. Rejects a value only when propagation proves it impossible. |
| `data_prep.py` | Builds train/dev views, reverses branch menus, adds corrected wrong-branch rows, upsamples branches. |
| `train_kaggle.py` | LoRA SFT with fp16/bf16 auto-selection and DDP. Kaggle port of `modal/train.py`. `--examples/--dev-examples` train on a precomputed JSONL, `--init-adapter` continues an existing adapter. |
| `evaluate_kaggle.py` | Rollout evaluator with `--branch-search-depth {0,2}`. Kaggle port of `modal/evaluate.py`. |
| `medium_traces.py` | Turns audited medium solutions into branch-state SFT rows: oracle rollout per puzzle, the three-way split (240 train / 32 eval / 32 fresh) and a `--limit` smoke mode. |
| `dagger_traces.py` | DAgger: records the policy's own branch states, labels them from the audited solution, and builds branch-only rows (unrecoverable states counted and skipped). |
| `action_policy.py` | Legal-action menu, prompt, parser, validator (unchanged from the baseline). |
| `trace_logic.py` | Rule engine and board helpers (unchanged). |
| `data/train.jsonl` | 24,565 small-puzzle trace rows — **the only training data**. |
| `data/train-puzzles-small.jsonl` | 212 small puzzle solutions; labels corrected branch rows and defines the 32-puzzle dev holdout. |
| `data/train-puzzles-medium.jsonl` | 304 held-out medium puzzles — **evaluation only**; `--limit 32` picks the same deterministic shape-stratified sample as run-03. |
| `configs/accelerate_t4x2.yaml` | `MULTI_GPU`, 2 processes, `fp16`. |
| `notebooks/train_tracks_traces_kaggle.ipynb` | **Standalone** end-to-end notebook: embeds the library and writes it to `/kaggle/working/tt`, so it only needs the three data files. Install → env check → stats → smoke → train → baseline eval → improved eval → compare, with per-step logs. |
| `notebooks/probe_action_budget.ipynb` | **Standalone** probe: re-runs the 10 medium puzzles run-04 left unsolved with `--max-actions 512` and `--max-new-tokens 16`, sharded over both GPUs. No training. |
| `notebooks/train_tracks_medium_branch_sft.ipynb` | **Standalone** run-05: builds the branch-state dataset from 240 medium puzzles, continues the run-04 adapter for one epoch at 1e-4, then re-evaluates all 32 ids at 512 actions. |
| `notebooks/train_tracks_medium_dagger.ipynb` | **Standalone** run-06: collects the adapter's own branch states from 40 medium puzzles, trains branch-only rows at 2e-4, and re-runs the 10-id probe with a branch-sequence comparison against run-04. |
| `notebooks/train_tracks_dagger_adapter_only.ipynb` | Training half only: takes `train-examples.jsonl`/`dev-examples.jsonl` from a dataset and zips the run-06 adapter (~35-45 min) without paying for collection again. |
| `selfcheck.py` | CPU-only harness check: oracle rollout plus a step-by-step replay of run-03 (14/32, 123 undos, 0 mismatches), so board updates, undo and filter safety are verified before spending GPU time. |
| `requirements.txt` | Pins, without torch. Used by the CLI route; the notebook installs the same versions inline. |

Medium puzzles are never read by `data_prep.py` or `train_kaggle.py`. The
evaluation split (`--limit 32`) is read only by `evaluate_kaggle.py
--dataset medium`; `medium_traces.py` reads the other 272 medium puzzles for the
run-05 fine-tune and re-asserts at generation time that the 32 evaluation ids are
absent from what it writes.

---

## Verify locally first (CPU, ~1 min)

```bash
cd kaggle
python data_prep.py                 # dataset counts
python selfcheck.py                 # oracle rollout over all shipped puzzles
```

`selfcheck.py` replaces the model with an oracle policy that reads the audited
solution and drives the *same* rollout loop, board updates, branch stack and
undo path used on Kaggle. It then re-drives the recorded run-03 medium rollout
through the same harness. Current result:

```
medium             depth 0: solved 32/32  branches 0    both-survive 0   overrides 0  16.2s
medium             depth 2: solved 32/32  branches 215  both-survive 39  overrides 0  30.8s
small dev          depth 0: solved 32/32  branches 0    both-survive 0   overrides 0   6.2s
small dev          depth 2: solved 32/32  branches 89   both-survive 9   overrides 0   9.2s
replay run-03      depth 0: solved 14/32  undos 123     mismatches 0
SELF-CHECK PASSED
```

Three things this proves before you rent GPU time:

* the harness can finish every puzzle (oracle 32/32 on medium and dev);
* the depth-2 filter never overrides a correct branch on a clean board
  (215 medium branch decisions, zero overrides, 39 ambiguous cases left to the
  model);
* the Kaggle port is faithful to the original: replaying the recorded run-03
  choices reproduces **14/32** and all 123 undo actions with zero mismatches.

The replay step is skipped automatically if
`output/eval/run-03-legal-action-menu/medium/rollout.json` is not present, so
`selfcheck.py` also works from a standalone copy of this folder.

The notebook embeds a generated copy of these modules. That copy was extracted
back out and run through the same self-check (32/32 oracle, run-03 replay 14/32,
123 undos, zero mismatches), so Route A and Route B execute identical logic.

---

## Kaggle setup

There are two ways to run this. The notebook is **standalone** — it carries all
the code and writes it to `/kaggle/working/tt` at runtime — so it only needs the
three data files from a dataset. The CLI route runs the scripts in this folder
and needs the folder uploaded.

### Route A — standalone notebook (recommended)

1. **Create the notebook** at <https://www.kaggle.com/code> with
   *Accelerator → GPU T4 x2* and *Internet → On*.
2. **Upload only the data** as a private dataset: a folder containing
   `train.jsonl`, `train-puzzles-small.jsonl`, `train-puzzles-medium.jsonl`
   (17 MB, e.g. `kaggle/data/`). Name it something like `train-tracks-data`.
3. **Import the notebook**: *File → Import Notebook* →
   `notebooks/train_tracks_traces_kaggle.ipynb`, then attach the dataset.
4. **Run All.** The notebook installs its own dependencies, materializes
   `tt_core.py`, `tt_lib.py`, `tt_train.py` and `tt_eval.py` from cells, finds
   the data under `/kaggle/input`, and asserts that no medium puzzle id appears
   in `train.jsonl`.
5. Attach a Hugging Face token only if the model download is rate-limited
   (`Qwen/Qwen2.5-1.5B-Instruct` is public, so normally you do not need one).

Nothing from this folder has to be uploaded for Route A. The four modules it
writes are a generated copy of the scripts here; if you edit the scripts, rerun
the notebook cells rather than expecting the notebook to pick up the change.

**If a cell fails**, open the log it names — every step streams to
`/kaggle/working/logs/<step>.log` and raises on a non-zero exit instead of
silently continuing. The three common cases:

* `smoke` fails in seconds with `ImportError: Found an incompatible version of
  torchao` → the kernel had already imported PEFT before the uninstall, or the
  uninstall cell was skipped. Run section 2, then *Restart & Run All*.
* `smoke` fails with an `ImportError`/`TypeError` from `trl` or `transformers` →
  the dependency cell did not run or the kernel still holds the old modules. Run
  section 2, then *Restart & Run All*.
* `smoke` fails with `ValueError: SFTTrainer does not support nn.DataParallel` →
  the process was started without a launcher while several GPUs were visible and
  `CUDA_VISIBLE_DEVICES` was already set to something with >1 device. Use
  `CUDA_VISIBLE_DEVICES=0 python tt_train.py ...`, or launch through
  `accelerate`.
* `train-ddp` fails but `train-single` works → the launcher, not the training.
  The notebook falls back automatically; expect about twice the runtime.

### Route B — CLI from this folder

1. Same notebook settings; upload the whole `kaggle/` folder as a dataset
   (17 MB, e.g. `train-tracks-kaggle`).
2. Attach it and copy it out of the read-only mount:

```bash
import os, shutil
PKG = "/kaggle/input/train-tracks-kaggle"   # adjust to `find /kaggle/input -name train_kaggle.py`
shutil.copytree(PKG, "/kaggle/working/pkg")
os.chdir("/kaggle/working/pkg")
```

3. Install the pins:

```bash
!pip install -q -r requirements.txt
```

`requirements.txt` deliberately does **not** install torch. Kaggle's image
already has a CUDA torch that matches its driver; replacing it is the most
common way to break a Kaggle GPU notebook. After installing, restart the kernel
only if `transformers` was already imported.

4. Sanity check:

```bash
!nvidia-smi
!python -c "import torch; print(torch.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

You want `2` devices and two `Tesla T4` entries.

---

## Run (CLI route)

Notebook users can skip this section — the notebook performs the same steps in
cells. These commands assume `cwd` is the copied `kaggle/` folder.

### 0. Data statistics (CPU, seconds)

```bash
!python data_prep.py --branch-repeat 2
```

Expected on the shipped data:

```
total 11824, base_selected 9642, branch 538, branch_corrected 142,
reversed_branch_orders 680, branch_examples_after_repeat 2720,
puzzles_train 180, puzzles_dev 32, dev_examples 1707
```

### 1. Smoke test (minutes, catches config errors)

```bash
# stage 1: model + TRL + data, one GPU, no launcher
!python train_kaggle.py --smoke --output /kaggle/working/smoke

# stage 2: the exact 2-GPU launcher the full run uses
!python -m accelerate.commands.launch --multi_gpu --num_processes 2 \
    --mixed_precision fp16 --main_process_port 29501 \
    train_kaggle.py --smoke --output /kaggle/working/smoke-ddp
```

4 optimizer steps, 16 train / 8 dev examples. Each stage should end with
`development_loss_before` / `development_loss_after` in its `training.json`
and `dtype` should be `torch.float16` on a T4.

Stage 1 is a plain process, so the script pins it to GPU 0: with several GPUs
visible, `Trainer` picks `nn.DataParallel`, and `SFTTrainer` rejects that with
`ValueError: SFTTrainer does not support nn.DataParallel`. Multi-GPU only works
through a launcher, which is what stage 2 checks.

### 2. Full training

```bash
!accelerate launch --config_file configs/accelerate_t4x2.yaml \
    train_kaggle.py --output /kaggle/working/run-branch
```

* 1 epoch, 11,824 examples, 370 optimizer steps, effective batch 32
  (2 per device × 8 accumulation × 2 GPUs).
* LoRA r=16 / α=32 on all attention and MLP projections; LR 2e-4 cosine.
* `max_length=2048` (same as the baseline), gradient checkpointing on.
* fp16 on T4; bf16 is selected automatically on newer GPUs.
* Expect **1–1.5 h** on T4 x2 (about twice as long on a single T4).

To reproduce the baseline's effective batch of 16 instead of 32, add
`--grad-accum 4`.

### 3. Baseline evaluation (model alone)

```bash
!python evaluate_kaggle.py \
    --adapter /kaggle/working/run-branch/adapter \
    --branch-search-depth 0 \
    --output /kaggle/working/eval-baseline.json --compact
```

### 4. Improved evaluation (depth-2 branch filter)

```bash
!python evaluate_kaggle.py \
    --adapter /kaggle/working/run-branch/adapter \
    --branch-search-depth 2 \
    --output /kaggle/working/eval-improved.json --compact
```

Both runs use the same 32 shape-stratified medium puzzles (`--limit 32`,
deterministic by puzzle-id hash), so the two JSONs are directly comparable.
Each rollout is up to 256 actions; expect **~25 min per evaluation**. At
`--max-actions 512` (the recommended setting after the budget probe) budget
~40–45 min per full evaluation, since cost scales linearly with actions.

Drop `--compact` if you want per-step board snapshots for failure analysis
(large: ~30 MB per run).

### 4b. Targeted re-runs (`--only`) and the action-budget probe

`--only` replaces both sampling paths with an explicit id list, so you can re-run
just the puzzles that failed:

```bash
!python evaluate_kaggle.py \
    --adapter /kaggle/working/run-branch/adapter \
    --branch-search-depth 2 \
    --max-actions 512 --max-new-tokens 16 \
    --only 03092026-medium,10012026-medium \
    --output /kaggle/working/eval-only.json --compact
```

Ids must exist in the selected split (`test_medium`, or the small dev holdout);
anything else raises `--only: not in this split/sample`. The report records
`"sample_strategy": "explicit_ids"` so a partial run is never mistaken for a full
one. Every other flag behaves as usual.

`notebooks/probe_action_budget.ipynb` wraps this: it re-runs the 10 medium
puzzles run-04 left unsolved with `--max-actions 512` and
`--max-new-tokens 16`, split 5/5 across both GPUs as concurrent processes. No
training. Attach two datasets — the run-04 adapter files (flat, not the zip) and
the data dataset containing `train-puzzles-medium.jsonl` — then Run All; expect
**~45–55 min**.

**Measured result (2026-11-15): 6/10 flipped → 28/32**, at 42 min wall clock for
the slower shard. Cells at 256 → 512 were `23→16, 76→90, 40→90, 35→38, 71→81,
64→90, 76→100, 20→90, 40→81, 49→54`; 0 invalid actions in 173 branch decisions,
16 filter overrides. The finishing counts split into two groups:

| Flips | Actions needed | Cheap at 384? |
| --- | --- | --- |
| `18102026`, `10012026`, `22012026` | 270, 274, 289 | yes |
| `19042026`, `24032026`, `13072026` | 407, 422, 434 | no |

So **use `--max-actions 512`, not 384**: 384 returns only 3 of the 6 flips
(25/32). The four remaining ids (`03092026`, `14072026`, `25022026`,
`30082026`) still hit the cap, and `03092026` actually regressed 23 → 16 cells —
the extra budget let it commit to a wrong branch it used to abandon. Those are
policy-bound, so the next gain has to come from branch quality, not from a larger
cap or a larger model.

The report for that run is
`output/eval/run-04-branch-search-depth/probe-merged.html` (self-contained, no
CDN) beside the raw `probe-merged.json`.

`--max-new-tokens 16` is safe for the current action format (the reply targets
~8 tokens); the probe's `invalid` column verifies it — a non-zero count means a
truncated reply.

### 4c. Medium branch-state SFT (run-05)

The four remaining failures are policy-bound, so the next lever is branch quality.
Two cheaper ideas were measured and **rejected**:

| Idea | Measurement on the recorded 10-probe run | Verdict |
| --- | --- | --- |
| Don't re-take a branch value that already led to a contradiction | sound (subset-context) scoping fires **0/173** decisions, because every retry happens under a different branch stack; the aggressive cell-scope blocks 25 decisions of which **15 were correct guesses** (`19042026` 6/6, `24032026` 3/3, `22012026` 2/2, `13072026` 1/1) | would cost solved puzzles |
| Deeper branch filter (depth 3/4) | replaying the exact committed actions: depth 3 changes 1/173 decisions, depth 4 changes 9 for a net **+3** | filter is out of information |

`medium_traces.py` builds the training data instead: for each puzzle it replays the
audited solution with an oracle (undo when pending, take the true value at every
branch), keeps every branch state plus a deduction sample, and emits each branch in
**both menu orders** so `a000`/`a001` carries no information. The split is
deterministic and leak-checked:

| pool | puzzles | contents |
| --- | --- | --- |
| eval | 32 | exactly the ids run-04 measured; asserted unchanged |
| fresh | 32 | held out; development loss only |
| train | 240 | 7,488 rows (~6.9M tokens): 3,408 branch, 3,840 deduction, 240 finish |

Measured generation cost: **2.7 min on a laptop CPU** for the full 240 puzzles, so
it runs inside the notebook without a GPU.

```bash
python medium_traces.py --data data/train-puzzles-medium.jsonl \
    --output-dir /kaggle/working/examples --max-deductions 16   # 240/32/32 split
python train_kaggle.py --examples /kaggle/working/examples/train-examples.jsonl \
    --dev-examples /kaggle/working/examples/dev-examples.jsonl \
    --init-adapter /kaggle/input/run-04-branch-search-depth/adapter \
    --output /kaggle/working/run-medium --epochs 1 --lr 1e-4 \
    --max-length 4096 --batch-size 1 --grad-accum 16
```

`--max-length 4096`, not 2048: the mean medium prompt is ~900 tokens but the
longest is ~3.3k, and truncating cuts the menu **off the end of the prompt** while
the completion is a menu index, so the label would point at a deleted menu. Batch 1
x accum 16 keeps the same effective 32 examples per step with half the peak memory.

`notebooks/train_tracks_medium_branch_sft.ipynb` wraps all of it: dataset build with
the split assertion and an explicit train/held-out overlap check, single-GPU and DDP
smoke runs, the epoch, then the 10-id probe and the remaining 22 ids at depth 2 /
512 actions. Attach the run-04 adapter dataset and the medium data dataset, then Run
All; expect **~3.5 h** (2.7 min data, ~6 min smoke, ~1–2 h training, 40 min probe,
~80 min for the other 22).

The comparison cell prints gains **and losses**, because a new solved puzzle that
costs an old one is not progress. Success bar: probe ≥ 6/10 (otherwise stop before
the 80 min evaluation), full set **30+/32** with zero losses and zero invalid
actions.

**Scope: medium only.** Run-05 trains and evaluates on medium, continuing weights
that already contain the small-puzzle training (24,565 rows). The small dev holdout
was **31/32** under run-03 and is *not* re-measured here, so a forgetting cost on
easy puzzles would go unrecorded. That is a deliberate trade: the eval target is
medium, the easy distribution has almost nothing left to teach, and a mixed set
would drop medium to ~23% of the batch. If one policy for both sizes is ever
needed, the guard is cheap — the 32 small dev puzzles take ~12 min at 256 actions
(6 min on two shards), and a rehearsal slice of ~60 small puzzles' branch examples
(~1.2k rows) is the fix if that guard fails.

**Measured result (run-05): no transfer, and the reason is instructive.**

| Measurement | Value |
| --- | --- |
| Training | 7,488 rows, 234 optimizer steps, 2.12 h, 2 GPUs, 6.66 GiB peak |
| Dev loss, 32 puzzles never trained on | 0.1157 → **0.0491** (−58%) |
| Per-token probability of the correct completion | 89.1% → 95.2% |
| Probe | **6/10, identical to run-04**: 0 fixed, 0 lost, 0 invalid |
| Branch decisions changed | 3 of 10 puzzles, one decision each |
| Correct branch decisions | 96/173 → 95/163 |
| Deduction actions | 3,753 → 3,975 (+5.9%) |

It learned — the drop is on *unseen* puzzles, so this is not memorisation — but what
it learned was the wrong states. A baseline dev loss of 0.116 per token over a ~7-token
completion (~0.8 nats) means run-04 already put the correct choice first on essentially
every teacher-state example; the only headroom was the margin, and the epoch spent 2 h
consuming it. The boards that fail are the ones the model **creates itself** after a
wrong branch, and the oracle never visits them. Loss on teacher states is therefore a
metric that cannot see the failure — see §4d for the fix.

### 4d. DAgger on the policy's own branch states (run-06)

`dagger_traces.py` closes the distribution gap. Collection mode wraps any policy,
rolls out the adapter, and stores the **exact board at every branch decision** plus
the audited value:

```bash
python dagger_traces.py --adapter /kaggle/input/run-04-branch-search-depth \
    --data data/train-puzzles-medium.jsonl \
    --only 25032026-medium,02012026-medium,... \
    --max-actions 256 --branch-search-depth 2 --max-new-tokens 16 \
    --output /kaggle/working/dagger/dagger-shard0.jsonl
```

Build mode turns those states into branch-only rows:

```bash
python dagger_traces.py --build --states dagger-shard0.jsonl,dagger-shard1.jsonl \
    --output-dir /kaggle/working/dagger --role train --only <ids>
```

Three properties matter:

* **Label = the audited value**, so a state where the policy guessed wrong becomes a
  corrected positive. Both menu orders are still generated.
* **Unrecoverable states are counted, not trained.** If a poisoned board no longer
  offers the true tile, no label would be correct; the builder reports
  `unrecoverable` and skips it rather than teaching the other value.
* **Repeated boards are deduped.** A revisit after undo renders the same prompt.

`notebooks/train_tracks_medium_dagger.ipynb` runs the whole loop: 32 training puzzles
(plus 8 held-out fresh puzzles for dev loss), 2 shards, ~70–85 min collection, ~900
branch rows, 3 epochs at 2e-4 (~40–60 min), then the same 10-id probe. It also
compares branch **sequences** against run-04's, embedded in the notebook — run-05
changed 3 of 10 and gained nothing, so "did the policy move at all" is now an explicit
check rather than something inferred from counts.

Success bar: **7+/10** on the probe with sequences that differ from run-04. 6/10 with
different sequences means the policy moved but not enough; 6/10 with identical
sequences means these states still were not the binding ones. If it clears 7/10,
scale collection to all 240 training puzzles (~7 h on 2 GPUs) for the final adapter.

**Measured first collection (run-06), and where this work stops.** 40 puzzles asked,
39 delivered (one never stalled, so it offered no branch):

| | Train | Dev |
| --- | --- | --- |
| Branch states | 334 | 51 |
| Model already correct on its own board | 244 (73%) | 38 (75%) |
| Unrecoverable (truth no longer offered) | 51 (15%) | 10 (20%) |
| Rows after both menu orders | 566 | 41 |

So the teachable signal is **42 wrong decisions**, 23 of them the eval failure
signature (`XX → SE`), and the training set is **53 optimizer steps** at batch 32 —
an order of magnitude smaller than run-05. The run was stopped after collection: the
notebook's own guard (`dev puzzles == 8`) aborted papermill before training, which is
the right failure but the wrong severity for a puzzle that simply never branched. The
guard now warns instead, and the adapter-only path is sections 8 → 10 → 14.

**Final state of this work: run-04 at 28/32 is the checkpoint.** Teacher-state SFT
(run-05) was measured and rejected; DAgger (run-06) was collected, trained and
evaluated on all 32 ids: **27/32**, with `24032026` lost because one branch decision
that run-04 made correctly (`r2c4 = NS`) flipped to `NE`, after which the run needed 24
branches instead of 19 and exhausted 512 actions. Branch accuracy on the four targets
was unchanged (37/39/43/25%), and the own-state dev loss moved only 3% in 54 steps.
The data was right and roughly 10× too small; see
`output/eval/run-06-medium-dagger/analysis.md`.

### 5. Compare

```python
import json
for name in ("eval-baseline", "eval-improved"):
    r = json.load(open(f"/kaggle/working/{name}.json"))
    print(f"{name:14s} solved {r['solved']:2d}/{r['attempted']}  "
          f"valid boards {r['valid_complete_boards']:2d}  "
          f"branch overrides {r['search_overrides']:3d}  "
          f"failures {r['failures']}")
```

---

## What to expect

| Metric | run-03 baseline | run-04, 256 actions | run-04, 512 actions |
| --- | --- | --- | --- |
| Medium solved | 14/32 | 22/32 (measured) | 28/32 (measured) |
| Branch accuracy on clean boards | 68.2% | 98.5% solved / 54% on failures | unchanged |
| True branch value rejected | n/a | never, while the board still contains the solution | same |

Run-05 (§4c) continued from the 28/32 adapter and fine-tuned on medium branch
states: it moved dev loss 58% and the probe **not at all** (6/10, identical branch
sequences on 7 of 10 puzzles). Run-06 (§4d) trains on the adapter's own branch
states instead, which is the only distribution that contains the failures. Its
result is not in this table until the notebook has run; the target is **30+/32 with
no puzzle lost**, and the notebook prints gains and losses separately so a trade
cannot hide inside the total.

The search is an **oracle-free safety net**, not a solver: it only removes a
branch value when propagation plus two guesses hits a contradiction. When both
values survive (≈16% of clean decisions) it keeps the model's choice, so the SFT
half carries those cases.

The ceiling is set by boards that are already poisoned: once a wrong branch
commits a tile that the true solution needs, no filter can recover it. The
branch-focused SFT exists to reduce how often that happens.

If the improved run is **not** better than baseline, check in order:

1. `search_overrides` in `eval-improved.json` — if it is 0, the filter never ran
   (wrong `--branch-search-depth`, or the adapter emitted no branches).
2. `failures` — a spike in `invalid_choice` means the tokenizer/chat template
   was rebuilt differently at eval time; a spike in `unsupported_rule` means
   deduction quality regressed, so retrain with `--branch-repeat 1` (branch
   share drops from 23% to 11.5%).
3. Development loss — if `development_loss_after` is not well below
   `development_loss_before`, the adapter did not train (LR, dtype, or data).

---

## T4 x2 notes

* **fp16 only.** T4 is sm_75 and cannot run bf16. Both scripts gate on
  `torch.cuda.get_device_capability(0)[0] >= 8` **and**
  `is_bf16_supported()`, because `is_bf16_supported()` returns True on some T4
  images, where bf16 would then run in slow emulation (or fail). On T4 you must
  see `dtype: torch.float16` in the smoke report. The accelerate config pins
  `mixed_precision: fp16` to match.
* **No in-process multi-GPU.** `SFTTrainer` rejects `nn.DataParallel`, which
  `Trainer` selects whenever one process sees several GPUs. So DDP only happens
  through `accelerate launch`/`torchrun`, and `train_kaggle.py` pins an
  unlaunched process to GPU 0 (printing a one-line note). Never set
  `CUDA_VISIBLE_DEVICES` after CUDA has been touched: `cudaGetDeviceCount`
  caches the device list.
* **torchao must be removed.** Kaggle ships torchao 0.10, and PEFT ≥0.21 raises
  `ImportError: Found an incompatible version of torchao. Found version 0.10.0,
  but only versions above 0.16.0 are supported` while injecting LoRA. Upgrading
  torchao drags torch with it, so uninstall it instead:

  ```bash
  %pip uninstall -y -q torchao
  ```

  The notebook does this in the dependency cell and then runs a `torchao-check`
  step that fails loudly if it is still importable. For the CLI route, run the
  same uninstall before `train_kaggle.py`. It must happen before anything
  imports PEFT; otherwise restart the kernel.
* **OOM.** 1.5B parameters in fp16 is ~3 GB per GPU, so 16 GB is comfortable at
  batch 2 / 2048 tokens with gradient checkpointing. If you still OOM, use
  `--batch-size 1 --grad-accum 16`, or `--max-length 1536`.
* **DDP port.** If `accelerate launch` cannot bind, append
  `--main_process_port 29501`. The notebook invokes
  `python -m accelerate.commands.launch ...` so it pins the same interpreter
  that imported the wheels instead of relying on PATH.
* **No config file fallback.** If Kaggle's accelerate rejects the YAML, use
  `accelerate launch --multi_gpu --num_processes 2 --mixed_precision fp16 train_kaggle.py ...`.
* **Session limits.** A Kaggle GPU session lasts up to 12 h (≈30 GPU-h/week).
  The run-04 flow needs roughly **2–3 h** (training 1–1.5 h, two evaluations
  ~25 min each); the run-05 notebook (§4c) needs **~3.5 h** including its two
  evaluation rounds.
* **Save output.** `/kaggle/working` is the only persisted directory; download
  `run-branch/adapter` plus the two eval JSONs before the session ends.

---

## Design notes

**Why depth 2 and not 3.** Validated offline on all 157 fully clean branch
decisions in the run-03 medium rollouts:

| Depth | Branch accuracy | True value survived | Both survived |
| --- | --- | --- | --- |
| 0 (model) | 68.2% | — | — |
| 1 | 87.9% | 100% | 26.1% |
| 2 | **92.4%** | **100%** | 15.9% |
| 3 | 92.4% | 100% | 14.0% |

Depth 3 costs more and changes no decision, so `DEFAULT_DEPTH = 2`. A node
budget (`--branch-search-budget`, default 4000) keeps pathological boards from
stalling the evaluator.

**Why menu reversal matters.** The evaluator sorts the legal menu, so branch
values always appear in the same slot order per pair. The model learned
"`a001`" rather than "the value that fits", choosing `a001` 205/256 times. Both
orders are now in the training set, which removes the positional shortcut.

**Why corrected wrong-branch rows are safe.** `split_examples` drops rows whose
`expert_policy_label` is false, but their *state* is a legitimate pre-branch
board. The audited solution tells us the right value, so the row is re-added
with a corrected target — more hard negatives-to-positives without ever training
a wrong label.

**Determinism.** Seed 42 for data shuffling, training and data loader. The dev
holdout is 32 small puzzles chosen by puzzle-id hash
(`development_ids`), disjoint from training. Run-04 never reads medium; run-05
(§4c) trains only on the 240 medium puzzles outside the 32 evaluation ids and the
32 fresh ids, and the generator refuses to write anything if that split moved.

---

## Next steps (not included)

* **Scale DAgger to the full training pool.** Run-06 (§4d) collects 40 of the 240
  training puzzles. If it clears the 7/10 bar, roll out all 240 (~7 h on 2 GPUs at
  256 actions) and train on the resulting ~3,000 branch states. That is the version
  worth keeping as a checkpoint.
* **Track decision accuracy, not just loss.** Run-05's 58% dev-loss drop came with
  zero decision changes. A teacher-forced argmax accuracy on the dev rows would have
  shown the headroom was ~0 before spending 2 GPU-hours; worth adding to
  `train_kaggle.py` before the next training run.
* **Calibrate the fallback.** When both values survive the filter, the model
  decides. A small value-prior head or a deeper budget on ambiguous boards is
  where the remaining headroom is.
* **Ensemble the two menu orders** at inference: query the adapter with both
  orders and require agreement before committing a branch.
