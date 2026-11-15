"""Evaluate a trained adapter on Kaggle, with an optional safe branch search.

Baseline (model alone):

    python evaluate_kaggle.py --adapter /kaggle/working/run-branch/adapter \
        --branch-search-depth 0 --output baseline.json

Improved (depth-2 branch filter):

    python evaluate_kaggle.py --adapter /kaggle/working/run-branch/adapter \
        --branch-search-depth 2 --output improved.json

The search only rejects a value when propagation proves it impossible, so it
cannot reject the true value on a board that still contains the solution.  When
both values survive, the model's choice is kept.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from action_policy import (SYSTEM, development_ids, legal_actions,  # noqa: E402
                           parse_action, prompt, validate_action)
from branch_search import DEFAULT_DEPTH, choose_branch  # noqa: E402
from trace_logic import (BIT, clue_from_record, completed_error,  # noqa: E402
                         initial_domains, one, snapshot)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data", type=Path,
                        default=HERE / "data/train-puzzles-medium.jsonl")
    parser.add_argument("--small-data", type=Path,
                        default=HERE / "data/train-puzzles-small.jsonl")
    parser.add_argument("--dataset", choices=("medium", "small"), default="medium")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--limit", type=int, default=32,
                        help="0 means all; medium uses a shape-stratified sample")
    parser.add_argument("--max-actions", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--branch-search-depth", type=int, default=DEFAULT_DEPTH,
                        help="0 disables the branch filter")
    parser.add_argument("--branch-search-budget", type=int, default=4000,
                        help="Max search nodes per branch decision")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/eval.json"))
    parser.add_argument("--only", default=None,
                        help="Comma-separated puzzle ids to evaluate; bypasses sampling")
    parser.add_argument("--compact", action="store_true",
                        help="Drop board snapshots from the saved history")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def state_for(board, clue, stack):
    return {
        "row_clues": clue["rows"], "column_clues": clue["columns"],
        "givens": [{"row": r + 1, "column": c + 1, "value": shape}
                   for r, c, shape in clue["given_shapes"]],
        "domains": [line.copy() for line in board],
        "branch_stack": [{"to_step": item["to_step"], "cell": item["cell"],
                          "value": item["value"]} for item in stack],
    }


def select_medium_sample(puzzles, limit):
    groups = defaultdict(list)
    for puzzle in puzzles:
        shape = (len(puzzle["puzzle"]["rows"]), len(puzzle["puzzle"]["columns"]))
        groups[shape].append(puzzle)
    for group in groups.values():
        group.sort(key=lambda row: hashlib.sha256(row["puzzle_id"].encode()).hexdigest())
    target = min(limit, sum(len(group) for group in groups.values()))
    selected = {shape: [] for shape in groups}
    while sum(len(items) for items in selected.values()) < target:
        eligible = [shape for shape in groups if len(selected[shape]) < len(groups[shape])]
        shape = min(eligible, key=lambda key: (len(selected[key]), key))
        selected[shape].append(groups[shape][len(selected[shape])])
    return sorted((p for items in selected.values() for p in items),
                  key=lambda row: row["puzzle_id"])


def model_policy(model, tokenizer, args):
    """Return ``policy(state, actions) -> raw completion text`` for the adapter."""
    import torch

    def policy(state, actions):
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt(state, actions)}]
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **encoded, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(generated[0, encoded["input_ids"].shape[-1]:],
                                skip_special_tokens=True)

    return policy


def rollout(policy, record, args):
    """Run one puzzle with a ``policy(state, actions) -> raw text`` callable.

    The policy seam keeps the harness testable without a GPU: ``selfcheck.py``
    drives the exact same board updates, branch stack and undo path with an
    oracle policy that reads the audited solution.
    """
    clue = clue_from_record(record)
    board = initial_domains(clue)
    stack, history = [], []
    pending = False
    failure = None
    started = time.monotonic()
    for step_index in range(args.max_actions):
        state = state_for(board, clue, stack)
        actions = legal_actions(state, pending=pending)
        if not actions:
            failure = "no_legal_actions"
            break
        raw = policy(state, actions)
        action = parse_action(raw, actions)
        if action is None:
            valid, reason, canonical = False, "invalid_choice", None
        else:
            valid, reason, canonical = validate_action(board, clue, action, stack, pending)

        model_action = canonical
        search = None
        if valid and canonical["type"] == "branch" and args.branch_search_depth > 0:
            decision = choose_branch(
                state, canonical["cell"], canonical["value"],
                depth=args.branch_search_depth,
                node_budget=args.branch_search_budget)
            search = {"model_value": canonical["value"],
                      "chosen_value": decision["value"],
                      "override": decision["override"],
                      "survivors": decision["search"]["survivors"] if decision["search"] else None,
                      "feasible": decision["search"]["feasible"] if decision["search"] else None,
                      "depth": args.branch_search_depth}
            if decision["override"]:
                candidate = {"type": "branch", "cell": canonical["cell"],
                             "value": decision["value"]}
                valid, reason, canonical = validate_action(board, clue, candidate, stack, pending)
                if not valid:  # should not happen; keep the model choice
                    canonical = model_action
                    search["override"] = False

        event = {"step": step_index, "state": state, "raw": raw,
                 "action": canonical if valid else action,
                 "model_action": model_action,
                 "branch_search": search,
                 "legal_action_count": len(actions), "valid": valid, "reason": reason}
        if args.compact:
            event.pop("state", None)
        history.append(event)
        if not valid:
            failure = reason
            break
        kind = canonical["type"]
        if kind == "deduce":
            cell = canonical["cell"]
            r, c = cell["row"] - 1, cell["column"] - 1
            board[r][c] = sum(BIT[x] for x in canonical["candidates"])
        elif kind == "branch":
            cell = canonical["cell"]
            r, c = cell["row"] - 1, cell["column"] - 1
            stack.append({"to_step": step_index, "cell": cell,
                          "value": canonical["value"], "checkpoint": snapshot(board)})
            board[r][c] = BIT[canonical["value"]]
        elif kind == "contradiction":
            if not stack:
                failure = "contradiction_without_branch"
                break
            pending = True
        elif kind == "undo":
            branch = stack.pop()
            board = snapshot(branch["checkpoint"])
            r, c = branch["cell"]["row"] - 1, branch["cell"]["column"] - 1
            board[r][c] &= ~BIT[branch["value"]]
            pending = False
        elif kind == "finish":
            break
    else:
        failure = "action_limit"

    grid = [[one(mask) for mask in row] for row in board]
    reference = [[tile or "XX" for tile in row] for row in record["solution"]]
    correct = sum(a == b for row_a, row_b in zip(grid, reference)
                  for a, b in zip(row_a, row_b))
    finished = bool(history and history[-1]["valid"]
                    and history[-1]["action"]["type"] == "finish")
    return {
        "puzzle_id": record["puzzle_id"],
        "solved": finished and grid == reference,
        "valid_complete_board": finished and completed_error(board, clue) is None,
        "finished": finished, "failure": failure,
        "correct_cells": correct,
        "total_cells": len(reference) * len(reference[0]),
        "actions": len(history),
        "search_overrides": sum(1 for event in history
                                if event.get("branch_search")
                                and event["branch_search"]["override"]),
        "seconds": round(time.monotonic() - started, 3),
        "prediction": grid, "reference": reference, "history": history,
    }


def filter_only(puzzles, spec):
    """Restrict a puzzle list to the ids in ``spec`` (comma separated)."""
    wanted = [part.strip() for part in spec.split(",") if part.strip()]
    by_id = {record["puzzle_id"]: record for record in puzzles}
    missing = [pid for pid in wanted if pid not in by_id]
    if missing:
        raise SystemExit(f"--only: not in this split/sample: {missing}")
    return [by_id[pid] for pid in wanted]


def main():
    args = parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.dataset == "medium":
        source = [json.loads(line) for line in args.data.open(encoding="utf-8")]
        puzzles = [record for record in source if record.get("split") == "test_medium"]
        if not puzzles:
            raise SystemExit("No test_medium puzzles in the data file")
        split = "test_medium"
        sample_strategy = "full_set"
        if args.only:
            puzzles = filter_only(puzzles, args.only)
            sample_strategy = "explicit_ids"
        elif args.limit:
            puzzles = select_medium_sample(puzzles, args.limit)
            sample_strategy = "deterministic_board_size_stratified"
    else:
        source = [json.loads(line) for line in args.small_data.open(encoding="utf-8")]
        dev_ids = development_ids(record["puzzle_id"] for record in source)
        puzzles = [record for record in source if record["puzzle_id"] in dev_ids]
        puzzles.sort(key=lambda record: record["puzzle_id"])
        split = "internal_puzzle_holdout_from_212_training_puzzles"
        sample_strategy = "dev_holdout"
        if args.only:
            puzzles = filter_only(puzzles, args.only)
            sample_strategy = "explicit_ids"
        elif args.limit:
            puzzles = puzzles[:args.limit]

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.adapter, local_files_only=args.local_files_only)
    except (OSError, ValueError):
        # Older adapters may only contain weights/config; the base tokenizer is
        # the same one training used.
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # bf16 requires sm_80+.  `is_bf16_supported()` can return True on some T4
    # images, where bf16 then falls back to slow emulation.
    capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    bf16_ok = capability[0] >= 8 and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16_ok else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation="sdpa",
        local_files_only=args.local_files_only)
    model = PeftModel.from_pretrained(base, str(args.adapter)).to("cuda").eval()
    model.config.use_cache = True

    results = []
    policy = model_policy(model, tokenizer, args)
    for index, record in enumerate(puzzles, 1):
        result = rollout(policy, record, args)
        results.append(result)
        print(f"{args.dataset} evaluation: {index}/{len(puzzles)} "
              f"({result['puzzle_id']}: {'solved' if result['solved'] else result['failure']})",
              flush=True)

    report = {
        "run_id": args.adapter.parent.name if args.adapter.name == "adapter" else args.adapter.name,
        "dataset": args.dataset,
        "split": split,
        "sample_strategy": sample_strategy,
        "branch_search_depth": args.branch_search_depth,
        "attempted": len(results),
        "solved": sum(r["solved"] for r in results),
        "valid_complete_boards": sum(r["valid_complete_board"] for r in results),
        "correct_cells": sum(r["correct_cells"] for r in results),
        "total_cells": sum(r["total_cells"] for r in results),
        "failures": {reason: sum(r["failure"] == reason for r in results)
                     for reason in sorted({r["failure"] for r in results if r["failure"]})},
        "search_overrides": sum(r["search_overrides"] for r in results),
        "puzzles": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "puzzles"}
    summary["output"] = str(args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
