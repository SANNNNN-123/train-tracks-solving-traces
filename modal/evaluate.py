"""Interactive, model-owned rollout on held-out puzzles.

Run: uv run modal run --detach modal/evaluate.py --run-id RUN_ID --dataset medium --limit-puzzles 32
"""

import json
import hashlib
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))
from action_policy import (SYSTEM, development_ids, legal_actions, parse_action,
                           prompt, validate_action)

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
VOLUME_PATH = "/mnt/train-tracks-solving-traces"
ROOT = Path(VOLUME_PATH)
REMOTE_SOURCES = {
    "small": "/root/traces/train-puzzles-small.jsonl",
    "medium": "/root/traces/train-puzzles-medium.jsonl",
}
app = modal.App("train-tracks-trace-action-evaluate")
volume = modal.Volume.from_name("train-tracks-solving-traces", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.14.0", "transformers==5.22.2",
                         "peft>=0.21.2,<1", "accelerate>=1.15.0,<2",
                         "huggingface-hub>=1.31.0,<2", "safetensors>=0.8.0,<1")
         .env({"HF_HUB_CACHE": f"{VOLUME_PATH}/hf-hub"})
         .add_local_file(str(HERE / "action_policy.py"), remote_path="/root/action_policy.py")
         .add_local_file(str(HERE.parent / "scripts/trace_logic.py"), remote_path="/root/trace_logic.py")
         .add_local_file(str(HERE.parent / "source/train-puzzles.jsonl"),
                         remote_path=REMOTE_SOURCES["small"])
         .add_local_file(str(HERE.parent / "source/train-puzzles-medium.jsonl"),
                         remote_path=REMOTE_SOURCES["medium"]))


def ensure_hf_token():
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
             or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    if not token:
        raise ValueError("Modal Secret hf-train-tracks needs a Hugging Face token")
    os.environ["HF_TOKEN"] = token


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
    """Choose a stable sample balanced across medium board dimensions."""
    groups = defaultdict(list)
    for puzzle in puzzles:
        shape = (len(puzzle["puzzle"]["rows"]), len(puzzle["puzzle"]["columns"]))
        groups[shape].append(puzzle)
    shapes = sorted(groups)
    for group in groups.values():
        group.sort(key=lambda row: hashlib.sha256(row["puzzle_id"].encode()).hexdigest())

    target = min(limit, sum(map(len, groups.values())))
    selected_by_shape = {shape: [] for shape in shapes}
    while sum(map(len, selected_by_shape.values())) < target:
        eligible = [shape for shape in shapes
                    if len(selected_by_shape[shape]) < len(groups[shape])]
        shape = min(eligible, key=lambda key: (len(selected_by_shape[key]), key))
        selected_by_shape[shape].append(groups[shape][len(selected_by_shape[shape])])
    selected = [puzzle for group in selected_by_shape.values() for puzzle in group]
    return sorted(selected, key=lambda row: row["puzzle_id"])


def rollout(model, tokenizer, record, max_actions):
    import torch
    from trace_logic import BIT, clue_from_record, completed_error, initial_domains, one, snapshot

    clue = clue_from_record(record)
    board = initial_domains(clue)
    stack, history = [], []
    pending = False
    failure = None
    started = time.monotonic()
    for step_index in range(max_actions):
        state = state_for(board, clue, stack)
        actions = legal_actions(state, pending=pending)
        if not actions:
            failure = "no_legal_actions"
            break
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt(state, actions)}]
        encoded = tokenizer.apply_chat_template(messages, tokenize=True,
                                                add_generation_prompt=True,
                                                return_dict=True,
                                                return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=160, do_sample=False,
                                       pad_token_id=tokenizer.eos_token_id)
        raw = tokenizer.decode(generated[0, encoded["input_ids"].shape[-1]:], skip_special_tokens=True)
        action = parse_action(raw, actions)
        if action is None:
            valid, reason, canonical = False, "invalid_choice", None
        else:
            valid, reason, canonical = validate_action(board, clue, action, stack, pending)
        event = {"step": step_index, "state": state, "raw": raw,
                 "action": action, "legal_action_count": len(actions),
                 "valid": valid, "reason": reason}
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
    finished = bool(history and history[-1]["valid"] and
                    history[-1]["action"]["type"] == "finish")
    return {
        "puzzle_id": record["puzzle_id"], "solved": finished and grid == reference,
        "valid_complete_board": finished and completed_error(board, clue) is None,
        "finished": finished, "failure": failure,
        "correct_cells": correct, "total_cells": len(reference) * len(reference[0]),
        "actions": len(history), "seconds": round(time.monotonic() - started, 3),
        "prediction": grid, "reference": reference, "history": history,
    }


@app.function(image=image, gpu="RTX-PRO-6000", cpu=4, memory=24576,
              timeout=24 * 60 * 60, secrets=[modal.Secret.from_name("hf-train-tracks")],
              volumes={VOLUME_PATH: volume})
def evaluate(run_id: str, dataset: str = "small", limit_puzzles: int = 8,
             max_actions: int = 256):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not run_id or "/" in run_id or "\\" in run_id or run_id in {".", ".."}:
        raise ValueError("Pass a training run ID")
    if dataset not in REMOTE_SOURCES:
        raise ValueError("dataset must be small or medium")
    if limit_puzzles < 0 or max_actions < 1:
        raise ValueError("limit_puzzles must be nonnegative and max_actions positive")
    ensure_hf_token()
    volume.reload()
    run_dir = ROOT / "runs" / run_id
    training = json.loads((run_dir / "training.json").read_text())
    if training["task"] != "trace_action_policy" or training["model"] != MODEL:
        raise ValueError("Run is not a trace action policy")
    adapter = run_dir / "adapter"
    if not (adapter / "adapter_config.json").exists():
        raise FileNotFoundError(adapter)
    with open(REMOTE_SOURCES[dataset], encoding="utf-8") as handle:
        source = [json.loads(line) for line in handle]
    if dataset == "small":
        dev_ids = development_ids(record["puzzle_id"] for record in source)
        puzzles = [r for r in source if r["puzzle_id"] in dev_ids]
        split = "internal_puzzle_holdout_from_212_training_puzzles"
    else:
        puzzles = [r for r in source if r.get("split") == "test_medium"]
        if not puzzles:
            raise ValueError("Medium source has no test_medium puzzles")
        split = "test_medium"
    puzzles.sort(key=lambda r: r["puzzle_id"])
    sample_strategy = "full_set"
    if limit_puzzles and dataset == "medium":
        puzzles = select_medium_sample(puzzles, limit_puzzles)
        sample_strategy = "deterministic_board_size_stratified"
    elif limit_puzzles:
        puzzles = puzzles[:limit_puzzles]
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=training["model_revision"], dtype=torch.bfloat16,
        attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, adapter).to("cuda").eval()
    model.config.use_cache = True
    results = []
    for record in puzzles:
        results.append(rollout(model, tokenizer, record, max_actions))
        print(f"{dataset} evaluation: {len(results)}/{len(puzzles)} puzzles", flush=True)
    board_shapes = Counter(f"{len(r['puzzle']['rows'])}x{len(r['puzzle']['columns'])}"
                           for r in puzzles)
    report = {
        "run_id": run_id, "dataset": dataset, "split": split,
        "sample_strategy": sample_strategy,
        "board_shapes": dict(sorted(board_shapes.items())),
        "attempted": len(results),
        "solved": sum(r["solved"] for r in results),
        "valid_complete_boards": sum(r["valid_complete_board"] for r in results),
        "correct_cells": sum(r["correct_cells"] for r in results),
        "total_cells": sum(r["total_cells"] for r in results),
        "failures": {reason: sum(r["failure"] == reason for r in results)
                     for reason in sorted({r["failure"] for r in results if r["failure"]})},
        "puzzles": results,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = "development" if dataset == "small" else "test-medium"
    output = ROOT / "rollouts" / f"{prefix}-{run_id}-{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    volume.commit()
    summary = {k: v for k, v in report.items() if k != "puzzles"}
    summary["output"] = str(output)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def main(run_id: str, dataset: str = "small", limit_puzzles: int = 8,
         max_actions: int = 256):
    call = evaluate.spawn(run_id, dataset, limit_puzzles, max_actions)
    print(f"Evaluation call: {call.object_id}")
    print(call.get())
