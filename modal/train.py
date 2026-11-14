"""Train a text action policy from positive, training-only solving traces.

Run from this repository: uv run modal run --detach modal/train.py --mode full
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))
from action_policy import (SYSTEM, choice_id, completion, development_ids,
                           legal_actions, prompt, split_examples)

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
VOLUME_PATH = "/mnt/train-tracks-solving-traces"  # Keep a POSIX string on Windows.
ROOT = Path(VOLUME_PATH)
REMOTE_TRACES = "/root/traces/train.jsonl"
app = modal.App("train-tracks-trace-action-policy")
volume = modal.Volume.from_name("train-tracks-solving-traces", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.14.0", "transformers==5.22.2",
                         "trl>=1.15.0,<2", "peft>=0.21.2,<1",
                         "datasets>=4.8.4,<5", "accelerate>=1.15.0,<2",
                         "huggingface-hub>=1.31.0,<2", "safetensors>=0.8.0,<1")
         .env({"HF_HUB_CACHE": f"{VOLUME_PATH}/hf-hub"})
         .add_local_file(str(HERE / "action_policy.py"), remote_path="/root/action_policy.py")
         .add_local_file(str(HERE.parent / "scripts/trace_logic.py"), remote_path="/root/trace_logic.py")
         .add_local_file(str(HERE.parent / "data/train.jsonl"), remote_path=REMOTE_TRACES))


def ensure_hf_token():
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
             or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    if not token:
        raise ValueError("Modal Secret hf-train-tracks needs a Hugging Face token")
    os.environ["HF_TOKEN"] = token


@app.function(image=image, gpu="RTX-PRO-6000", cpu=4, memory=24576,
              timeout=4 * 60 * 60, secrets=[modal.Secret.from_name("hf-train-tracks")],
              volumes={VOLUME_PATH: volume})
def train(mode: str = "full"):
    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be smoke or full")
    ensure_hf_token()
    set_seed(42)
    with open(REMOTE_TRACES, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    dev_ids = development_ids(r["puzzle_id"] for r in rows)
    train_rows = split_examples(r for r in rows if r["puzzle_id"] not in dev_ids)
    dev_rows = split_examples(r for r in rows if r["puzzle_id"] in dev_ids)
    if mode == "smoke":
        train_rows, dev_rows = train_rows[:16], dev_rows[:8]
    revision = HfApi().model_info(MODEL).sha
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def convert(items):
        examples = []
        menu_sizes = []
        for row in items:
            actions = legal_actions(row["state"], pending=row["action"]["type"] == "undo")
            target = choice_id(row["action"], actions)
            menu_sizes.append(len(actions))
            examples.append({
                "prompt": tokenizer.apply_chat_template([
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt(row["state"], actions)}],
                    tokenize=False, add_generation_prompt=True),
                "completion": completion(target) + tokenizer.eos_token,
            })
        return Dataset.from_list(examples), menu_sizes

    train_data, train_menu_sizes = convert(train_rows)
    dev_data, dev_menu_sizes = convert(dev_rows)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=revision, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    run_id = f"qwen25-trace-policy-{mode}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(run_dir / "trainer"), num_train_epochs=1,
            max_steps=4 if mode == "smoke" else -1,
            per_device_train_batch_size=2, gradient_accumulation_steps=8,
            per_device_eval_batch_size=2, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=2e-4, lr_scheduler_type="cosine",
            warmup_steps=1 if mode == "smoke" else 10,
            logging_steps=1 if mode == "smoke" else 10,
            eval_strategy="no", save_strategy="no", bf16=True, report_to="none",
            completion_only_loss=True, max_length=2048, seed=42, data_seed=42,
            dataloader_num_workers=0),
        train_dataset=train_data, eval_dataset=dev_data, processing_class=tokenizer,
        peft_config=LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]),
    )
    before = trainer.evaluate()["eval_loss"]
    outcome = trainer.train()
    after = trainer.evaluate()["eval_loss"]
    trainer.save_model(str(run_dir / "adapter"))
    tokenizer.save_pretrained(str(run_dir / "adapter"))
    report = {
        "run_id": run_id, "mode": mode, "model": MODEL, "model_revision": revision,
        "task": "trace_action_policy", "split": "internal_puzzle_holdout",
        "train_puzzles": 180, "development_puzzles": len(dev_ids),
        "train_examples": len(train_rows), "development_examples": len(dev_rows),
        "wrong_branches_excluded": True, "max_deductions_per_puzzle": 48,
        "initial_state_examples_kept": True,
        "deductions_sampled_by_rule": True,
        "legal_action_menu": True,
        "action_target": "menu_choice_id",
        "mean_train_menu_size": sum(train_menu_sizes) / len(train_menu_sizes),
        "max_train_menu_size": max(train_menu_sizes),
        "mean_development_menu_size": sum(dev_menu_sizes) / len(dev_menu_sizes),
        "optimizer_steps": trainer.state.global_step,
        "development_loss_before": before, "development_loss_after": after,
        "train_runtime_seconds": outcome.metrics["train_runtime"],
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "output_dir": str(run_dir),
    }
    (run_dir / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    print(json.dumps(report, indent=2), flush=True)
    return report


@app.local_entrypoint()
def main(mode: str = "full"):
    call = train.spawn(mode)
    print(f"Training call: {call.object_id}")
    print(call.get())
