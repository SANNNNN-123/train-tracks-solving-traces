"""Train the branch-focused Qwen2.5-1.5B action policy on Kaggle T4 x2.

This is the Kaggle port of ``modal/train.py`` with three changes:

* no Modal dependency and local data paths;
* fp16 on T4 (bf16 only when the GPU supports it);
* branch-focused data augmentation from ``data_prep.py`` (both menu orders,
  corrected wrong-branch states, branch upsampling).

Run from the ``kaggle/`` directory:

    python train_kaggle.py --output /kaggle/working/run-branch

For 2-GPU DDP:

    accelerate launch --config_file configs/accelerate_t4x2.yaml train_kaggle.py ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from action_policy import SYSTEM  # noqa: E402
from data_prep import (build_dev_examples, build_examples, load_jsonl,
                       load_solutions, split_rows)  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "data/train.jsonl")
    parser.add_argument("--source", type=Path,
                        default=HERE / "data/train-puzzles-small.jsonl")
    parser.add_argument("--examples", type=Path, default=None,
                        help="Precomputed user/completion JSONL; skips --data/--source")
    parser.add_argument("--dev-examples", type=Path, default=None,
                        help="Precomputed dev JSONL paired with --examples")
    parser.add_argument("--init-adapter", type=Path, default=None,
                        help="Continue training this LoRA adapter instead of adding a new one")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/run-branch"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--branch-repeat", type=int, default=2)
    parser.add_argument("--no-branch-augment", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="16 train / 8 dev examples, 4 optimizer steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def load_model_and_tokenizer(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # With --init-adapter the adapter directory also holds the tokenizer that
    # rendered its training prompts, so the chat template matches exactly.
    tokenizer_source = args.model
    if args.init_adapter and (args.init_adapter / "tokenizer_config.json").exists():
        tokenizer_source = args.init_adapter
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, revision=None if tokenizer_source != args.model else args.revision,
        local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # bf16 requires sm_80+.  `is_bf16_supported()` can return True on some T4
    # images, where bf16 then falls back to slow emulation, so check the
    # capability explicitly.
    capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    bf16_ok = capability[0] >= 8 and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16_ok else torch.float16
    kwargs = dict(revision=args.revision, local_files_only=args.local_files_only,
                  attn_implementation="sdpa")
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, **kwargs)
    except TypeError:  # older transformers
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, **kwargs)
    if args.init_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.init_adapter), is_trainable=True)
        # Gradient checkpointing recomputes forward passes; some transformers/PEFT
        # combinations need the embedding output to require grad for LoRA grads.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.config.use_cache = False
    return model, tokenizer, dtype


def to_dataset(examples, tokenizer):
    from datasets import Dataset

    rows = []
    for example in examples:
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": example["user"]}],
            tokenize=False, add_generation_prompt=True)
        rows.append({
            "prompt": prompt,
            "completion": example["completion"] + tokenizer.eos_token,
            "kind": example["kind"],
        })
    return Dataset.from_list(rows)


def sft_config(args, output_dir, fp16, bf16):
    from trl import SFTConfig

    fields = set(SFTConfig.__dataclass_fields__)
    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.epochs,
        "max_steps": 4 if args.smoke else args.max_steps,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "per_device_eval_batch_size": max(1, args.batch_size),
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "learning_rate": args.lr,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 1 if args.smoke else 10,
        "logging_steps": 1 if args.smoke else 10,
        "eval_strategy": "no",
        "save_strategy": "no",
        "bf16": bf16,
        "fp16": fp16,
        "report_to": "none",
        "completion_only_loss": True,
        "packing": False,
        "max_length": args.max_length,
        "seed": args.seed,
        "data_seed": args.seed,
        "dataloader_num_workers": 2,
    }
    if "max_seq_length" in fields and "max_length" not in fields:
        kwargs["max_seq_length"] = kwargs.pop("max_length")
    if "eval_strategy" not in fields and "evaluation_strategy" in fields:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    kwargs = {key: value for key, value in kwargs.items() if key in fields}
    return SFTConfig(**kwargs)


def pin_single_gpu_if_unlaunched():
    """Keep TRL's SFTTrainer out of ``nn.DataParallel``.

    A single process that can see several GPUs makes ``Trainer`` wrap the model
    in ``nn.DataParallel``, and ``SFTTrainer`` refuses that outright.  Launchers
    (``accelerate launch``, ``torchrun``) set ``WORLD_SIZE`` and assign devices
    themselves; for a plain run we pin the process to GPU 0.

    This has to happen before the first CUDA call: ``cudaGetDeviceCount``
    caches the visible-device list, so setting the variable afterwards is too
    late.  That is why there is no device query here.
    """
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        return
    print("single-process run: pinning to GPU 0 "
          "(SFTTrainer cannot use nn.DataParallel; use accelerate launch for 2 GPUs)",
          flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    args = parse_args()
    if args.branch_repeat < 1:
        raise SystemExit("--branch-repeat must be at least 1")
    pin_single_gpu_if_unlaunched()
    import torch
    from peft import LoraConfig
    from transformers import set_seed
    from trl import SFTTrainer

    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    if args.examples:
        train_examples = load_jsonl(args.examples)
        dev_examples = load_jsonl(args.dev_examples) if args.dev_examples else []
        train_puzzles = len({example.get("puzzle_id") for example in train_examples})
        dev_puzzles = len({example.get("puzzle_id") for example in dev_examples})
        train_rows = []
        stats = {"source": "precomputed", "total": len(train_examples),
                 "puzzles": train_puzzles}
        print(f"precomputed examples: {len(train_examples):,} train, "
              f"{len(dev_examples):,} dev")
    else:
        rows = load_jsonl(args.data)
        solutions = load_solutions(args.source)
        train_rows, dev_rows, dev_puzzles = split_rows(rows)
        if args.no_branch_augment:
            train_examples, stats = build_examples(
                train_rows, solutions, branch_repeat=1,
                reverse_branch_orders=False, corrected_wrong_branches=False)
        else:
            train_examples, stats = build_examples(
                train_rows, solutions, branch_repeat=args.branch_repeat)
        dev_examples = build_dev_examples(dev_rows)
        train_puzzles = len({row["puzzle_id"] for row in train_rows})
    if args.smoke:
        train_examples = train_examples[:16]
        dev_examples = dev_examples[:8]

    model, tokenizer, dtype = load_model_and_tokenizer(args)
    train_data = to_dataset(train_examples, tokenizer)
    dev_data = to_dataset(dev_examples, tokenizer)
    print(f"train examples: {len(train_data):,}  dev examples: {len(dev_data):,}")
    print(f"dtype: {dtype}  gpus: {torch.cuda.device_count()}  "
          f"gpu0: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    fp16 = dtype == torch.float16
    bf16 = dtype == torch.bfloat16
    peft_config = None if args.init_adapter else LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    trainer = SFTTrainer(
        model=model,
        args=sft_config(args, args.output / "trainer", fp16=fp16, bf16=bf16),
        train_dataset=train_data,
        eval_dataset=dev_data,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    started = time.monotonic()
    before = trainer.evaluate().get("eval_loss")
    trainer.train()
    after = trainer.evaluate().get("eval_loss")
    adapter = args.output / "adapter"

    # Under DDP every process would otherwise write the same files concurrently.
    if trainer.is_world_process_zero():
        trainer.save_model(str(adapter))
        tokenizer.save_pretrained(adapter)
        report = {
            "run_id": args.output.name,
            "model": args.model,
            "task": "trace_action_policy",
            "branch_augment": not args.no_branch_augment,
            "branch_repeat": 1 if args.no_branch_augment else args.branch_repeat,
            "puzzles_train": train_puzzles,
            "puzzles_dev": dev_puzzles,
            "train_examples": len(train_data),
            "dev_examples": len(dev_data),
            "data_stats": stats,
            "examples_file": str(args.examples) if args.examples else None,
            "init_adapter": str(args.init_adapter) if args.init_adapter else None,
            "continued_training": bool(args.init_adapter),
            "optimizer_steps": trainer.state.global_step,
            "development_loss_before": before,
            "development_loss_after": after,
            "train_runtime_seconds": round(time.monotonic() - started, 1),
            "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "dtype": str(dtype),
            "gpus": torch.cuda.device_count(),
            "adapter": str(adapter),
            "output_dir": str(args.output),
        }
        (args.output / "training.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
