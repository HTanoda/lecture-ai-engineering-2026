"""Run the supplied notebook in a fresh Python process for each condition."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, choices=[10, 20], required=True)
    args = parser.parse_args()
    work = ROOT / "runs" / f"epochs_{args.epochs}"
    if work.exists():
        raise SystemExit(f"Refusing to overwrite an existing run: {work}")
    work.mkdir(parents=True)
    original = ROOT / "notebooks" / "lora_format_tuning.original.ipynb"
    notebook = json.loads(original.read_text(encoding="utf-8"))
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    import transformers, peft, trl, accelerate, datasets, torch
    expected = {"transformers": "5.18.0", "peft": "0.21.1", "trl": "1.14.2",
                "accelerate": "1.15.0", "datasets": "5.1.0"}
    for name, version in expected.items():
        actual = sys.modules[name].__version__
        if actual != version:
            raise RuntimeError(f"{name}: expected {version}, got {actual}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required; refusing a CPU fallback.")
    from transformers import set_seed
    set_seed(42)
    env = {
        "__name__": "__notebook__",
        "transformers": transformers, "peft": peft, "trl": trl,
        "accelerate": accelerate, "datasets": datasets, "torch": torch,
        "display": lambda value: print(value.to_string() if hasattr(value, "to_string") else value, flush=True),
    }
    for i in range(2, 10):  # CONFIG through result.json; optional 4bit exercise is excluded.
        source = "".join(cells[i]["source"])
        if i == 2:
            source = source.replace('"num_train_epochs": 20,', f'"num_train_epochs": {args.epochs},')
            source = source.replace('"/content/qlora_lab"', repr(str(work)))
        if i == 6:
            source = source.replace(
                'trainer.train()  #',
                'torch.cuda.synchronize()\ntrain_start = time.perf_counter()\ntrain_output = trainer.train()\ntorch.cuda.synchronize()\ntraining_seconds = time.perf_counter() - train_start\n#'
            )
        print(f"=== Notebook code cell {i}: epochs={args.epochs} ===", flush=True)
        exec(compile(source, f"notebook_cell_{i}", "exec"), env)
    result_path = work / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    trainer = env["trainer"]
    loss_logs = [row for row in trainer.state.log_history if "loss" in row]
    if not loss_logs or trainer.state.global_step != args.epochs * 2:
        raise RuntimeError("Training did not complete the expected steps.")
    result["training"] = {
        "seconds": env["training_seconds"],
        "global_step": trainer.state.global_step,
        "epoch": trainer.state.epoch,
        "final_logged_loss": loss_logs[-1]["loss"],
        "final_logged_loss_step": loss_logs[-1]["step"],
        "mean_training_loss": env["train_output"].training_loss,
        "loss_log": loss_logs,
        "effective_args": {key: getattr(trainer.args, key) for key in (
            "fp16", "bf16", "gradient_checkpointing", "tf32",
            "per_device_train_batch_size", "gradient_accumulation_steps",
            "optim", "lr_scheduler_type", "warmup_steps", "seed")},
    }
    result["execution"] = {
        "operator": "Codex, at the user's request, on the user's local GPU",
        "platform": platform.platform(), "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
        "fresh_process": True,
        "source_url": "https://github.com/matsuolab/lecture-ai-engineering-2026/blob/main/notebooks/lora_format_tuning.ipynb",
        "source_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "changes": ["Local output/cache paths", "num_train_epochs for the 10-epoch run",
                    "Added timing and training-log capture; display converted to console output",
                    "Explicit seed 42 before model load in both runs"],
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (work / "training_args_effective.json").write_text(
        json.dumps(result["training"]["effective_args"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trainer.state.save_to_json(str(work / "trainer_state.json"))
    # Keep Windows-generated JSON evidence in the requested UTF-8/LF format.
    for path in work.iterdir():
        if path.is_file() and path.suffix in (".json", ".jsonl"):
            path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    print("COMPLETED:", result_path, flush=True)

if __name__ == "__main__":
    main()

