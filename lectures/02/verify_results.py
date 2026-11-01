"""Check that the comparison uses identical data/configuration and real recorded outputs."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
results = {}
for epochs in (20, 10):
    run = ROOT / "runs" / f"epochs_{epochs}"
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    assert result["executed_training_myself"] is True
    assert result["config"]["num_train_epochs"] == epochs
    assert result["training"]["global_step"] == epochs * 2
    assert result["training"]["epoch"] == epochs
    assert result["training"]["final_logged_loss_step"] == epochs * 2
    assert math.isfinite(result["training"]["final_logged_loss"])
    assert result["training"]["seconds"] > 0
    test = [json.loads(line) for line in (run / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    gold = json.loads((run / "gold.json").read_text(encoding="utf-8"))
    train = [json.loads(line) for line in (run / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(train) == 14 and len(test) == len(gold) == 21
    for phase in ("before", "after"):
        evaluation = json.loads((run / f"eval_{phase}.json").read_text(encoding="utf-8"))
        rows = evaluation["rows"]
        assert len(rows) == 21
        assert [row["question"] for row in rows] == [row["question"] for row in test]
        for key, flag in (("format_rate", "format_ok"), ("content_rate", "content_ok"), ("both_rate", "both_ok")):
            measured = sum(row[flag] for row in rows) / 21
            assert measured == evaluation["summary"][key] == result[phase][key]
        assert all(isinstance(row["raw"], str) and row["raw"] for row in rows)
    results[epochs] = result
config_differences = {key for key in results[20]["config"] if results[20]["config"][key] != results[10]["config"][key]}
assert config_differences == {"num_train_epochs"}, config_differences
assert results[20]["versions"] == results[10]["versions"]
assert results[20]["training"]["effective_args"] == results[10]["training"]["effective_args"]
for name in ("train.jsonl", "test.jsonl", "gold.json"):
    blobs = [(ROOT / "runs" / f"epochs_{n}" / name).read_bytes() for n in (20, 10)]
    assert blobs[0] == blobs[1], name
before = [json.loads((ROOT / "runs" / f"epochs_{n}" / "eval_before.json").read_text(encoding="utf-8")) for n in (20, 10)]
assert before[0]["rows"] == before[1]["rows"], "Fresh-model baseline outputs differ; investigate."
files = list((ROOT / "runs").glob("epochs_*/*.json")) + list((ROOT / "runs").glob("epochs_*/*.jsonl"))
manifest = {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)}
(ROOT / "verified_sha256.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
print("Verified: 2 fresh-process runs; only epochs differ; identical data and baseline outputs; 21 outputs per evaluation; metrics and step counts match.")
for epochs, result in results.items():
    print(epochs, result["after"], result["training"])

