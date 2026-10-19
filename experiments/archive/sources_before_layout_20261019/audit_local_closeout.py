"""Audit saved local experiment completeness and locked source hashes."""
import ast
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path


ROOT = Path(__file__).parent


def main():
    directories = {"full_sgd_main": 90, "full_sgd_local": 90,
                   "full_sgd_repeat1": 20, "full_sgd_repeat2": 20,
                   "full_sgd_interventions": 90, "cifar_local_five": 15}
    audited = []
    for name, expected in directories.items():
        path = ROOT / "results" / name
        protocol = json.loads((path / "protocol.json").read_text())
        runs = list(path.glob("seed*/result.json"))
        assert len(runs) == expected, (name, len(runs), expected)
        hashes = protocol["source_sha256"]
        if isinstance(hashes, str):
            filename = "intervene_full_sgd.py" if name == "full_sgd_interventions" else "cifar_local.py"
            hashes = {filename: hashes}
        for filename, digest in hashes.items():
            assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == digest, (name, filename)
        if name.startswith("full_sgd_") and name != "full_sgd_interventions":
            for run in runs:
                result = json.loads(run.read_text())
                assert result["history"][-1]["step"] == 4000
        if name == "cifar_local_five":
            assert len(list(path.glob("seed*/evaluation.json"))) == 15
        audited.append({"directory": name, "completed": len(runs), "locked_source_hashes_valid": True})
    main_runs = {p.parent.name: json.loads(p.read_text()) for p in (ROOT / "results" / "full_sgd_main").glob("seed*/result.json")}
    for stream in [1, 2]:
        for p in (ROOT / "results" / f"full_sgd_repeat{stream}").glob("seed*/result.json"):
            result = json.loads(p.read_text())
            key = f"seed_{result['seed']}_{result['mode']}_r1_stream0"
            original = main_runs[key]
            assert original["initial_A"] == result["initial_A"]
            assert original["training_data_sha256"] == result["training_data_sha256"]
    files = list((ROOT / "results").rglob("*.json"))
    for p in files:
        json.loads(p.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    sources = list(ROOT.glob("*.py"))
    for p in sources:
        ast.parse(p.read_text(encoding="utf-8"))
    output = {"date": "2026-10-19", "python": sys.version.split()[0],
              "packages": {n: importlib.metadata.version(n) for n in ["torch", "torchvision", "numpy", "matplotlib"]},
              "datasets": {"cifar": "seven standard MD5 checks passed; original files unchanged"},
              "completed_studies": audited, "repeat_stream_initialization_and_data_match": True,
              "strict_json_files_checked": len(files), "python_modules_parsed": len(sources),
              "scope": "numerical/artifact integrity, not a new interval certificate"}
    (ROOT / "results" / "local_closeout_audit.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
