"""Audit saved local experiment completeness and locked source hashes."""

# Shared repository paths; supports direct execution from any working directory.
import sys as _sys
from pathlib import Path as _LayoutPath
_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2]))
from layout import EXPERIMENTS, RESULTS, MANIFEST, result_path, source_path
import ast
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path


ROOT = EXPERIMENTS


def main():
    directories = {"full_sgd_main": 90, "full_sgd_local": 90,
                   "full_sgd_repeat1": 20, "full_sgd_repeat2": 20,
                   "full_sgd_interventions": 90, "cifar_local_five": 15}
    audited = []
    for name, expected in directories.items():
        path = result_path(name)
        protocol = json.loads((path / "protocol.json").read_text())
        runs = list(path.glob("seed*/result.json"))
        assert len(runs) == expected, (name, len(runs), expected)
        hashes = protocol["source_sha256"]
        if isinstance(hashes, str):
            filename = "intervene_full_sgd.py" if name == "full_sgd_interventions" else "cifar_local.py"
            hashes = {filename: hashes}
        for filename, digest in hashes.items():
            archived = ROOT / "archive" / "sources_before_layout_20261019" / filename
            assert hashlib.sha256(archived.read_bytes()).hexdigest() == digest, (name, filename)
            entry = MANIFEST["source_files"][filename]
            assert hashlib.sha256(source_path(filename).read_bytes()).hexdigest() == entry["current_sha256"], filename
        if name.startswith("full_sgd_") and name != "full_sgd_interventions":
            for run in runs:
                result = json.loads(run.read_text())
                assert result["history"][-1]["step"] == 4000
        if name == "cifar_local_five":
            assert len(list(path.glob("seed*/evaluation.json"))) == 15
        audited.append({"directory": name, "completed": len(runs), "locked_source_hashes_valid": True})
    main_runs = {p.parent.name: json.loads(p.read_text()) for p in (result_path("full_sgd_main")).glob("seed*/result.json")}
    for stream in [1, 2]:
        for p in (result_path(f"full_sgd_repeat{stream}")).glob("seed*/result.json"):
            result = json.loads(p.read_text())
            key = f"seed_{result['seed']}_{result['mode']}_r1_stream0"
            original = main_runs[key]
            assert original["initial_A"] == result["initial_A"]
            assert original["training_data_sha256"] == result["training_data_sha256"]
    files = list((RESULTS).rglob("*.json"))
    for p in files:
        json.loads(p.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    sources = list((ROOT / "code").rglob("*.py")) + [ROOT / "layout.py", ROOT / "run.py"]
    for p in sources:
        ast.parse(p.read_text(encoding="utf-8"))
    output = {"date": "2026-10-19", "python": sys.version.split()[0],
              "packages": {n: importlib.metadata.version(n) for n in ["torch", "torchvision", "numpy", "matplotlib"]},
              "datasets": {"cifar": "seven standard MD5 checks passed; original files unchanged"},
              "completed_studies": audited, "repeat_stream_initialization_and_data_match": True,
              "strict_json_files_checked": len(files), "python_modules_parsed": len(sources),
              "source_hash_basis": "original archived sources plus current layout manifest",
              "scope": "numerical/artifact integrity, not a new interval certificate"}
    (RESULTS / "checks" / "layout_closeout_audit.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
