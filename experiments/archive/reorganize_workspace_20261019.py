"""One-time, workspace-contained migration; preserves original sources and result bytes."""
import ast
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
REPO = BASE.parent
CODE = {
    'theory': ['interval_core', 'verify', 'check_arithmetic', 'mechanism_checks', 'representation_checks'],
    'shared': ['square_images'],
    'pilot': ['train_textures', 'texture_controls', 'verify_spatial', 'plot_image_pilots', 'recheck_interventions'],
    'population': ['subspace_mechanism', 'empirical_subspace_interventions', 'summarize_subspaces', 'summarize_empirical_subspaces'],
    'empirical': ['full_empirical', 'sgd_state_tools', 'batch_drift_diagnostic', 'conditional_moment_diagnostic', 'intervene_full_sgd', 'summarize_full_interventions', 'summarize_full_sgd', 'plot_batch_drift', 'compare_empirical_snapshots'],
    'cifar': ['audit_cifar', 'cifar_local', 'evaluate_cifar_local', 'summarize_cifar_local'],
    'checks': ['check_image_pipeline', 'check_subspace_mechanism', 'check_full_empirical', 'check_trained_sgd', 'audit_local_closeout'],
}
DOCS = ['IMAGE_EXPERIMENTS.md', 'PILOT_FINDINGS.md', 'SUBSPACE_MECHANISM.md', 'FULL_EMPIRICAL_PROTOCOL.md', 'LOCAL_CLOSEOUT.md']
FORMAL = {
    'population': ['subspace_confirm_k1', 'subspace_confirm_k2', 'subspace_confirm_local', 'subspace_summary', 'subspace_empirical_confirm'],
    'empirical': ['full_sgd_main', 'full_sgd_local', 'full_sgd_repeat1', 'full_sgd_repeat2', 'full_sgd_interventions', 'batch_drift_diagnostic'],
    'cifar': ['cifar_local_five', 'cifar_audit.json'],
    'checks': ['empirical_existing_population_comparison.json', 'full_empirical_checks.json', 'local_closeout_audit.json', 'subspace_checks.json', 'trained_sgd_image_checks.json'],
    'pilot': ['figures', 'spatial', 'texture_controls', 'texture_controls_balanced', 'texture_controls_fine', 'textures_balanced_empirical_pilot', 'textures_balanced_population_pilot', 'textures_empirical_pilot', 'textures_population_pilot'],
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe(path):
    path = path.resolve()
    assert path.is_relative_to(REPO), path
    return path


def move(src, dst):
    safe(src); safe(dst)
    assert not dst.exists(), dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def main():
    if BASE.name != 'experiments':
        raise SystemExit('Historical migration record; use experiments/run.py audit_layout to inspect the completed layout')
    archive = BASE / 'archive' / 'sources_before_layout_20261019'
    assert not (BASE / 'layout_manifest.json').exists(), 'Migration already completed'
    archive.mkdir(parents=True, exist_ok=True)
    sources = {f'{name}.py': f'code/{group}/{name}.py' for group, names in CODE.items() for name in names}
    actual = {p.name for p in BASE.glob('*.py')} - {Path(__file__).name}
    assert actual == set(sources), (actual - set(sources), set(sources) - actual)
    source_manifest = {}
    for name, new in sources.items():
        if (archive / name).exists():
            shutil.copyfile(archive / name, BASE / name)
        else:
            shutil.copyfile(BASE / name, archive / name)
        source_manifest[name] = {'original_sha256': digest(BASE / name), 'current_path': new}
    result_groups = {name: group for group, names in FORMAL.items() for name in names}
    for p in (BASE / 'results').iterdir():
        result_groups.setdefault(p.name, 'archive/development')
    moves = {f'experiments/{old}': f'experiments/{new}' for old, new in sources.items()}
    moves.update({f'experiments/{name}': f'experiments/docs/{name}' for name in DOCS})
    for name, group in result_groups.items():
        moves[f'experiments/results/{name}'] = f'experiments/results/{group}/{name}'
    for name in ['certificate.json', 'mechanism_results.json', 'mechanism_profiles.csv', 'representation_results.json']:
        moves[f'experiments/{name}'] = f'experiments/results/theory/{name}'
        result_groups[name] = 'theory'
    result_inventory = {}
    for p in (BASE / 'results').rglob('*'):
        if p.is_file():
            rel = p.relative_to(REPO).as_posix()
            name = p.relative_to(BASE / 'results').parts[0]
            new = rel.replace(f'experiments/results/{name}', moves[f'experiments/results/{name}'], 1)
            result_inventory[new] = digest(p)
    for name in ['certificate.json', 'mechanism_results.json', 'mechanism_profiles.csv', 'representation_results.json']:
        result_inventory[moves[f'experiments/{name}']] = digest(BASE / name)
    # Rewrite only paths/import bootstrapping, never the mathematical update rules.
    for name, destination in sources.items():
        p = BASE / name
        s = p.read_text(encoding='utf-8')
        s = s.replace('ROOT = Path(__file__).resolve().parent', 'ROOT = EXPERIMENTS')
        s = s.replace('ROOT = Path(__file__).parent', 'ROOT = EXPERIMENTS')
        if name in ['mechanism_checks.py', 'representation_checks.py']:
            s = s.replace('ROOT = EXPERIMENTS', 'ROOT = RESULTS / "theory"')
        s = s.replace('root = Path(__file__).resolve().parent.parent', 'root = EXPERIMENTS.parent')
        s = s.replace('(ROOT / name).read_bytes()', 'source_path(name).read_bytes()')
        s = s.replace('(ROOT / filename).read_bytes()', 'source_path(filename).read_bytes()')
        # Rewrite the common path prefix while preserving dynamically selected names.
        s = re.sub(r'(?:ROOT|Path\(__file__\)\.parent) / "results" / (f"[^"]+"|"[^"]+"|[a-zA-Z_][a-zA-Z_0-9]*)', r'result_path(\1)', s)
        s = s.replace('root / "experiments" / "results" / "cifar_audit.json"', 'result_path("cifar_audit.json")')
        s = s.replace('ROOT / "results"', 'RESULTS')
        s = s.replace('Path(__file__).parent / "results"', 'RESULTS')
        s = s.replace("Path(__file__).resolve().parent/'certificate.json'", "result_path('certificate.json')")
        if name == 'summarize_subspaces.py':
            s = s.replace('default=RESULTS)', 'default=RESULTS / "population")')
        if name == 'plot_image_pilots.py':
            s = s.replace('default=RESULTS)', 'default=RESULTS / "pilot")')
        # Future filenames and examples use the new code paths.
        for old, new in sorted(moves.items(), key=lambda x: -len(x[0])):
            if old.startswith('experiments/') and old.endswith('.py'):
                s = s.replace(old, new)
        tree = ast.parse(s)
        insert_line = tree.body[0].end_lineno if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str) else 0
        bootstrap = ('\n# Shared repository paths; supports direct execution from any working directory.\n'
                     'import sys as _sys\nfrom pathlib import Path as _LayoutPath\n'
                     '_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2]))\n'
                     'from layout import EXPERIMENTS, RESULTS, result_path, source_path\n')
        lines = s.splitlines(keepends=True)
        lines.insert(insert_line, bootstrap)
        p.write_text(''.join(lines), encoding='utf-8')
    # Resolve document links against their ORIGINAL location, then map to the new one.
    markdown = list(REPO.glob('*.md')) + list(BASE.glob('*.md')) + list((BASE / 'results').rglob('*.md'))
    def mapped(rel):
        for old, new in sorted(moves.items(), key=lambda x: -len(x[0])):
            if rel == old or rel.startswith(old + '/'):
                return new + rel[len(old):]
        return rel
    for p in markdown:
        old_rel = p.relative_to(REPO).as_posix()
        new_rel = mapped(old_rel)
        s = p.read_text(encoding='utf-8')
        def link(match):
            target = match.group(2)
            if '://' in target or target.startswith('#') or target.startswith('mailto:'):
                return match.group(0)
            clean, sep, fragment = target.partition('#')
            absolute = (p.parent / clean).resolve()
            if not absolute.is_relative_to(REPO):
                return match.group(0)
            dest = REPO / mapped(absolute.relative_to(REPO).as_posix())
            relative = Path(os.path.relpath(dest, (REPO / new_rel).parent)).as_posix()
            return match.group(1) + '(' + relative + (sep + fragment if sep else '') + ')'
        s = re.sub(r'(!?\[[^\]]*\])\(([^)]+)\)', link, s)
        # Root-relative command examples and backtick paths.
        for old, new in sorted(moves.items(), key=lambda x: -len(x[0])):
            s = re.sub(re.escape(old) + r'(?![A-Za-z0-9_])', lambda m: new, s)
        p.write_text(s, encoding='utf-8')
    for old, new in moves.items():
        move(REPO / old, REPO / new)
    for name, entry in source_manifest.items():
        entry['current_sha256'] = digest(BASE / entry['current_path'])
    manifest = {'date': '2026-10-19', 'moves': moves, 'source_files': source_manifest,
                'result_groups': result_groups, 'original_result_sha256': result_inventory,
                'notes': 'Original JSON/CSV/PT/PNG payloads unchanged; Markdown paths updated. Locked runs retain original protocols. Original source snapshots verify their historical hashes.'}
    (BASE / 'layout_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    # Markdown result summaries change only links; all numerical payloads must match.
    checked = 0
    for rel, expected in result_inventory.items():
        p = REPO / rel
        if p.suffix != '.md':
            assert digest(p) == expected, rel
            checked += 1
    print(json.dumps({'moved_entries': len(moves), 'sources_archived': len(sources), 'unchanged_result_files': checked}, indent=2))


if __name__ == '__main__':
    main()
