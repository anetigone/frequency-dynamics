"""Check moved payloads, source provenance, Python syntax and local Markdown links."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from layout import EXPERIMENTS, RESULTS, MANIFEST
import ast
import hashlib
import json
import re
from urllib.parse import unquote


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    repo = EXPERIMENTS.parent
    count = 0
    for rel, expected in MANIFEST['original_result_sha256'].items():
        path = repo / rel
        assert path.is_file(), rel
        if path.suffix != '.md':
            assert sha256(path) == expected, f'Original result changed: {rel}'
            count += 1
    originals = 0
    for name, entry in MANIFEST['source_files'].items():
        assert sha256(EXPERIMENTS / entry['current_path']) == entry['current_sha256'], name
        if entry.get('original_sha256'):
            original = EXPERIMENTS / 'archive' / 'sources_before_layout_20261019' / name
            assert sha256(original) == entry['original_sha256'], name
            originals += 1
    for destination in MANIFEST['moves'].values():
        assert (repo / destination).exists(), destination
    for relative, expected in MANIFEST.get('support_sha256', {}).items():
        assert sha256(EXPERIMENTS / relative) == expected, relative
    sources = list((EXPERIMENTS / 'code').rglob('*.py')) + [EXPERIMENTS / 'layout.py', EXPERIMENTS / 'run.py']
    for path in sources:
        ast.parse(path.read_text(encoding='utf-8'))
    documents = list(repo.glob('*.md')) + list(EXPERIMENTS.rglob('*.md'))
    broken, links = [], 0
    for document in documents:
        if 'sources_before_layout_20261019' in document.parts:
            continue
        for match in re.finditer(r'!?\[[^\]]*\]\(([^)]+)\)', document.read_text(encoding='utf-8')):
            target = match.group(1).strip('<>')
            if '://' in target or target.startswith(('#', 'mailto:')):
                continue
            target = unquote(target.split('#', 1)[0])
            if not target:
                continue
            links += 1
            if not (document.parent / target).exists():
                broken.append({'document': document.relative_to(repo).as_posix(), 'target': target})
    report = {'date': '2026-10-19', 'original_result_files_unchanged': count,
              'original_sources_verified': originals, 'current_sources_verified': len(MANIFEST['source_files']),
              'python_files_parsed': len(sources), 'local_links_checked': links, 'broken_links': broken}
    (RESULTS / 'checks' / 'layout_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    assert not broken, 'Broken internal links; see layout_audit.json'


if __name__ == '__main__':
    main()
