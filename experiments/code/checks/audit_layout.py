"""Check Python syntax and local Markdown links."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from layout import EXPERIMENTS, RESULTS
import ast
import json
import re
from urllib.parse import unquote


def main():
    repo = EXPERIMENTS.parent
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
    report = {'python_files_parsed': len(sources), 'local_links_checked': links, 'broken_links': broken}
    (RESULTS / 'checks' / 'layout_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    assert not broken, 'Broken internal links; see layout_audit.json'


if __name__ == '__main__':
    main()
