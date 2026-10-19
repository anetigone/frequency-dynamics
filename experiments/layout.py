"""Repository paths shared by directly executable experiment scripts."""
import json
import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parent
RESULTS = EXPERIMENTS / 'results'
MANIFEST = json.loads((EXPERIMENTS / 'layout_manifest.json').read_text(encoding='utf-8'))
for group in ('theory', 'shared', 'pilot', 'population', 'empirical', 'cifar', 'checks'):
    sys.path.insert(0, str(EXPERIMENTS / 'code' / group))


def result_path(name):
    """Resolve a saved study name, including archived development studies."""
    group = MANIFEST['result_groups'].get(str(name))
    if str(name) == 'textures_pilot':
        group = 'pilot'
    if group is None:
        raise KeyError(f'Unknown saved study: {name}; pass an explicit output directory for a new study')
    return RESULTS / group / str(name)


def source_path(name):
    """Resolve CURRENT source; historical protocol hashes use the archive."""
    return EXPERIMENTS / MANIFEST['source_files'][str(name)]['current_path']
