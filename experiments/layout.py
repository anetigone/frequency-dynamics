"""Repository paths shared by directly executable experiment scripts."""
import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parent
RESULTS = EXPERIMENTS / 'results'
for group in ('theory', 'shared', 'pilot', 'population', 'empirical', 'cifar', 'checks'):
    sys.path.insert(0, str(EXPERIMENTS / 'code' / group))


def result_path(name):
    """Find an existing result by name; new studies pass their output explicitly."""
    matches = [p for p in RESULTS.glob(f'*/{name}') if p.exists()]
    archived = RESULTS / 'archive' / 'development' / str(name)
    if archived.exists():
        matches.append(archived)
    if len(matches) == 1:
        return matches[0]
    if str(name) == 'textures_pilot' and not matches:
        return RESULTS / 'pilot' / str(name)
    raise KeyError(f'Result name is missing or ambiguous: {name}; pass an explicit path')


def source_path(name):
    """Find a script directly in code/ without a registry."""
    matches = [p for p in (EXPERIMENTS / 'code').glob('*/*.py') if p.name == str(name)]
    if len(matches) != 1:
        raise KeyError(f'Script name is missing or ambiguous: {name}')
    return matches[0]
