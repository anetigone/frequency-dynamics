"""Repository paths shared by directly executable experiment scripts."""
import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parent
RESULTS = EXPERIMENTS / 'results'
for group in ('theory', 'shared', 'pilot', 'population', 'empirical', 'cifar', 'checks'):
    sys.path.insert(0, str(EXPERIMENTS / 'code' / group))


def result_path(name, *, default=None):
    """Find a unique existing result, including the legacy flat layout.

    An explicit default is used only when no result exists, never on ambiguity.
    """
    matches = [p for p in RESULTS.glob(f'*/{name}') if p.exists()]
    legacy = RESULTS / str(name)
    if legacy.exists():
        matches.append(legacy)
    archived = RESULTS / 'archive' / 'development' / str(name)
    if archived.exists():
        matches.append(archived)
    if len(matches) == 1:
        return matches[0]
    if not matches and default is not None:
        return Path(default)
    if str(name) == 'textures_pilot' and not matches:
        return RESULTS / 'pilot' / str(name)
    locations = ', '.join(str(p) for p in matches) or 'none'
    raise KeyError(f'Result name is missing or ambiguous: {name}; matches: {locations}; pass an explicit path')


def source_path(name):
    """Find a script directly in code/ without a registry."""
    matches = [p for p in (EXPERIMENTS / 'code').glob('*/*.py') if p.name == str(name)]
    if len(matches) != 1:
        raise KeyError(f'Script name is missing or ambiguous: {name}')
    return matches[0]
