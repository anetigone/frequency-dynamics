"""List or execute an experiment by its unique script name."""
import runpy
import sys
from layout import MANIFEST, source_path

LIBRARIES = {'interval_core.py', 'square_images.py', 'sgd_state_tools.py'}

if len(sys.argv) < 2 or sys.argv[1] in ('--list', '--help', '-h'):
    print('Usage: python experiments/run.py SCRIPT [arguments...]\n')
    for name, entry in sorted(MANIFEST['source_files'].items(), key=lambda item: item[1]['current_path']):
        if name in LIBRARIES:
            continue
        print(f'{name[:-3]:34} {entry["current_path"]}')
else:
    name = sys.argv[1].removesuffix('.py') + '.py'
    if name not in MANIFEST['source_files']:
        raise SystemExit(f'Unknown script: {name}; use --list')
    if name in LIBRARIES:
        raise SystemExit(f'{name} is a shared library; use --list for executable scripts')
    path = source_path(name)
    sys.argv = [str(path), *sys.argv[2:]]
    runpy.run_path(str(path), run_name='__main__')
