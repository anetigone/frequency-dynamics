"""List or execute an experiment by its unique script name."""
import runpy
import sys
from layout import EXPERIMENTS, source_path

LIBRARIES = {'interval_core.py', 'square_images.py', 'sgd_state_tools.py', 'dtd_hf.py'}

if len(sys.argv) < 2 or sys.argv[1] in ('--list', '--help', '-h'):
    print('Usage: python experiments/run.py SCRIPT [arguments...]\n')
    for path in sorted((EXPERIMENTS / 'code').glob('*/*.py')):
        if path.name in LIBRARIES:
            continue
        print(f'{path.stem:34} {path.relative_to(EXPERIMENTS)}')
else:
    name = sys.argv[1].removesuffix('.py') + '.py'
    if name in LIBRARIES:
        raise SystemExit(f'{name} is a shared library; use --list for executable scripts')
    try:
        path = source_path(name)
    except KeyError as error:
        raise SystemExit(f'{error}; use --list') from None
    sys.argv = [str(path), *sys.argv[2:]]
    runpy.run_path(str(path), run_name='__main__')
