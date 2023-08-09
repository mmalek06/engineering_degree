import os
import sys
import subprocess
import time

from functions.program_running import get_arguments, get_runs_data, should_exit


arguments = get_arguments()
model_type = arguments['type']
how_many_runs = int(arguments['runs'])
exit_file = arguments['exitfile']
root_path = os.path.join('classifiers', 'pretrained_models', model_type)
runs_data = get_runs_data(root_path)
module_dir = os.path.join('..', 'functions')
env = os.environ.copy()
env['PYTHONPATH'] = module_dir + ':' + env.get('PYTHONPATH', '')

for notebook_path, run in runs_data.items():
    if run >= how_many_runs:
        continue

    for _ in range(how_many_runs - run):
        if should_exit(exit_file):
            sys.exit(0)

        start_time = time.time()

        print(f'Running {notebook_path} for the {run} time.')
        subprocess.run(f'jupyter nbconvert --execute --to notebook --inplace {notebook_path}', shell=True, env=env)

        end_time = time.time()
        elapsed_seconds = end_time - start_time
        elapsed_minutes = elapsed_seconds / 60

        print(f'Run complete, elapsed seconds: {elapsed_seconds}, elapsed minutes: {elapsed_minutes}.')
        print()
