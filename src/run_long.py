import os
import sys
import subprocess
import time

from functions.program_running import \
    estimate_etas, \
    get_arguments, \
    get_runs_data, \
    should_exit, \
    print_green, \
    print_red


arguments = get_arguments()
model_type = arguments['type']
how_many_runs = int(arguments['runs'])
exit_file = arguments['exitfile']
chosen_model = arguments.get('model')
root_path = os.path.join('classifiers', 'pretrained_models', model_type)
runs_data = get_runs_data(root_path)

if chosen_model is not None:
    runs_data = dict(filter(lambda pair: chosen_model in pair[0], runs_data.items()))

total_runs = len(runs_data) * how_many_runs
current_run = sum([run - 1 for run in runs_data.values()]) + 1
all_run_times = []

for notebook_path, run in runs_data.items():
    if run >= how_many_runs:
        continue

    for _ in range(how_many_runs - run + 1):
        if should_exit(exit_file):
            print_red('Exit file encountered, aborting...')
            sys.exit(0)

        start_time = time.time()

        print_green(f'Run {current_run} out of {total_runs}...')
        print_green(f'Running {notebook_path}.')
        subprocess.run(f'jupyter nbconvert --execute --to notebook --inplace {notebook_path}', shell=True)

        end_time = time.time()
        elapsed_seconds = end_time - start_time
        elapsed_minutes = elapsed_seconds / 60

        all_run_times.append(elapsed_seconds)

        estimated_eta_seconds, estimated_eta_minutes = estimate_etas(all_run_times, total_runs)

        print_green(f'Run complete, elapsed seconds: {elapsed_seconds}, elapsed minutes: {elapsed_minutes}.')
        print_green(f'Estimated ETA is: {estimated_eta_seconds} seconds, {estimated_eta_minutes} minutes.')
        print()

        current_run += 1
