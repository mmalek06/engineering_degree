import argparse
import os

from collections import defaultdict

from functions.model_running import get_run_number


def get_arguments() -> dict[str, str]:
    parser = argparse.ArgumentParser()

    parser.add_argument('--type', help='Model type to run.')
    parser.add_argument('--runs', help='How many times each notebook should be run.')
    parser.add_argument('--exitfile', help='Program will look for this file in the user folder and if it\'s present, '
                                           'it will terminate after the current notebook processing completes.')

    args = parser.parse_args()
    arg_dict = vars(args)

    return arg_dict


def should_exit(exit_file: str) -> bool:
    path = os.path \
        .join(
            os.path.expanduser('~'),
            f'.{exit_file}')

    return os.path.isfile(path)


def get_runs_data(root_path: str) -> dict[str, int]:
    files = os.listdir(root_path)
    run_data = defaultdict(lambda: 1)
    category = os.path.basename(root_path)

    for file in files:
        file_path = os.path.join(root_path, file)
        file_base_name = file.split('.')[0]
        run_file_name = f'{category}_{file_base_name}'
        run_number = get_run_number(run_file_name)

        run_data[file_path] = run_number

    return run_data


def estimate_etas(all_run_times: list[float], total_runs: int) -> (float, float):
    estimated_run_time = _mean(all_run_times)
    total_time_elapsed = sum(all_run_times)
    estimated_total_time_required = total_runs * estimated_run_time
    estimated_eta_seconds = estimated_total_time_required - total_time_elapsed
    estimated_eta_minutes = estimated_eta_seconds / 60

    return estimated_eta_seconds, estimated_eta_minutes


def print_green(text: str) -> None:
    print(f'\033[92m {text}\033[00m')


def print_red(text: str) -> None:
    print(f'\033[91m {text}\033[00m')


def _mean(seq: list[float]) -> float:
    return sum(seq) / len(seq)
