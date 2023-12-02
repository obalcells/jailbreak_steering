import argparse
import json
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

BENCHMARK_DATASET_PATH = 'benchmarking/dataset/harmful_behaviors_for_bench_hard.csv'
BENCHMARK_RESULTS_DIR = 'benchmarking/results'
BENCHMARK_CONFIG_DIR = 'benchmarking/configs'

def run_benchmark(dataset_path=None, results_dir=None, logs_dir=None, config_path=None):
    command = ['python3', '-m', 'jailbreak_steering.suffix_gen.run_suffix_gen']

    if dataset_path is not None:
        command.extend(['--dataset_path', dataset_path])
    if results_dir is not None:
        command.extend(['--results_dir', results_dir])
    if logs_dir is not None:
        command.extend(['--logs_dir', logs_dir])
    if config_path is not None:
        command.extend(['--config_path', config_path])

    subprocess.run(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", type=int)
    args = parser.parse_args()

    for config_number in args.configs:
        config_path = os.path.join(BENCHMARK_CONFIG_DIR, f"{config_number}.json")
        results_dir = os.path.join(BENCHMARK_RESULTS_DIR, f"{config_number}", "results")
        logs_dir = os.path.join(BENCHMARK_RESULTS_DIR, f"{config_number}", "logs")
        plots_dir = os.path.join(BENCHMARK_RESULTS_DIR, f"{config_number}")

        run_benchmark(
            dataset_path=BENCHMARK_DATASET_PATH,
            results_dir=results_dir,
            logs_dir=logs_dir,
            config_path=config_path
        )