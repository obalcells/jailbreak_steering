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

def process_results(logs_dir: str, output_dir: str, plot_label: str):
    loss_vs_time_fig = go.Figure()
    loss_vs_step_fig = go.Figure()

    for i, file_name in enumerate(os.listdir(logs_dir)):
        file_path = os.path.join(logs_dir, file_name)
        with open(file_path, 'r') as file:

            data = json.load(file)
            steps_df = pd.DataFrame(data['steps'])
            steps_df['time'] = steps_df['time'] - data['start_time']

            loss_vs_time_fig.add_trace(go.Scatter(x=steps_df['time'], y=steps_df['loss'], mode='lines+markers', name=i))
            loss_vs_step_fig.add_trace(go.Scatter(x=steps_df['n_step'], y=steps_df['loss'], mode='lines+markers', name=i))
    loss_vs_time_fig.update_layout(title=f'Loss vs Time, {plot_label}', xaxis_title='Time (sec)', yaxis_title='Loss', legend_title='Example')
    loss_vs_time_fig.write_image(os.path.join(output_dir, "loss_vs_time.png"), scale=5)

    loss_vs_step_fig.update_layout(title=f'Loss vs Step, {plot_label}', xaxis_title='Step', yaxis_title='Loss', legend_title='Example')
    loss_vs_step_fig.write_image(os.path.join(output_dir, "loss_vs_step.png"), scale=5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", type=int)
    args = parser.parse_args()

    for config_number in args.configs:
        config_path = os.path.join(BENCHMARK_CONFIG_DIR, f"{config_number}.json")
        results_dir = os.path.join(BENCHMARK_RESULTS_DIR, f"{config_number}", "results")
        logs_dir = os.path.join(BENCHMARK_RESULTS_DIR, f"{config_number}", "logs")
        plots_dir = os.path.join(BENCHMARK_RESULTS_DIR, f"{config_number}")

        process_results(logs_dir, plots_dir, f"Config {config_number}")
