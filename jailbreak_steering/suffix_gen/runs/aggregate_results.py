"""
This script aggregates JSON data from specific files located in subdirectories of the script's directory. 
It looks for subdirectories named in the format {number}_{number}, and within each, it aggregates data 
from 'all_results.json' and 'successful_results.json'. The aggregated data is then saved into separate 
JSON files in an 'aggregated_results' directory.

Usage:
python3 -m jailbreak_steering.suffix_gen.runs.aggregate_results
"""

import os
import json

def is_valid_subdirectory(dirname):
    parts = dirname.split('_')
    return len(parts) == 2 and all(part.isdigit() for part in parts)

def aggregate_results(script_dir, filename):
    aggregated_data = []
    for subdir in os.listdir(script_dir):
        if is_valid_subdirectory(subdir):
            file_path = os.path.join(script_dir, subdir, 'results', filename)
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    aggregated_data.extend(data)

    return aggregated_data

def main():
    runs_dir = os.path.dirname(os.path.abspath(__file__))
    aggregated_all_data = aggregate_results(runs_dir, 'all_results.json')
    aggregated_successful_data = aggregate_results(runs_dir, 'successful_results.json')

    os.makedirs(os.path.join(runs_dir, 'aggregated_results'), exist_ok=True)
    
    with open(os.path.join(runs_dir, 'aggregated_results', 'all_results.json'), 'w') as file:
        json.dump(aggregated_all_data, file, indent=4)

    with open(os.path.join(runs_dir, 'aggregated_results', 'successful_results.json'), 'w') as file:
        json.dump(aggregated_successful_data, file, indent=4)

if __name__ == "__main__":
    main()
