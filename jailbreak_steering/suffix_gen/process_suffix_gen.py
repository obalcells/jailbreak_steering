'''
Processes results from suffix_gen, converting them into a format that can be used by the steering vector generator.
'''

import argparse
import json
import os
import numpy as np

from jailbreak_steering.utils.load_model import load_llama_2_7b_chat_tokenizer
from jailbreak_steering.utils.tokenize_llama_chat import get_ascii_toks
from jailbreak_steering.suffix_gen.run_suffix_gen import DEFAULT_SUFFIX_GEN_RESULTS_DIR, SUCCESSFUL_SUFFIX_GEN_RESULTS_FILENAME

DEFAULT_SUFFIX_GEN_RESULTS_PATH = os.path.join(DEFAULT_SUFFIX_GEN_RESULTS_DIR, SUCCESSFUL_SUFFIX_GEN_RESULTS_FILENAME)
DEFAULT_OUTPUT_DIR = os.path.join("datasets", "processed")

NO_SUFFIX_DATASET_FILENAME ="suffix_no_suffix.json"
RANDOM_SUFFIX_DATASET_FILENAME = "suffix_random_suffix.json"

def process_suffix_gen(suffix_gen_results_path: str, output_dir: str):
        
    successful_results = json.load(open(suffix_gen_results_path, "r"))

    dataset_no_suffix = []
    dataset_random_suffix = []

    tokenizer = load_llama_2_7b_chat_tokenizer()
    ascii_toks = get_ascii_toks(tokenizer)

    for successful_result in successful_results:
        instruction = successful_result["instruction"]
        control_str = successful_result["control_str"]

        dataset_no_suffix.append(
            {
                "instruction_inducing_behavior": f"{instruction}",
                "instruction_not_inducing_behavior": f"{instruction} {control_str}",
            }
        )

        # generate a random suffix with the same token length as control_str
        random_ascii_toks = np.random.choice(ascii_toks, size=len(tokenizer.encode(control_str)))
        random_suffix = tokenizer.decode(random_ascii_toks)

        dataset_random_suffix.append(
            {
                "instruction_inducing_behavior": f"{instruction} {random_suffix}",
                "instruction_not_inducing_behavior": f"{instruction} {control_str}",
            }
        )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, NO_SUFFIX_DATASET_FILENAME), "w") as file:
        json.dump(dataset_no_suffix, file, indent=4)
    with open(os.path.join(output_dir, RANDOM_SUFFIX_DATASET_FILENAME), "w") as file:
        json.dump(dataset_random_suffix, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix_gen_results_path", type=str, default=DEFAULT_SUFFIX_GEN_RESULTS_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)

    args = parser.parse_args()

    process_suffix_gen(args.suffix_gen_results_path, args.output_dir)
