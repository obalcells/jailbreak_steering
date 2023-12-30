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
DEFAULT_OUTPUT_PATH = os.path.join("datasets", "processed", "processed_suffix_gen.json")

def process_suffix_gen(
    suffix_gen_results_path: str,
    output_path: str,
    pair_with_random_suffix: bool,
    suffix_induces_behavior: bool
):
    successful_results = json.load(open(suffix_gen_results_path, "r"))

    dataset = []

    tokenizer = load_llama_2_7b_chat_tokenizer()
    ascii_toks = get_ascii_toks(tokenizer)

    for successful_result in successful_results:
        instruction = successful_result["instruction"]
        control = successful_result["control"]

        instruction_plus_suffix = f"{instruction} {control}"
        instruction_baseline = f"{instruction}"

        if pair_with_random_suffix:
            # generate a random suffix with the same token length as control_str
            random_ascii_toks = np.random.choice(ascii_toks, size=len(tokenizer.encode(control)))
            random_suffix = tokenizer.decode(random_ascii_toks)

            instruction_baseline = f"{instruction} {random_suffix}"

        if suffix_induces_behavior:
            dataset.append(
                {
                    "instruction_inducing_behavior": instruction_plus_suffix,
                    "instruction_not_inducing_behavior": instruction_baseline,
                }
            )
        else:
            dataset.append(
                {
                    "instruction_inducing_behavior": instruction_baseline,
                    "instruction_not_inducing_behavior": instruction_plus_suffix,
                }
            )
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    with open(os.path.join(output_path), "w") as file:
        json.dump(dataset, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix_gen_results_path", type=str, default=DEFAULT_SUFFIX_GEN_RESULTS_PATH)
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--pair_with_random_suffix", default=False, action='store_true')

    parser.add_argument('--suffix_induces_behavior', dest='suffix_induces_behavior', action='store_true')
    parser.add_argument('--suffix_does_not_induce_behavior', dest='suffix_induces_behavior', action='store_false')
    parser.set_defaults(suffix_induces_behavior=True)

    args = parser.parse_args()

    process_suffix_gen(args.suffix_gen_results_path, args.output_path, args.pair_with_random_suffix, args.suffix_induces_behavior)
