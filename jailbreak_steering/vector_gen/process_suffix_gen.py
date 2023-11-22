'''
Processes results from suffix_gen, converting them into a format that can be used by the steering vector generator.
'''

import json
import os
import numpy as np

from jailbreak_steering.suffix_gen.run_suffix_gen import SUCCESSFUL_RESULTS_PATH
from jailbreak_steering.utils.load_model import load_llama_2_7b_chat_tokenizer
from jailbreak_steering.utils.tokenize_llama_chat import get_ascii_toks

NO_SUFFIX_DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "instruction_datasets", "jailbreak_no_suffix_dataset.json")
RANDOM_SUFFIX_DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "instruction_datasets", "jailbreak_random_suffix_dataset.json")

successful_results = json.load(open(SUCCESSFUL_RESULTS_PATH, "r"))

dataset_no_suffix = []
dataset_random_suffix = []

tokenizer = load_llama_2_7b_chat_tokenizer()
ascii_toks = get_ascii_toks(tokenizer)

for successful_result in successful_results:
    instruction = successful_result["instruction"]
    control_str = successful_result["control_str"]

    dataset_no_suffix.append(
        {
            "instruction_inducing_behavior": f"{instruction} {control_str}",
            "instruction_not_inducing_behavior": f"{instruction}",
        }
    )

    # generate a random suffix with the same token length as control_str
    random_ascii_toks = np.random.choice(ascii_toks, size=len(tokenizer.encode(control_str)))
    random_suffix = tokenizer.decode(random_ascii_toks)

    dataset_random_suffix.append(
        {
            "instruction_inducing_behavior": f"{instruction} {control_str}",
            "instruction_not_inducing_behavior": f"{instruction} {random_suffix}",
        }
    )

if not os.path.exists(os.path.dirname(NO_SUFFIX_DATASET_PATH)):
    os.makedirs(os.path.dirname(NO_SUFFIX_DATASET_PATH))
with open(NO_SUFFIX_DATASET_PATH, "w") as file:
    json.dump(dataset_no_suffix, file, indent=4)

if not os.path.exists(os.path.dirname(RANDOM_SUFFIX_DATASET_PATH)):
    os.makedirs(os.path.dirname(RANDOM_SUFFIX_DATASET_PATH))
with open(RANDOM_SUFFIX_DATASET_PATH, "w") as file:
    json.dump(dataset_random_suffix, file, indent=4)
