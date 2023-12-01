# %%
import argparse
import os
import json
import time
import numpy as np
import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM

from jailbreak_steering.suffix_gen.prompt_manager import PromptManager
from jailbreak_steering.suffix_gen.suffix_gen import SuffixGen
from jailbreak_steering.utils.load_model import load_llama_2_7b_chat_model, load_llama_2_7b_chat_tokenizer

DEFAULT_DATASET_PATH = "datasets/unprocessed/advbench/harmful_behaviors_train.csv"
DEFAULT_SUFFIX_GEN_LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
DEFAULT_SUFFIX_GEN_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DEFAULT_SUFFIX_GEN_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "suffix_gen_config.json")
ALL_SUFFIX_GEN_RESULTS_FILENAME = "all_results.json"
SUCCESSFUL_SUFFIX_GEN_RESULTS_FILENAME = "successful_results.json"

N_INSTRUCTIONS = 10 # change to 'None' to run on all instructions

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def load_dataset(dataset_path: str, n: int=None):
    instructions = []
    targets = []

    dataset = pd.read_csv(dataset_path)
    instructions = dataset['goal'].tolist()
    targets = dataset['target'].tolist()

    if n is not None:
        instructions = instructions[:n]
        targets = targets[:n]

    return instructions, targets

def evaluate_target_loss(
    model: AutoModelForCausalLM,
    prompt: PromptManager
):
    logits = model(prompt.input_ids.unsqueeze(0).to(model.device)).logits

    crit = torch.nn.CrossEntropyLoss(reduction='none')
    loss = crit(
        logits[0, prompt._loss_slice, :],
        prompt.input_ids[prompt._target_slice].to(model.device)
    ).mean(dim=-1)
    return loss

def evaluate_generation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: PromptManager,
    max_new_tokens: int=256,
):
    outputs = model.generate(
        prompt.input_ids[:prompt._target_slice.start].unsqueeze(0).to(model.device),
        max_new_tokens=max_new_tokens,
    )
    generation = tokenizer.batch_decode(outputs[:, prompt._target_slice.start:])[0]

    return generation

def generate_suffix(model, tokenizer, instruction, target, control_init, config, logs_dir):
    suffix_gen = SuffixGen(
        model,
        tokenizer,
        instruction,
        target,
        control_init=control_init,
        early_stop_threshold=config['early_stop_threshold'],
        system_prompt=config['system_prompt'],
        verbose=config['verbose'],
        logs_dir=logs_dir,
    )

    start = time.time()

    control_str, _, steps = suffix_gen.run(
        max_steps=config['max_steps'],
        batch_size=config['batch_size'],
        topk=config['topk'],
    )

    runtime = time.time() - start
    loss = evaluate_target_loss(model, suffix_gen.prompt).item()
    gen_str = evaluate_generation(model, tokenizer, suffix_gen.prompt, max_new_tokens=256)

    success = (target in gen_str or loss < config['success_threshold'])

    return control_str, runtime, loss, steps, gen_str, success

def save_results(results, results_dir, filename):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    with open(os.path.join(results_dir, filename), "w") as file:
        json.dump(results, file, indent=4)

model = load_llama_2_7b_chat_model()
tokenizer = load_llama_2_7b_chat_tokenizer()

def run_suffix_gen(dataset_path: str, results_dir: str, logs_dir: str, config_path: str):
    instructions, targets = load_dataset(dataset_path, n=N_INSTRUCTIONS)
    config = load_config(config_path)
    default_control_init = ' '.join(['!' for _ in range(config['control_len'])])
 
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    control_init = default_control_init
    all_results = []
    successful_results = []

    for instruction, target in zip(instructions, targets):
        control_str, runtime, loss, steps, gen_str, success = generate_suffix(
            model, tokenizer, instruction, target, control_init, config, logs_dir
        )

        print(f"************************************")
        print(f"For instruction {instruction[:30]}, suffix generation {'succeeded' if success else 'did not succeed'}")
        print(f"Runtime {runtime:.5f}s, Loss {loss:.5f}, Steps {steps}, Control str: {control_str}")
        print(f"Gen text: {gen_str}")
        print(f"************************************")

        result = {
            "control_str": control_str,
            "instruction": instruction,
            "target": target,
            "gen_str": gen_str,
            "system_prompt": config['system_prompt'],
            "steps": steps,
            "loss": loss,
        }

        all_results.append(result)
        save_results(all_results, results_dir, ALL_SUFFIX_GEN_RESULTS_FILENAME)

        if success:
            successful_results.append(result)
            save_results(successful_results, results_dir, SUCCESSFUL_SUFFIX_GEN_RESULTS_FILENAME)

        if config['reuse_control'] and success and len(all_results) % config['reset_control_after'] != 0:
            control_init = control_str
        else:
            control_init = default_control_init

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH)
#     parser.add_argument("--results_dir", type=str, default=DEFAULT_SUFFIX_GEN_RESULTS_DIR)
#     parser.add_argument("--logs_dir", type=str, default=DEFAULT_SUFFIX_GEN_LOGS_DIR)
#     parser.add_argument("--config_path", type=str, default=DEFAULT_SUFFIX_GEN_CONFIG_PATH)

#     args = parser.parse_args()

#     run_suffix_gen(args.dataset_path, args.results_dir, args.logs_dir, args.config_path)

# %%

run_suffix_gen(DEFAULT_DATASET_PATH, DEFAULT_SUFFIX_GEN_RESULTS_DIR, DEFAULT_SUFFIX_GEN_LOGS_DIR, DEFAULT_SUFFIX_GEN_CONFIG_PATH)
# %%
