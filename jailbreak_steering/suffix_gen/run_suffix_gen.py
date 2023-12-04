import argparse
import os
import json
import time
import numpy as np
import torch
import pandas as pd
import gc

from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from jailbreak_steering.suffix_gen.prompt_manager import PromptManager
from jailbreak_steering.suffix_gen.suffix_gen import SuffixGen
from jailbreak_steering.utils.load_model import load_llama_2_7b_chat_model, load_llama_2_7b_chat_tokenizer

DEFAULT_DATASET_PATH = "datasets/unprocessed/advbench/harmful_behaviors_train.csv"
DEFAULT_SUFFIX_GEN_LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
DEFAULT_SUFFIX_GEN_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DEFAULT_SUFFIX_GEN_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "suffix_gen_config.json")
ALL_SUFFIX_GEN_RESULTS_FILENAME = "all_results.json"
SUCCESSFUL_SUFFIX_GEN_RESULTS_FILENAME = "successful_results.json"

MAX_NEW_TOKENS = 256

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

def get_log_path(logs_dir: str, dataset_idx: int):
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    return os.path.join(logs_dir, f'{dataset_idx:04}_{time.strftime("%H:%M:%S")}.json')

def evaluate_generation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    system_prompt: Optional[str],
    instruction: str,
    control: str,
    target: str,
    max_new_tokens: int=256,
):
    prompt = PromptManager(instruction, target, tokenizer, system_prompt, control, device=model.device)

    outputs = model.generate(
        prompt.input_ids[:prompt._target_slice.start].unsqueeze(0),
        max_new_tokens=max_new_tokens,
    )
    generation = tokenizer.batch_decode(outputs[:, prompt._target_slice.start:])[0]

    return generation

def generate_suffix(model, tokenizer, instruction, target, control_init, config, log_path):
    suffix_gen = SuffixGen(
        model,
        tokenizer,
        instruction,
        target,
        control_init=control_init,
        topk=config['topk'],
        batch_size=config['batch_size'],
        max_steps=config['max_steps'],
        early_stop_threshold=config['early_stop_threshold'],
        system_prompt=config['system_prompt'],
        verbose=config['verbose'],
        log_path=log_path,
        use_cache=config['use_cache'],
    )

    control, loss, steps = suffix_gen.run()

    return control, loss, steps

def save_results(results, results_dir, filename):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    with open(os.path.join(results_dir, filename), "w") as file:
        json.dump(results, file, indent=4)

def run_suffix_gen(dataset_path: str, results_dir: str, logs_dir: str, config_path: str, start_idx: int, end_idx: int):

    model = load_llama_2_7b_chat_model()
    tokenizer = load_llama_2_7b_chat_tokenizer()

    instructions, targets = load_dataset(dataset_path, n=None)
    config = load_config(config_path)
    default_control_init = ' '.join(['!' for _ in range(config['control_len'])])
 
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    control_init = default_control_init
    all_results = []
    successful_results = []

    for idx in range(start_idx, end_idx):
        instruction, target = instructions[idx], targets[idx]
        
        start_time = time.time()
        control, loss, steps = generate_suffix(
            model, tokenizer, instruction, target, control_init, config, get_log_path(logs_dir, idx),
        )
        runtime = time.time() - start_time

        gen_str = evaluate_generation(model, tokenizer, config['system_prompt'], instruction, control, target, max_new_tokens=MAX_NEW_TOKENS)
        success = (target in gen_str or loss < config['success_threshold'])

        print(f"************************************")
        print(f"*Instruction*: {instruction}")
        print(f"*Succeeded*: {success}")
        print(f"*Runtime*: {runtime:.5f}s; *Loss*: {loss:.5f}; *Steps*: {steps}; *Control str*: {control}")
        print(f"*Gen text*: \n{gen_str}")
        print(f"************************************")

        result = {
            "control": control,
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
            control_init = control
        else:
            control_init = default_control_init

        gc.collect(); torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--results_dir", type=str, default=DEFAULT_SUFFIX_GEN_RESULTS_DIR)
    parser.add_argument("--logs_dir", type=str, default=DEFAULT_SUFFIX_GEN_LOGS_DIR)
    parser.add_argument("--config_path", type=str, default=DEFAULT_SUFFIX_GEN_CONFIG_PATH)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=10) # non-inclusive

    args = parser.parse_args()

    run_suffix_gen(args.dataset_path, args.results_dir, args.logs_dir, args.config_path, args.start_idx, args.end_idx)
