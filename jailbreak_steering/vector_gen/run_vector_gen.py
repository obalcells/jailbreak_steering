# This file based off of https://github.com/nrimsky/SycophancySteering/blob/main/generate_vectors.py

"""
Generates steering vectors for each layer of the model by averaging the activations of all the positive and negative examples.
"""

import torch as t
import argparse
import os

from tqdm import tqdm
from typing import List

from jailbreak_steering.utils.llama_wrapper import LlamaWrapper
from jailbreak_steering.vector_gen.comparison_datasets import InstructionAnswerComparisonDataset, InstructionComparisonDataset
from jailbreak_steering.suffix_gen.process_suffix_gen import DEFAULT_OUTPUT_PATH

DEFAULT_DATASET_PATH = DEFAULT_OUTPUT_PATH
DEFAULT_LABEL = "label"
DEFAULT_VECTORS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectors")
DEFAULT_ACTIVATIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "activations")

SYSTEM_PROMPT = None

def generate_save_vectors(dataset_path: str, data_type: str, vectors_dir: str, use_default_system_prompt: bool, activations_dir: str, save_activations: bool):
    if not os.path.exists(vectors_dir):
        os.makedirs(vectors_dir)
    if save_activations and not os.path.exists(activations_dir):
        os.makedirs(activations_dir)

    system_prompt = "default" if use_default_system_prompt else None

    model = LlamaWrapper(system_prompt)
    model.set_save_internal_decodings(False)
    model.reset_all()

    layers = list(range(len(model.model.model.layers)))

    pos_activations = dict([(layer, []) for layer in layers])
    neg_activations = dict([(layer, []) for layer in layers])

    if data_type == "instruction":
        dataset = InstructionComparisonDataset(
            dataset_path,
            system_prompt,
            model.tokenizer,
        )
    elif data_type == "instruction_answer":
        dataset = InstructionAnswerComparisonDataset(
            dataset_path,
            system_prompt,
            model.tokenizer,
        )
    else:
        raise ValueError(f"Invalid data type {data_type}. Must be 'instruction' or 'instruction_answer'.")

    for p_tokens, n_tokens in tqdm(dataset, desc="Processing prompts"):
        p_tokens = p_tokens.to(model.device)
        n_tokens = n_tokens.to(model.device)
        model.reset_all()
        model.get_logits(p_tokens)
        for layer in layers:
            p_activations = model.get_last_activations(layer)
            p_activations = p_activations[0, -1, :].detach().cpu()
            pos_activations[layer].append(p_activations)
        model.reset_all()
        model.get_logits(n_tokens)
        for layer in layers:
            n_activations = model.get_last_activations(layer)
            n_activations = n_activations[0, -1, :].detach().cpu()
            neg_activations[layer].append(n_activations)

    for layer in layers:
        all_pos_layer = t.stack(pos_activations[layer])
        all_neg_layer = t.stack(neg_activations[layer])
        vec = (all_pos_layer - all_neg_layer).mean(dim=0)
        t.save(
            vec,
            os.path.join(
                vectors_dir,
                f"vec_layer_{layer}.pt",
            ),
        )
        if save_activations:
            t.save(
                all_pos_layer,
                os.path.join(
                    activations_dir,
                    f"activations_pos_{layer}.pt",
                ),
            )
            t.save(
                all_neg_layer,
                os.path.join(
                    activations_dir,
                    f"activations_neg_{layer}.pt",
                ),
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--vectors_dir", type=str, default=DEFAULT_VECTORS_DIR)
    parser.add_argument("--data_type", type=str, default="instruction")
    parser.add_argument("--use_default_system_prompt", action='store_true', default=False)
    parser.add_argument("--save_activations", action='store_true', default=False)
    parser.add_argument("--activations_dir", type=str, default=DEFAULT_ACTIVATIONS_DIR)

    args = parser.parse_args()
    generate_save_vectors(
        args.dataset_path, args.data_type, args.vectors_dir, args.use_default_system_prompt, args.activations_dir, args.save_activations
    )