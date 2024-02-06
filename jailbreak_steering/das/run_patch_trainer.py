# %%
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# %cd ..
# assert os.getcwd().endswith("mats_refusal")
# %%
# from IPython import get_ipython
# ipython = get_ipython()
# if ipython is not None:
#     ipython.magic("load_ext autoreload")
#     ipython.magic("autoreload 2")
import sys
sys.path.append('..')

# %%
import time
import torch
import torch.nn as nn
import json
import gc
import argparse
from typing import Callable, Tuple, Union, Dict, Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens import utils as tl_utils

from mats_refusal.das.patch_trainer import train
from mats_refusal.das.das_utils import Patcher, Node, Rotation, RotationMatrix
from mats_refusal.utils.utils import tokenize_instructions

DEFAULT_DAS_DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "das_train_dataset.json")
DEFAULT_DAS_PATCHER_CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patcher_configs")
DEFAULT_DAS_PATCHER_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patcher_weights")
DEFAULT_DAS_TRAIN_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_configs", "default_train_params.json")
DEFAULT_DAS_PATCHER = "early_patcher"

# %%

def load_model_and_tokenizer(model_name_path: str, device: str = "cpu") -> Tuple[HookedTransformer, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(
        model_name_path,
        token=os.environ["HF_TOKEN"],
        low_cpu_mem_usage=True,
        use_cache=False,
        torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_path,
        token=os.environ["HF_TOKEN"],
        use_fast=False
    )

    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = 'left'

    tl_model = HookedTransformer.from_pretrained(
        model_name_path,
        hf_model=model,
        device='cpu',
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        tokenizer=tokenizer,
        default_padding_side='left',
        dtype=torch.float16,
    ).to(device)

    tl_model = tl_model.eval()

    # we have to do it again because TL resets it back to 'right'
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = 'left'

    return tl_model, tokenizer


def load_patcher(patcher_label, load_weights):

    patcher_config_folder = os.path.join(DEFAULT_DAS_PATCHER_CONFIGS_FOLDER, patcher_label)
    weights_dir = os.path.join(DEFAULT_DAS_PATCHER_WEIGHTS_FOLDER, patcher_label)

    patcher_config_path = os.path.join(patcher_config_folder, patcher_label + ".json")
    assert os.path.exists(patcher_config_path), f"Could not find patcher config at {patcher_config_path}"

    patcher_config = json.load(open(patcher_config_path, "r"))

    nodes_config = patcher_config["nodes"]

    # check that all the nodes are at the same layer and activation
    # we'll change this later to enable nodes at different layers; they require different patch_impls
    assert all([n["layer"] == nodes_config[0]["layer"] for n in nodes_config]), "All nodes must be at the same layer"
    assert all([n["name"] == nodes_config[0]["name"] for n in nodes_config]), "All nodes must be at the same activation"

    nodes = []
    for node_config in nodes_config:
        seq_pos = node_config["seq_pos"]
        if isinstance(seq_pos, list):
            assert all([seq_pos[i] == seq_pos[i-1]+1 for i in range(1, len(seq_pos))]), "seq_pos must be a list of consecutive integers"
            seq_pos = slice(seq_pos[0], seq_pos[-1]+1)
        elif isinstance(seq_pos, int):
            seq_pos = slice(seq_pos, seq_pos+1)
        else:
            raise ValueError("Invalid seq_pos type")
        nodes.append(Node(node_config["name"], layer=node_config["layer"], seq_pos=seq_pos))

    patch_impl_list = []
    impl_configs = patcher_config["patch_impl"] if isinstance(patcher_config["patch_impl"], list) else [patcher_config["patch_impl"]]

    for i, patcher_impl_config in enumerate(impl_configs):

        if patcher_impl_config["name"] == "rotation":
            type = patcher_impl_config["rotation"]["type"]
            assert type in ["float16", "float32", "float64"], f"Invalid type {type}"
            str_type_to_dtype = {"float16": torch.float16, "float32": torch.float32, "float64": torch.float64}
            dtype = str_type_to_dtype[type]

            rotation_matrix = RotationMatrix(
                n=4096
            ).type(dtype).cuda()

            if load_weights == "latest":
                if not os.path.exists(weights_dir):
                    os.makedirs(weights_dir)
                all_weights_unfiltered = os.listdir(weights_dir)
                all_weights = [w.split("/")[-1] for w in all_weights_unfiltered if w.endswith(".pt") and (f'rot_{i}' in w or len(nodes) == 1)]
                if len(all_weights) == 0:
                    print(f"No weights found at {weights_dir}")
                else:
                    # weights are indexed starting from 0 -> "0_early_patcher.pt"
                    def calc_order(x):
                        x = x.split("_")
                        idx = int(x[0])
                        if 'epoch' in x:
                            # then it looks like '0_early_patcher_epoch_1.pt'
                            return idx * 1000 + int(x[-1][:-3])
                        return idx * 1000
                    all_weights = sorted(all_weights, key=lambda x: calc_order(x))
                    weights_path = os.path.join(weights_dir, all_weights[-1])
                    print(f"Loading weights from {weights_path}")
                    rotation_matrix.load_state_dict(torch.load(weights_path))
            elif load_weights == "none":
                pass
            else:
                assert False

            patch_impl_list.append(
                Rotation(
                    rotation=rotation_matrix,
                    dim=patcher_impl_config["dim"]
                )
            )

        else:
            raise NotImplementedError("Not implemented patcher_impl type")

    return Patcher(
        nodes=nodes,
        patch_impl_list=patch_impl_list
    )

def train_patcher(
    patcher_label: str,
    load_weights: str,
    dataset_path: str,
    n_epochs: int,
    train_config_path: str,
    batch_size: int
):
    """
    Train a patcher on the given dataset.
    """
    # Load the train config
    assert os.path.exists(train_config_path), f"Could not find train config at {train_config_path}"
    train_config = json.load(open(train_config_path, "r"))

    # Load the patcher
    patcher = load_patcher(patcher_label, load_weights)

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name_path = "meta-llama/Llama-2-7b-chat-hf"
    model, tokenizer = load_model_and_tokenizer(model_name_path, device)
    torch.set_grad_enabled(False)

    # Now we load the dataset
    assert os.path.exists(dataset_path), f"Could not find dataset at {dataset_path}"
    dataset = json.load(open(dataset_path, "r"))

    append_space_after_inst = train_config["append_space_after_inst"]

    instructions_matching_behavior = [el["instruction_matching_behavior"] for el in dataset] 
    instructions_not_matching_behavior = [el["instruction_not_matching_behavior"] for el in dataset]
    answers_matching_behavior = [el["answer_matching_behavior"] for el in dataset]
    answers_not_matching_behavior = [el["answer_not_matching_behavior"] for el in dataset]

    instructions = [instructions_matching_behavior, instructions_not_matching_behavior]
    answers = [answers_matching_behavior, answers_not_matching_behavior]

    instruction_tokens = []
    for i in range(2):
        instruction_tokens.append(tokenize_instructions(tokenizer=tokenizer, instructions=instructions[i], append_space_after_inst=append_space_after_inst))

    instruction_len = len(instruction_tokens[0][0])
    assert instructions[0].shape == instructions[1].shape, "Datasets must be of the same size"

    num_target_tokens = train_config["num_target_tokens"]

    answer_tokens = []
    for i in range(2):
        answer_tokens.append(
            torch.stack(
                # here we concatenate each tokenized instruction with its corresponding tokenized answer
                # we remove the first token because it's the <s> token
                [torch.cat([instruction_tokens[i], tokenizer.encode(answers[i], return_tensors="pt")[0, 1:1+num_target_tokens]], dim=0) for answer in answers[i]],
                dim=0
            ).to(device)
        )

    # Get the training run idx and the path for saving the weights and the logs
    patcher_config_folder = os.path.join(DEFAULT_DAS_PATCHER_CONFIGS_DIR, patcher_label)
    patcher_config_path = os.path.join(patcher_config_folder, patcher_label + ".json")
    assert os.path.exists(patcher_config_path), f"Could not find patcher config at {patcher_config_path}"
    patcher_config = json.load(open(patcher_config_path, "r"))
    patcher_training_run_idx = len(patcher_config["training_runs"])

    save_patcher_path = os.path.join(patcher_folder, "weights", f"{patcher_training_run_idx}_{patcher_label}")
    log_path = os.path.join(patcher_folder, "logs", f"{patcher_training_run_idx}_{patcher_label}.log")

    results = train(
        model=model,
        harmful_tokens=harmful_tokens,
        harmless_tokens=harmless_tokens,
        patcher=patcher,
        save_patcher_path=save_patcher_path,
        log_path=log_path,
        instruction_len=instruction_len,
        n_epochs=n_epochs,
        initial_lr=train_config["initial_lr"],
        batch_size=batch_size,
        eval_every=1,
        optim_metric=train_config["optim_params"],
        verbose=True
    )

    datetime = time.strftime("%Y%m%d-%H:%M")
    loss_harmful_to_harmless = results["loss_harmful->harmless"]
    loss_harmless_to_harmful = results["loss_harmless->harmful"]
    metric_harmful_to_harmless = results["metric_harmful->harmless"]
    metric_harmless_to_harmful = results["metric_harmless->harmful"]

    # we reload the config before updating it
    # we do this in case some other patcher is also currently running and might have edited the file after we've loaded it here
    patcher_config = json.load(open(patcher_config_path, "r"))

    # we append the new training run to the config
    patcher_config["training_runs"].append({
        "datetime": datetime,
        "loss_harmful->harmless": loss_harmful_to_harmless,
        "loss_harmless->harmful": loss_harmless_to_harmful,
        "metric_harmful->harmless": metric_harmful_to_harmless,
        "metric_harmless->harmful": metric_harmless_to_harmful,
        "config_path": train_config_path,
        "n_epochs": n_epochs,
    })

    json.dump(patcher_config, open(patcher_config_path, "w"), indent=4)

# %%

# user_input = input("Load again? (y/n) ")
# if user_input == "yes":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model_name_path = "meta-llama/Llama-2-7b-chat-hf"

#     model = AutoModelForCausalLM.from_pretrained(
#         model_name_path,
#         token=os.environ["HF_TOKEN"],
#         low_cpu_mem_usage=True,
#         use_cache=False,
#         torch_dtype=torch.float16,
#     )

#     tokenizer = AutoTokenizer.from_pretrained(
#         model_name_path,
#         token=os.environ["HF_TOKEN"],
#         use_fast=False
#     )

#     tokenizer.pad_token = tokenizer.unk_token
#     tokenizer.padding_side = 'left'

#     tl_model = HookedTransformer.from_pretrained(
#         model_name_path,
#         hf_model=model,
#         device='cpu',
#         fold_ln=False,
#         center_writing_weights=False,
#         center_unembed=False,
#         tokenizer=tokenizer,
#         default_padding_side='left',
#         dtype=torch.float16,
#     ).to(device)

#     tl_model = tl_model.eval()

#     # we have to do it again because TL resets it back to 'right'
#     tokenizer.pad_token = tokenizer.unk_token
#     tokenizer.padding_side = 'left'

#     torch.set_grad_enabled(False)

# %%

# tl_model.reset_hooks()
# tl_model.zero_grad()
# torch.cuda.empty_cache()
# gc.collect()


# %%

# patcher = "early_patcher"
# load_weights = "latest"
# dataset_path = "./data/das_train_dataset.json"
# num_epochs = 1
# train_config = DEFAULT_DAS_TRAIN_CONFIG
# batch_size = 64

# # %%

# train_patcher(
#     tl_model,
#     tokenizer,
#     patcher,
#     load_weights,
#     dataset_path,
#     num_epochs,
#     train_config,
#     batch_size
# )

# %%

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patcher", type=str, default=DEFAULT_DAS_PATCHER)
    parser.add_argument("--load_weights", type=str, default="latest")
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DAS_DATASET_PATH)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--train_config", type=str, default=DEFAULT_DAS_TRAIN_CONFIG)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    train_patcher(
        args.patcher,
        args.load_weights,
        args.dataset_path,
        args.num_epochs,
        args.train_config,
        args.batch_size
    )