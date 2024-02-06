import gc
import os
import torch
import numpy as np
import einops
import transformer_lens
import functools
import plotly.graph_objects as go
import plotly.express as px
import circuitsvis as cv
import tqdm
import json
from functools import partial
import einops
from collections import defaultdict
import random

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens import utils as tl_utils
from transformer_lens.hook_points import HookPoint
from torch import Tensor
from torch.nn import Parameter
from torch import nn
from torch.nn.utils.parametrizations import orthogonal
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from jaxtyping import Int, Float
from typing import Callable, Dict, List, Optional, Tuple, Union
from sklearn.decomposition import PCA
import einops
from fancy_einsum import einsum
import torch
from jaxtyping import Bool, Float, Int
from typing import Callable, Tuple, Union, Dict, Optional, List
import torch

from jailbreak_steering.utils.utils import tokenize_instructions
from jailbreak_steering.das.das_utils import Node, Rotation, RotationMatrix, Patcher


def clear_memory():
    initial_mem = int(torch.cuda.memory_allocated() / 2**20)
    gc.collect()
    torch.cuda.empty_cache()
    after_mem = int(torch.cuda.memory_allocated() / 2**20)
    print(f"Cleared {initial_mem-after_mem} MB. Current CUDA memory is {after_mem} MB.")

def save_logs(all_metrics, log_path):
    log_dir_path = os.path.dirname(log_path)
    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)
    with open(log_path, "w") as file:
        json.dump(all_metrics, file, indent=4)

def save_patcher(patcher, save_patcher_path):
    save_patch_dir = os.path.dirname(save_patcher_path)
    if not os.path.exists(save_patch_dir):
        os.makedirs(save_patch_dir)
    for i, patch_impl in enumerate(patcher.patch_impl_list):
        save_path = save_patcher_path.replace(".pt", f"_patch_impl_{i}.pt")
        patch_impl.save(save_path)


refusal_tokens = [306, 8221] # 'I', 'Sorry'
answer_tokens = [18585, 7027] # 'Sure', 'Great'

def get_logit_avg(logits: Float[Tensor, 'batch seq_len n_vocab'], tokens=refusal_tokens) -> float:
    assert (logits.ndim == 3 or logits.ndim == 2)
    if logits.ndim == 2:
        return logits[:, tokens].mean()
    else:
        return logits[:, -1, tokens].mean()


def get_refusal_score(logits: Float[Tensor, "d_vocab"]):
    return logits[refusal_tokens].mean() - logits[answer_tokens].mean()


def get_refusal_score_avg(logits: Float[Tensor, 'batch seq_len n_vocab']) -> float:
    assert (logits.ndim == 3 or logits.ndim == 2)
    if logits.ndim == 2:
        return logits[:, refusal_tokens].mean() - logits[:, answer_tokens].mean() 
    else:
        return logits[:, -1, refusal_tokens].mean() - logits[:, -1, answer_tokens].mean() 


def metric(
    logits: Union[Float[Tensor, "batch seq_len d_vocab"], Float],
    corrupted_logit_diff: float = -9.0,
    clean_logit_diff: float = +13.7,
 ):
    if isinstance(logits, float):
        return (logits - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
    patched_logit_diff = get_refusal_score_avg(logits)
    return (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)


def calc_activation_avg(node):
    avg_activation = torch.zeros((4096), dtype=torch.float64, device=device)

    def hook_fn(activations, hook, avg_activation=avg_activation):
        avg_activation += activations[:, open_bracket_pos, :].sum(dim=0)

    tmp_hook_fn = functools.partial(hook_fn, avg_activation=avg_activation)

    with torch.no_grad():
        tl_model.run_with_hooks(clean_toks[:instruction_len], fwd_hooks=[(resid_node_open_bracket.activation_name, tmp_hook_fn)], return_type=None)
        tl_model.run_with_hooks(corrupt_toks[:instruction_len],fwd_hooks=[(resid_node_open_bracket.activation_name, tmp_hook_fn)],return_type=None)

    avg_activation /= len(corrupt_toks) + len(clean_toks)
    avg_activation = avg_activation.type(torch.float32)
    return avg_activation

def orth_loss_fn(patcher):
    matrix = patcher.patch_impl.rotation.R.weight
    return torch.linalg.matrix_norm(matrix.T @ matrix - torch.eye(matrix.shape[0]).to(matrix.device), ord='fro')

def loss_fn(logits: Float[Tensor, "batch_size seq_len vocab_size"], tokens: Int[Tensor, "batch_size seq_len"], instruction_len):
    # we only take the logits for the tokens after the instruction
    logits_for_loss = logits[:, instruction_len-1:-1, :].contiguous()
    labels_for_loss = tokens[:, instruction_len:].contiguous()

    loss = F.cross_entropy(
        logits_for_loss.view(-1, logits_for_loss.size(-1)),
        labels_for_loss.view(-1),
        ignore_index=0
    )

    return loss


def eval(patcher, model, clean_tokens, corrupt_tokens, instruction_len, batch_size=100):
    with torch.no_grad():
        patched_logits = patcher.run_patching(model, clean_tokens, corrupt_tokens, batch_size=batch_size)

        # clean_metric = metric(clean_logits[:, instruction_len-1]).item()
        patched_metric = metric(patched_logits[:, instruction_len-1]).item()
        # clean_loss = loss_fn(clean_logits, clean_tokens, instruction_len)
        patched_loss = loss_fn(patched_logits, clean_tokens, instruction_len)

        # print(f'Eval (loss) {patched_loss:.2f} (clean loss) {clean_loss:.2f} (patched metric) {patched_metric:+.2f} (clean metric) {clean_metric:+.2f}')
        print(f'Eval (loss) {patched_loss:.2f} (patched metric) {patched_metric:+.2f}')


def train(
    model: HookedTransformer,
    clean_tokens: Int[Tensor, "batch_size seq_len"],
    corrupt_tokens: Int[Tensor, "batch_size seq_len"],
    patcher: Patcher,
    save_patcher_path: str,
    log_path: str,
    instruction_len: int,
    n_epochs: int = 30,
    initial_lr: float = 0.01,
    batch_size: int = 10,
    eval_every: int = 1,
    optim_metric: dict={"target_tokens": 1.0, "orthogonal": 0.0, "refusal_score": 0.0},
    verbose: bool = True
):
    assert clean_tokens.shape == corrupt_tokens.shape, "Datasets must be of the same size"
    model.reset_hooks()

    pbar = tqdm.tqdm(range(n_epochs))
    optimizer = torch.optim.Adam(patcher.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, end_factor=0.1, total_iters=n_epochs
    )

    train_len = int(0.8 * len(clean_tokens))
    print(f'Train len {train_len}')

    all_metrics = []
    current_metrics = {
        "metric_clean->corrupt": 0.0,
        "metric_corrupt->clean": 0.0,
        "loss_clean->corrupt": 0.0,
        "loss_corrupt->clean": 0.0,
    }

    clean_cache, corrupt_cache = patcher.fill_cache(
        model,
        clean_tokens[:train_len],
        corrupt_tokens[:train_len],
    )

    # we remove the file extension because
    # the patcher might have multiple patch implementations (different rotation matrices)
    save_patcher_path = save_patcher_path.replace(".pt", "")

    # print("Eval clean->corrupt")
    # eval(patcher, model, clean_tokens[:train_len], corrupt_tokens[:train_len], instruction_len)

    # print("Eval corrupt->clean")
    # eval(patcher, model, corrupt_tokens[:train_len], clean_tokens[:train_len], instruction_len)

    with torch.enable_grad():
        for epoch in pbar:
            torch.cuda.empty_cache() # for the folks out there with small GPUs :)
            total_epoch_loss = 0.0
            epoch_losses = {"clean->corrupt": 0.0, "corrupt->clean": 0.0}
            epoch_refusal_metric = {"clean->corrupt": 0.0, "corrupt->clean": 0.0}

            for batch_idx in range(0, train_len, batch_size):
                batch_end = min(batch_idx + batch_size, train_len)
                assert batch_end > batch_idx, "Batch size must be greater than 0"

                for direction in ["clean->corrupt", "corrupt->clean"]:
                    
                    if direction == "clean->corrupt":
                        base_tokens = clean_tokens[batch_idx : batch_end]
                        source_tokens = corrupt_tokens[batch_idx : batch_end]
                        base_cache = clean_cache
                        source_cache = corrupt_cache
                    else:
                        base_tokens = corrupt_tokens[batch_idx : batch_end]
                        source_tokens = clean_tokens[batch_idx : batch_end]
                        base_cache = corrupt_cache
                        source_cache = clean_cache

                    patched_logits = patcher.run_patching(
                        model,
                        base_tokens,
                        source_tokens,
                        cache_base=base_cache,
                        cache_source=source_cache,
                        batch_size=batch_size
                    )

                    if torch.isnan(patched_logits).any():
                        print("NaNs in patched logits")
                        print(patched_logits)
                        assert False, "NaNs in patched logits"

                    loss = torch.zeros(1, device='cuda', requires_grad=True)

                    if "target_tokens" in optim_metric and optim_metric["target_tokens"] > 0.0:
                        multiplier = optim_metric["target_tokens"]
                        loss = loss + multiplier * loss_fn(patched_logits, base_tokens, instruction_len)

                    # if "orthogonal" in optim_metric:
                    #     multiplier = optim_metric["orthogonal"]
                    #     if multiplier > 0.0:
                    #         orth_loss = orth_loss_fn(patcher.patch_impl.rotation.R.weight)
                    #         loss = loss + multiplier * orth_loss_fn(patcher.patch_impl.rotation.R.weight)

                    if "refusal_score" in optim_metric and optim_metric["refusal_score"] > 0.0:
                        multiplier = optim_metric["refusal_score"]
                        if direction == "clean->corrupt":
                            loss = loss + multiplier * metric(patched_logits[:, instruction_len-1])
                        else:
                            loss = loss + multiplier * (1 - metric(patched_logits[:, instruction_len-1]))

                    # KL divergence loss:
                    #     # patched_probs = F.softmax(patched_logits, dim=-1) # ?
                    #     patched_log_probs = torch.log_softmax(patched_logits, dim=-1)
                    #     corrupt_probs = F.softmax(corrupt_logits, dim=-1)
                    #     loss = criterion(patched_log_probs, corrupt_probs)
                    #     print('Loss', loss)

                    print('Clearing memory before backward pass')
                    clear_memory()

                    optimizer.zero_grad()
                    loss.backward()

                    model.zero_grad()
                    optimizer.step()

                    # This is tricky to do because it changes the norms of the activations drastically
                    # patcher.patch_impl.rotation.orthogonalize()

                    epoch_refusal_metric[direction] += metric(patched_logits[:, instruction_len-1]).item() * (batch_end - batch_idx)
                    epoch_losses[direction] += loss.item() * (batch_end - batch_idx)
                    total_epoch_loss += loss.item() * (batch_end - batch_idx)

                    if verbose:
                        print(f'[{batch_idx} - {batch_end}] Loss', loss.item(), flush=True)
                        # if (batch_idx / batch_size) % 5 == 0:
                        #     orth_loss = orth_loss_fn(patcher)
                        #     print(f'[{batch_idx} - {batch_end}] Orth loss', orth_loss.item())

            scheduler.step()

            current_metrics["epoch"] = epoch
            current_metrics["loss_clean->corrupt"] = epoch_losses["clean->corrupt"] / train_len
            current_metrics["loss_corrupt->clean"] = epoch_losses["corrupt->clean"] / train_len
            current_metrics["metric_clean->corrupt"] = epoch_refusal_metric["clean->corrupt"] / train_len
            current_metrics["metric_corrupt->clean"] = epoch_refusal_metric["corrupt->clean"] / train_len
            all_metrics.append(current_metrics.copy())

            save_path = save_patcher_path + f".pt"
            save_logs(all_metrics, log_path)
        
            if epoch % 50 == 0 or epoch == n_epochs - 1:
                save_patcher(patcher, save_path)

            if epoch % eval_every == 0 and verbose:
                print("Eval clean->corrupt")
                eval(patcher, model, clean_tokens[train_len:], corrupt_tokens[train_len:], instruction_len)

                print("Eval corrupt->clean")
                eval(patcher, model, corrupt_tokens[train_len:], clean_tokens[train_len:], instruction_len)

    torch.cuda.empty_cache()
    return current_metrics