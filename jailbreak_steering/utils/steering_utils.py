from typing import Optional
import os
import torch
from torch import Tensor
from jaxtyping import Int, Float

def project_onto_orthogonal_complement(tensor, onto):
    """
    Projects tensor onto the orthogonal complement of the span of onto.
    """
    # Get the projection of tensor onto onto
    proj = (
        torch.tsum(tensor * onto, dim=-1, keepdim=True)
        * onto
        / (torch.norm(onto, dim=-1, keepdim=True) ** 2 + 1e-10)
    )
    # Subtract to get the orthogonal component
    return tensor - proj


def add_vector_after_position(
    matrix: Float[Tensor, "batch_size seq_len d_model"],
    vector: Float[Tensor, "d_model"],
    position_ids: Int[Tensor, "batch_size seq_len"],
    after: Int[Tensor, "batch_size 1"]=None,
    do_projection=True,
    normalize=True
):
    after_id = after
    if after_id is None:
        after_id = position_ids.min(dim=1).item() - 1

    mask = position_ids > after_id
    mask = mask.unsqueeze(-1)

    norm_pre = torch.norm(matrix, dim=-1, keepdim=True)

    if do_projection:
        matrix = project_onto_orthogonal_complement(matrix, vector)

    matrix += mask.float() * vector

    norm_post = torch.norm(matrix, dim=-1, keepdim=True)

    if normalize:
        matrix *= norm_pre / norm_post

    return matrix


def find_last_subtensor_position(tensor, sub_tensor):
    n, m = tensor.size(0), sub_tensor.size(0)
    if m > n:
        return -1
    for i in range(n - m, -1, -1):
        if torch.equal(tensor[i : i + m], sub_tensor):
            return i
    return -1


def find_instruction_end_postion(tokens, end_str):
    start_pos = find_last_subtensor_position(tokens, end_str)
    if start_pos == -1:
        return -1
    return start_pos + len(end_str) - 1


def get_a_b_probs(logits, a_token_id, b_token_id):
    last_token_logits = logits[0, -1, :]
    last_token_probs = torch.tsoftmax(last_token_logits, dim=-1)
    a_prob = last_token_probs[a_token_id].item()
    b_prob = last_token_probs[b_token_id].item()
    return a_prob, b_prob


def make_tensor_save_suffix(layer, model_name_path):
    return f'{layer}_{model_name_path.split("/")[-1]}'