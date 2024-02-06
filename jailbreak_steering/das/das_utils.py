# Taken from https://github.com/amakelov/activation-patching-illusion/blob/main/model_utils.py
import math
import joblib
import os
import inspect
from tqdm import tqdm
from collections import defaultdict
import functools
from collections import OrderedDict
from abc import ABC, abstractmethod
import json
from pathlib import Path
import random
from typing import Tuple, List, Sequence, Union, Any, Optional, Literal, Iterable, Callable, Dict
import typing
from fancy_einsum import einsum

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import Parameter
from torch import nn
from transformer_lens import HookedTransformer, ActivationCache
from jaxtyping import Float
from torch.utils.data import Dataset, DataLoader

from torch.nn.utils.parametrizations import orthogonal
from torch.nn import functional as F
from transformer_lens import utils
from transformer_lens.hook_points import HookPoint
from fancy_einsum import einsum
import einops
from jaxtyping import Bool, Float, Int
import tqdm
from torch.nn.utils.parametrizations import orthogonal
from torch.nn import functional as F

from jailbreak_steering.utils.utils import generate_with_hooks

class Node:
    """
    Mostly a copy of the one in path_patching.py, we'll see if it diverges
    """

    def __init__(
        self,
        component_name: Literal[
            "z",
            "attn_out",
            "pre",
            "post",
            "mlp_out",
            "resid_pre",
            "resid_post",
            "resid_mid",
            "q",
            "k",
            "v",
            "pattern",
            "attn_scores",
            "result",
            "q_input",
            "k_input",
            "v_input",
            'scale_ln0',
            'scale_ln1',
            'scale_final',
        ],
        layer: Optional[int] = None,
        head: Optional[int] = None,
        neuron: Optional[int] = None,
        seq_pos: Optional[Union[int, slice]] = None,
    ):
        assert isinstance(component_name, str)
        self.component_name = component_name
        if layer is not None:
            assert isinstance(layer, int)
        self.layer = layer
        if head is not None:
            assert isinstance(head, int)
        self.head = head
        if neuron is not None:
            assert isinstance(neuron, int)
        self.neuron = neuron
        if seq_pos is not None:
            assert isinstance(seq_pos, int) or isinstance(seq_pos, slice)
        self.seq_pos = seq_pos

    @property
    def activation_name(self) -> str:
        if self.component_name == 'scale_ln0':
            return utils.get_act_name('scale', layer=self.layer, layer_type='ln0')
        elif self.component_name == 'scale_ln1':
            return utils.get_act_name('scale', layer=self.layer, layer_type='ln1')
        elif self.component_name == 'scale_final':
             return utils.get_act_name('scale', layer=None)
        else:
            return utils.get_act_name(self.component_name, layer=self.layer)

    @property
    def shape_type(self) -> List[str]:
        """
        List of the meaning of each dimension of the full activation for this
        node (i.e., what you'd get if you did `cache[self.activation_name]`).
        
        This is just for reference
        """
        if self.component_name in [
            "resid_pre",
            "resid_post",
            "resid_mid",
            "q_input",
            "k_input",
            "v_input",
        ]:
            return ["batch", "seq", "d_model"]
        elif self.component_name == 'pattern':
            return ["batch", "head", "query_pos", "key_pos"]
        elif self.component_name in ["q", "k", "v", "z"]:
            return ["batch", "seq", "head", "d_head"]
        elif self.component_name in ["result"]:
            return ["batch", "seq", "head", "d_model"]
        elif self.component_name == 'scale':
            return ['batch', 'seq']
        elif self.component_name == 'post':
            return ['batch', 'seq', 'd_mlp']
        else:
            raise NotImplementedError

    @property
    def idx(self) -> Tuple[Union[int, slice, None], ...]:
        """
        Index into the full activation to restrict to layer / head / neuron /
        seq_pos
        """
        if self.neuron is not None:
            raise NotImplementedError
        elif self.component_name in ['pattern', 'attn_scores']:
            assert self.head is not None
            return tuple([slice(None), self.head, slice(None), slice(None)])
        elif self.component_name in ["q", "k", "v", "z", "result"]:
            assert self.head is not None, "head must be specified for this component"
            if self.seq_pos is not None:
                return tuple([slice(None), self.seq_pos, self.head, slice(None)])
            else:
                return tuple([slice(None), slice(None), self.head, slice(None)])
        elif self.component_name == 'scale':
            return tuple([slice(None), slice(None)])
        elif self.component_name == 'post':
            if self.seq_pos is not None:
                return tuple([slice(None), self.seq_pos, slice(None)])
            else:
                return tuple([slice(None), slice(None), slice(None)])
        else:
            if self.seq_pos is not None:
                return tuple([slice(None), self.seq_pos, slice(None)])
            else:
                return tuple([slice(None), slice(None), slice(None)])
    
    @property
    def names_filter(self) -> Callable:
        return lambda x: x in [self.activation_name]
    
    @staticmethod
    def get_names_filter(nodes: List['Node']) -> Callable:
        return lambda x: any(node.names_filter(x) for node in nodes)

    @property
    def needs_head_results(self) -> bool:
        return self.component_name in ['result']
    
    def get_value(self, cache: ActivationCache) -> Tensor:
        return cache[self.activation_name][self.idx]
    
    def __repr__(self) -> str:
        properties = OrderedDict({
            "component_name": self.component_name,
            "layer": self.layer,
            "head": self.head,
            "neuron": self.neuron,
            "seq_pos": self.seq_pos if isinstance(self.seq_pos, int) else f"{self.seq_pos.start}:{self.seq_pos.stop}",
        })
        properties = ", ".join(f"{k}={v}" for k, v in properties.items() if v is not None)
        return f"Node({properties})"


################################################################################
### patching utils
################################################################################
class PatchImplementation(ABC):
    """
    This is a class instead of a function b/c with a function it's hard to
    access the state of the patcher (e.g. the rotation, the direction, who knows
    what). This class is used to store the state of the patcher.
    """

    @abstractmethod
    def __call__(
        self,
        base_activation: Tensor,
        source_activation: Tensor,
    ) -> Tensor:
        """
        base_activation: the activation to patch
        source_activation: the activation to use for the patch
        """
        pass

    @abstractmethod
    def parameters(self) -> Iterable[Parameter]:
        """
        Parameters of the patching function
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the patching function to a file
        """
        pass

class RotationMatrix(nn.Module):
    """
    Parametrized rotation matrix that can be optimized
    """
    def reorthogonalize(self):
        with torch.no_grad():
            u, _, v = torch.svd(self.R.weight)
            self.R.weight.copy_(torch.mm(u, v.t()))  # Update weight to be orthogonal

    def __init__(self, n: int, avg_activation: Tensor=None):
        super().__init__()
        self.R = orthogonal(nn.Linear(n, n, bias=False))
        self.n = n

        nn.init.orthogonal_(self.R.weight)
        # self.R = torch.empty(n, n)
        # torch.nn.init.orthogonal_(weight)
        # self.R = orthogonal(nn.Linear(n, n, bias=True))
        # self.R = orthogonal(nn.Linear(n, n, bias=False))
        # self.R = nn.Parameter(torch.eye(n))
        # if avg_activation is not None:
        #     self.bias = nn.Parameter(data=avg_activation.type(torch.float32).cuda())
        # else:
        #     self.bias = nn.Parameter(data=torch.zeros(n))
        # self.n = n

    def forward(self, x):
        # x is of shape [batch, activation_dim]
        # init_dtype = x.dtype
        # x = x.type(self.bias.dtype)
        # x = x - self.bias
        out = einsum(
            "... activation_dim, activation_dim rotated_dim -> ... rotated_dim",
            x.type(self.R.weight.dtype),
            self.R.weight,
        )
        # out = out.type(init_dtype)
        return out

    def orthogonalize(self):
        pass

    def inverse(self, x):
        # init_dtype = x.dtype
        # x = x.type(self.bias.dtype)
        out = einsum(
            "... rotated_dim, rotated_dim activation_dim -> ... activation_dim",
            x.type(self.R.weight.dtype),
            self.R.weight.T,
        )
        # print(f'Avg norm of in {x.norm(dim=-1).mean()}')
        # print(f'Avg norm of out {out.norm(dim=-1).mean()}')
        # out = out + self.bias
        # out = out.type(init_dtype)
        return out

    @staticmethod
    def load_rotation(path: Union[str, Path]) -> "RotationMatrix":
        data = torch.load(path)
        n = data["n"]
        state_dict = data["state_dict"]
        rotation = RotationMatrix(n=n)
        rotation.R.load_state_dict(state_dict)
        return rotation
    
    @staticmethod
    def load_rotation_old(path: Path, n: int) -> "RotationMatrix":
        sd = torch.load(path)
        rotation = RotationMatrix(n=n)
        rotation.R.load_state_dict(sd)
        return rotation
    
    @staticmethod
    def from_state_dict(sd: Dict[str, torch.Tensor], n: int) -> "RotationMatrix":
        rotation = RotationMatrix(n=n)
        rotation.R.load_state_dict(sd)
        return rotation

    def dump(self, path: str) -> None:
        torch.save(self.state_dict(), path)


class Rotation(PatchImplementation):
    """
    DAS patching a single variable
    """

    def __init__(self, rotation: RotationMatrix, dim: int):
        self.rotation = rotation
        self.dim = dim
        self.last_intermediates = {}

    def __call__(
        self,
        base_activation: Float[Tensor, "batch ..."],
        source_activation: Float[Tensor, "batch ..."],
    ) -> Tensor:
        base_rot = self.rotation(base_activation)
        source_rot = self.rotation(source_activation)
        if len(source_rot.shape) == 2:
            patched_rot = torch.cat([source_rot[:, :self.dim], base_rot[:, self.dim:]], dim=1)
        else:
            patched_rot = torch.cat([source_rot[:, :, :self.dim], base_rot[:, :, self.dim:]], dim=-1)
        return self.rotation.inverse(patched_rot)

    def parameters(self) -> Iterable[Parameter]:
        return self.rotation.parameters()

        # rot_save_path = save_patcher_path.replace(".pt", f"_rot_{i}.pt")
        # patch_impl.rotation.dump(rot_save_path)

    def save(self, path: str) -> None:
        self.rotation.dump(path)


class DirectionPatch(PatchImplementation):
    """
    Patch values along a 1-dim subspace, i.e. a direction in activation
    space.

    Suppose we have vectors u (source), w (base) and a direction v. We want
    to change w to w' so that <w', v> = <u, v>, and <w', z> = <w, z> for all
    z orthogonal to v. We can do this by adding a multiple of v to w:
        w' = w + a * v

    <w', v> = <w, v> + a * <v, v> = <w, v> + a * ||v||_2^2
    We want this to be equal to <u, v>, so we solve for a:
        a = (<u, v> - <w, v>) / ||v||_2^2
    """

    def __init__(self, v: Tensor):
        self.v = v

    def __call__(self, base_activation: Tensor, source_activation: Tensor) -> Tensor:
        v = self.v
        assert base_activation.shape == source_activation.shape
        assert base_activation.shape[1:] == v.shape
        base_proj = einsum("batch ..., ... -> batch", base_activation, v)
        source_proj = einsum("batch ..., ... -> batch", source_activation, v)
        norm = v.norm()
        base_activation += einsum(
            "batch, ... -> batch ...", (source_proj - base_proj) / norm**2, v
        )
        return base_activation

    def parameters(self) -> Iterable[Parameter]:
        return []


################################################################################
### patcher
################################################################################

class Patcher:
    """
    A location where to perform a patch (DAS or other). It is either
        - a single node (e.g. residual stream at a given layer and position), in
          which case it works like you'd expect
        - or several nodes (only head results at the same position are supported
        for this, in which case the patch is applied to the residual stream, as
        soon as all heads have been computed.

    It decouples the
    - "where": which components, which layer, which position (`nodes` argument)
    - "how": this is implemented in the `patch_fn` argument
    - "when": the `get_hook` returns a hook you can combine w/ other hooks and
    whatever to do more complex things (e.g. DAS + path patching)

    """

    def __init__(
        self,
        nodes: List[Node],
        patch_impl_list: List[PatchImplementation],
    ):
        """
        nodes: which activations to patch
        patch_fn: (base activation, source activation) -> patched activation
        """
        self.nodes = nodes
        self.patch_impl_list = patch_impl_list

        assert all(node.layer is not None for node in nodes)
        assert all(node.seq_pos is not None for node in nodes)

    @property
    def needs_head_results(self) -> bool:
        """
        Whether we need to call `compute_head_results` before patching
        """
        return len(self.nodes) > 1

    @property
    def names_filter(self) -> Callable:
        """
        Get a thing that can be used as the `names_filter` argument of
        `model.run_with_cache`. It filters the activations to only keep the
        ones needed for the patching (reducing memory usage).

        Returns:
            Callable: _description_
        """
        # if len(self.nodes) == 1:
        #     return lambda x: x in [self.nodes[0].activation_name]
        # else:
        #     act_names = [node.activation_name for node in self.nodes] + [
        #         self.target_node.activation_name
        #     ]
        #     # unfortunately, `compute_head_results` requires all the `z`
        #     # activations to be present; don't know how to get around it
        #     return lambda x: ("hook_z" in x) or x in act_names
        return lambda x : x in [node.activation_name for node in self.nodes]

    @property
    def target_node(self) -> Node:
        """
        The node at which we do the actual intervention. This is useful if we
        are patching head results, b/c then we patch the sum of heads in the
        residual stream.
        """
        # if len(self.nodes) == 1:
        #     return self.nodes[0]
        # else:
        #     if not all(x.component_name == "result" for x in self.nodes):
        #         raise NotImplementedError("Only head results are supported")
        #     max_layer = max(x.layer for x in self.nodes)
        #     return Node(
        #         component_name="resid_mid",
        #         layer=max_layer,
        #         seq_pos=self.nodes[0].seq_pos,
        #     )
        return self.nodes[0]

    # def patch_slice(
    #     self,
    #     base_slice: Tensor,
    #     source_slice: Tensor,
    #     cache_base: Optional[ActivationCache] = None,
    #     cache_source: Optional[ActivationCache] = None,
    # ) -> Tensor:
    #     """
    #     This runs the actual patching on the *relevant slices* of the
    #     activations, i.e.  what you'd get when you restrict to proper seq_pos,
    #     head, etc.

    #     Returns the slice you should put back in the full activation.
    #     """
    #     if len(self.nodes) == 1:
    #         return self.patch_impl(base_activation=base_slice,
    #                                source_activation=source_slice)
    #     else:
    #         # patch resid contributions
    #         assert cache_base is not None
    #         assert cache_source is not None
    #         idxs = [node.idx for node in self.nodes]
    #         summed_base = sum(
    #             [
    #                 cache_base[node.activation_name][idx]
    #                 for node, idx in zip(self.nodes, idxs)
    #             ]
    #         )
    #         summed_source = sum(
    #             [
    #                 cache_source[node.activation_name][idx]
    #                 for node, idx in zip(self.nodes, idxs)
    #             ]
    #         )
    #         patched = self.patch_impl(
    #             base_activation=summed_base, 
    #             source_activation=summed_source
    #             )
    #         return base_slice - summed_base + patched

    def get_hook(
        self,
        cache_base: Optional[ActivationCache],
        cache_source: ActivationCache,
        slice_method: str = 'mask',
        batch_slice: slice = slice(None),
    ) -> Tuple[str, Callable]:
        """
        Return a pair (activation_name, hook_fn) that can be used to perform the
        full patching.
        """
        def hook_fn(base_activation: Tensor, hook: HookPoint) -> Tensor:
            activation_name = hook.name

            for node, patch_impl in zip(self.nodes, self.patch_impl_list):
                if node.activation_name != activation_name:
                    continue

                idx = node.idx

                base_activation_slice = base_activation[idx]
                source_slice = cache_source[activation_name][batch_slice][idx]

                new_activation_slice = patch_impl(
                    base_activation=base_activation_slice,
                    source_activation=source_slice
                )

                if slice_method == 'obvious' or len(base_activation.shape) == 3:
                    base_activation[idx] = new_activation_slice
                elif slice_method == 'mask':
                    # This is a very weird hack for the in-place backprop problem
                    # some sanity checks for the shapes. The slice should be 2D with
                    # the seq_pos being set to a constant value, and we insert this
                    # in the 3D tensor of the base activation, where the middle dim
                    # encodes the seq_pos
                    assert len(new_activation_slice.shape) == 2
                    assert len(base_activation.shape) == 3
                    assert new_activation_slice.shape[0] == base_activation.shape[0]
                    assert new_activation_slice.shape[1] == base_activation.shape[2]

                    # Construct a boolean mask of the same shape as base_activation
                    mask = torch.zeros_like(base_activation, dtype=torch.bool)
                    mask[idx] = 1
                    # Construct the new tensor using the mask
                    base_activation_new = torch.where(mask, new_activation_slice.unsqueeze(1), base_activation)
                    assert len(self.nodes) == 1, "This is not implemented for multiple nodes"
                    return base_activation_new

            return base_activation

        return (self.names_filter, hook_fn)

    def fill_cache(
        self,
        model: HookedTransformer,
        P_base: Float[Tensor, "batch seq_len"],
        P_source: Float[Tensor, "batch seq_len"],
    ):
        _, cache_base = model.run_with_cache(P_base, names_filter=self.names_filter)
        _, cache_source = model.run_with_cache(P_source, names_filter=self.names_filter)
        return cache_base, cache_source

    def run_patching(
        self,
        model: HookedTransformer,
        P_base: Float[Tensor, "batch seq_len"],
        P_source: Float[Tensor, "batch seq_len"],
        batch_size: Optional[int] = 4,
        cache_base: Optional[ActivationCache] = None,
        cache_source: Optional[ActivationCache] = None,
    ):
        assert len(P_base) == len(P_source)
        assert (cache_base is None and cache_source is None) or (cache_base is not None and cache_source is not None)

        use_cached = cache_base is not None and cache_source is not None
        names_filter = self.names_filter

        logits_patched = None

        for batch_idx in range(0, len(P_base), batch_size):
            batch_end = min(batch_idx + batch_size, len(P_base))

            if not use_cached:
                logits_base, cache_base = model.run_with_cache(
                    P_base[batch_idx:batch_end], names_filter=names_filter
                )
                # if self.needs_head_results:
                #     cache_base.compute_head_results()

                logits_source, cache_source = model.run_with_cache(
                    P_source[batch_idx:batch_end], names_filter=names_filter
                )
                # if self.needs_head_results:
                #     cache_source.compute_head_results()

                batch_slice = slice(None)
            else:
                batch_slice = slice(batch_idx, batch_end)

            hk = self.get_hook(cache_base=cache_base, cache_source=cache_source, batch_slice=batch_slice)
            model.reset_hooks()

            batch_logits_patched = model.run_with_hooks(
                P_base[batch_idx:batch_end], fwd_hooks=[hk]
            )

            if logits_patched is None:
                logits_patched = batch_logits_patched
            else:
                logits_patched = torch.cat([logits_patched, batch_logits_patched], dim=0)

        return logits_patched

    @torch.no_grad()
    def generate_with_patching(
        self,
        model,
        tokenizer,
        clean_tokens,
        corrupt_tokens,
        fwd_hooks=[],
        **kwargs
    ):
        """
        Run generations with model by patching the corrupted activations into the clean activations.
        """
        all_tokens = torch.cat([clean_tokens, corrupt_tokens], dim=0)

        def hook_fn(activations: Tensor, hook: HookPoint) -> Tensor:
            activation_name = hook.name

            for node, patch_impl in zip(self.nodes, self.patch_impl_list):

                if node.activation_name != activation_name:
                    continue

                idx = node.idx

                batch_size = activations.shape[0]
                assert batch_size % 2 == 0, "Batch size must be even (each half is clean or corrupt)"

                clean_batch_slice = slice(0, batch_size // 2)
                corrupt_batch_slice = slice(batch_size // 2, batch_size)

                base_activation_slice = activations[clean_batch_slice][idx]
                source_slice = activations[corrupt_batch_slice][idx]

                new_activation_slice = patch_impl(
                    base_activation=base_activation_slice,
                    source_activation=source_slice
                )

                assert len(activations.shape) == 3

                activations[clean_batch_slice][idx] = new_activation_slice


            return activations

        fwd_hooks.append((self.names_filter, hook_fn))        

        return generate_with_hooks(
            model,
            tokenizer,
            all_tokens,
            fwd_hooks=fwd_hooks,
            **kwargs
        )

    def parameters(self) -> Iterable[Parameter]:
        return sum([list(impl.parameters()) for impl in self.patch_impl_list], [])


    # @batched(
    #     args=["X_base", "X_source"],
    #     n_outputs=1,
    #     reducer="cat",
    #     shuffle=False,
    #     verbose=True,
    # )
    # def get_patched_activation(
    #     self, model: HookedTransformer, 
    #     node: Node,
    #     X_base: Optional[Tensor] = None,
    #     X_source: Optional[Tensor] = None,
    #     cache_base: Optional[ActivationCache] = None,
    #     cache_source: Optional[ActivationCache] = None,
    #     batch_size: Optional[int] = None,
    #     ) -> Tensor:
    #     """
    #     Return the activation of the given node after patching with this
    #     patcher.
    #     """
    #     if cache_base is None:
    #         _, cache_base = model.run_with_cache(X_base, names_filter=self.names_filter)
    #     if cache_source is None:
    #         _, cache_source = model.run_with_cache(X_source, names_filter=self.names_filter)
    #     if self.needs_head_results:
    #         cache_base.compute_head_results()
    #         cache_source.compute_head_results()
    #     hk = self.get_hook(cache_base=cache_base, cache_source=cache_source)
    #     model.reset_hooks()
    #     model.add_hook(name=hk[0], hook=hk[1], dir='fwd', is_permanent=False)
    #     node_filter = node.names_filter
    #     _, cache_patched = model.run_with_cache(X_base, names_filter=node_filter)
    #     model.reset_hooks()
    #     return node.get_value(cache_patched)
