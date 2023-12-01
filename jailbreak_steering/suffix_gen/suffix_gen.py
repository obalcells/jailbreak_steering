import torch
import torch.nn as nn
import time
import os
import json
import time
import einops

from torch import Tensor
from jaxtyping import Int, Float
from typing import Optional, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

from jailbreak_steering.suffix_gen.prompt_manager import PromptManager

class SuffixGen():

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer, 
        instruction: str,
        target: str,
        control_init: str,
        system_prompt: Optional[str],
        early_stop_threshold: float,
        allow_nonascii: bool=False,
        verbose: bool=False,
        logs_dir: Optional[str]=None,
        use_cache: bool=True,
    ):
        self.instruction = instruction
        self.target = target
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.early_stop_threshold = early_stop_threshold
        self.verbose = verbose
        self.allow_nonascii = allow_nonascii
        self._nonascii_toks = get_nonascii_toks(self.tokenizer).to(self.model.device)

        self.prompt = PromptManager(
            instruction,
            target,
            tokenizer,
            system_prompt,
            control_init,
            device=self.model.device,
        )

        self.log_path = os.path.join(logs_dir, f'suffix_gen_{time.strftime("%Y%m%d-%H:%M:%S")}.json')
        self.log = self._initialize_log()

        self.use_cache = use_cache
        if self.use_cache:
            self.past_key_values = self.model(
                self.prompt.input_ids[:self.prompt._instruction_slice.stop].unsqueeze(0),
                use_cache=True,
            ).past_key_values

    def run(self, max_steps: int, batch_size: int, topk: int):

        self._initialize_run_log(batch_size, topk)

        best_loss = float('inf')
        best_control = self.prompt.control_str

        for step in range(max_steps):
            start = time.time()

            control, loss = self.step(batch_size=batch_size, topk=topk)
            self.prompt.control_str = control

            self._log_step(step, control, loss, time.time() - start)

            if loss < best_loss:
                best_loss = loss
                best_control = control
            if loss < self.early_stop_threshold:
                break

            if (step+1) % 10 == 0:
                self._save_logs(self.log)

        self._save_logs(self.log)
        return best_control, best_loss, step

    def step(self, batch_size: int, topk: int):

        # 1. Compute gradients of target loss with respect to each control token
        control_tok_grads = self.compute_control_token_gradients()

        # 2. Get candidate control token modifications
        with torch.no_grad():
            sampled_control_cands_toks = self.sample_control_candidates(control_tok_grads, batch_size, topk, curr_control_toks=self.prompt.control_toks)
        control_cands, control_cands_toks = self.filter_cand_controls(sampled_control_cands_toks, curr_control=self.prompt.control_str)

        # 3. Evaluate each control token modification's target loss, pick the best one
        with torch.no_grad():
            losses = self.evaluate_control_cand_losses(control_cands_toks).detach().cpu()
        min_idx = losses.argmin()

        next_control, cand_loss = control_cands[min_idx], losses[min_idx].item()

        return next_control, cand_loss

    # Computes the gradient of the target loss w.r.t. each control token one-hot vector
    def compute_control_token_gradients(self) -> Float[Tensor, "suffix_len d_vocab"]:

        input_ids = self.prompt.input_ids
        control_slice = self.prompt._control_slice
        target_slice = self.prompt._target_slice
        loss_slice = self.prompt._loss_slice

        embed_weights = self.model.model.embed_tokens.weight

        suffix_len = control_slice.stop - control_slice.start
        d_vocab = embed_weights.shape[0]

        # Create one-hot vector for control tokens.
        # We need to compute the gradient of loss w.r.t. these one-hot vectors
        control_toks_one_hot = torch.zeros(suffix_len, d_vocab, dtype=embed_weights.dtype, device=self.model.device)
        control_toks_one_hot.scatter_(1, input_ids[control_slice].unsqueeze(1), 1)
        control_toks_one_hot.requires_grad_()

        # Weave the grad-enabled control embeddings in with the other embeddings
        input_embeds = self.model.model.embed_tokens(input_ids.unsqueeze(0)).detach()
        control_embeds = (control_toks_one_hot @ embed_weights).unsqueeze(0)
        full_embeds = torch.cat(
            [
                input_embeds[:, :control_slice.start, :],
                control_embeds,
                input_embeds[:, control_slice.stop:, :]
            ],
            dim=1
        )

        # Run model and compute loss
        logits = self.model(inputs_embeds=full_embeds).logits
        targets = input_ids[target_slice]
        loss = nn.CrossEntropyLoss()(logits[0, loss_slice, :], targets) 

        # Compute gradients
        loss.backward()
        grad = control_toks_one_hot.grad.clone()

        # Clean up
        self.model.zero_grad()

        return grad

    # Sample candidate control token modifications based on gradient values
    def sample_control_candidates(
        self,
        control_tok_grads: Float[Tensor, "suffix_len d_vocab"],
        batch_size: int,
        topk: int,
        curr_control_toks: Int[Tensor, "suffix_len"],
    ) -> Int[Tensor, "batch_size suffix_len"]:

        suffix_len = curr_control_toks.shape[0]

        # prevent sampling of non-ascii tokens
        if not self.allow_nonascii:
            control_tok_grads[:, self._nonascii_toks] = float('inf')

        # prevent resampling of current control tokens
        control_tok_grads[torch.arange(0, control_tok_grads.shape[0], device=self.model.device), curr_control_toks] = float('inf')

        _, topk_indices = (-control_tok_grads).topk(topk, dim=1)

        original_control_toks = curr_control_toks.repeat(batch_size, 1).to(self.model.device)

        new_token_pos = torch.arange(
            0,
            suffix_len,
            suffix_len / batch_size,
            device=self.model.device
        ).type(torch.int64)

        # Sample random indices in a way that minimizes resampling of the same token
        rand_indices = torch.randperm(topk, device=self.model.device).repeat(batch_size // topk + 1)[:batch_size].unsqueeze(1)

        new_token_val = torch.gather(
            topk_indices[new_token_pos], 1, 
            rand_indices,
        )

        new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

        return new_control_toks

    # Filters candidate controls that do not preserve the # of tokens in the control sequence
    def filter_cand_controls(
        self,
        control_cand: Int[Tensor, "batch_size suffix_len"],
        curr_control: str,
    ) -> Tuple[List[str], Int[Tensor, "batch_size suffix_len"]]:

        decoded_cands = self.tokenizer.batch_decode(control_cand, skip_special_tokens=True)

        re_encoded_cands = self.tokenizer(
            decoded_cands,
            add_special_tokens=False,
            padding=False,
        ).input_ids

        valid_cand_idxs = []

        for i in range(control_cand.shape[0]):
            if decoded_cands[i] != curr_control and len(re_encoded_cands[i]) == len(control_cand[i]):
                valid_cand_idxs.append(i)

        cands = [decoded_cands[i] for i in valid_cand_idxs]
        cands_toks = [control_cand[i] for i in valid_cand_idxs]

        if len(cands) == 0:
            cands.append(curr_control)
            cands_toks.append(torch.tensor(self.tokenizer(curr_control, add_special_tokens=False).input_ids).to(self.model.device))

        cands_toks = torch.stack(cands_toks, dim=0)

        return cands, cands_toks

    def evaluate_control_cand_losses(self, control_cands_toks: Int[Tensor, "batch_size suffix_len"]):
        cands_input_ids = self.prompt.input_ids.repeat(control_cands_toks.shape[0], 1)
        cands_input_ids[:, self.prompt._control_slice] = control_cands_toks

        if self.use_cache:
            cache_offset = self.prompt._instruction_slice.stop
            past_key_values = self.prepare_past_key_values(batch_size=cands_input_ids.shape[0])
        else:
            cache_offset = 0
            past_key_values = None

        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True); start.record()
        logits = self.model(
            cands_input_ids[:, cache_offset:],
            use_cache=self.use_cache,
            past_key_values=past_key_values
        ).logits
        end.record(); torch.cuda.synchronize(); print(f'model.fwd: {start.elapsed_time(end):.5f}')

        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True); start.record()
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        losses = loss_fn(
            logits[:,self.prompt._loss_slice.start - cache_offset : self.prompt._loss_slice.stop - cache_offset,:].transpose(1,2), # batch_size loss_slice d_vocab -> batch_size d_vocab loss_slice
            cands_input_ids[:, self.prompt._target_slice] # batch_size loss_slice
        ).mean(dim=-1)
        end.record(); torch.cuda.synchronize(); print(f'loss_fn: {start.elapsed_time(end):.5f}')

        return losses

    def prepare_past_key_values(self, batch_size: int) -> Tuple[Tuple[Float[Tensor, "batch_size n_head seq_len d_head"]]]:
        past_key_values = [
            [
                einops.repeat(
                    kv[0, :, :, :],
                    'n_head seq_len d_head -> batch_size n_head seq_len d_head',
                    batch_size=batch_size
                )
                for kv in layer
            ]
            for layer in self.past_key_values
        ]
        return tuple(map(tuple, past_key_values))

    def _initialize_log(self):
        log = {"instruction": self.instruction, "target": self.target, "system_prompt": self.system_prompt}
        self._save_logs(log)
        return log

    def _save_logs(self, log):
        log_dir_path = os.path.dirname(self.log_path)
        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path)
        with open(self.log_path, "w") as file:
            json.dump(log, file, indent=4)

    def _log_step(self, step, control, loss, step_runtime):
        if self.verbose:
            self._print_step_info(step, control, loss, step_runtime)
        self._update_log(step, control, loss, step_runtime)

    def _print_step_info(self, step, control, loss, step_runtime):
        control_length = len(self.tokenizer(control, add_special_tokens=False).input_ids)
        print(f'Current length: {control_length}, Control str: {control}')
        print(f'Step: {step}, Current Loss: {loss:.5f}, Last runtime: {step_runtime:.5f}')
        print(" ", flush=True)

    def _initialize_run_log(self, batch_size, topk):
        self.log.update({
            "n_steps": -1,
            "batch_size": batch_size,
            "topk": topk,
            "control_strs": [],
            "losses": [],
            "runtime": 0,
        })

    def _update_log(self, step, control, loss, step_runtime):
        self.log["control_str"] = control
        self.log["control_strs"].append(control)
        self.log["n_steps"] = step
        self.log["losses"].append(loss)
        self.log["runtime"] += step_runtime

def get_nonascii_toks(tokenizer, device='cpu'):
    def is_ascii(s):
        return s.isascii() and s.isprintable()

    ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            ascii_toks.append(i)
    
    if tokenizer.bos_token_id is not None:
        ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        ascii_toks.append(tokenizer.unk_token_id)
    
    return torch.tensor(ascii_toks, device=device)