import torch
import torch.nn as nn
import time
import os
import gc
import json

from torch import Tensor
from jaxtyping import Int, Float
from typing import Optional, List
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
    ):
        self.instruction = instruction
        self.target = target
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.early_stop_threshold = early_stop_threshold
        self.verbose = verbose
        self.allow_nonascii = allow_nonascii
        self._nonascii_toks = get_nonascii_toks(self.tokenizer)

        self.prompt = PromptManager(
            instruction,
            target,
            tokenizer,
            system_prompt,
            control_init
        )

        self.log_path = os.path.join(logs_dir, f'suffix_gen_{time.strftime("%Y%m%d-%H:%M:%S")}.json')
        self.log = self._initialize_log()

    def run(
        self, 
        n_steps: int, 
        batch_size: int, 
        topk: int,
    ):
        self._initialize_run_log(batch_size, topk)

        best_loss = float('inf')
        best_control = self.prompt.control_str

        for step in range(n_steps):
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

    def step(
        self,
        batch_size: int,
        topk: int,
    ):
        # 1. Compute gradients of target loss with respect to each control token
        control_tok_grads = self.compute_control_token_gradients()

        # 2. Get candidate control token modifications
        with torch.no_grad():
            sampled_control_cands = self.sample_control_candidates(control_tok_grads, batch_size, topk)

        control_cands = self.filter_cand_controls(sampled_control_cands, curr_control=self.prompt.control_str)

        # 3. Evaluate each control token modification's target loss , pick the best one
        with torch.no_grad():
            losses = self.evaluate_control_cand_losses(control_cands)
        min_idx = losses.argmin()
        next_control, cand_loss = control_cands[min_idx], losses[min_idx]

        return next_control, cand_loss.item()

    # Computes the gradient of the target loss w.r.t. each control token one-hot vector
    def compute_control_token_gradients(self) -> Float[Tensor, "suffix_len d_vocab"]:

        input_ids = self.prompt.input_ids.to(self.model.device)
        control_slice = self.prompt._control_slice
        target_slice = self.prompt._target_slice
        loss_slice = self.prompt._loss_slice

        embed_weights = self.model.model.embed_tokens.weight

        suffix_len = control_slice.stop - control_slice.start
        d_vocab = embed_weights.shape[0]

        # Create one-hot vector for control tokens.
        # We need to compute the gradient of loss w.r.t. these one-hot vectors
        control_toks_one_hot = torch.zeros(suffix_len, d_vocab, device=self.model.device, dtype=embed_weights.dtype)
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
        gc.collect(); torch.cuda.empty_cache()

        return grad

    # Sample candidate control token modifications based on gradient values
    def sample_control_candidates(
        self,
        control_tok_grads: Float[Tensor, "suffix_len d_vocab"],
        batch_size: int,
        topk: int,
    ) -> Int[Tensor, "batch_size suffix_len"]:
        if not self.allow_nonascii:
            control_tok_grads[:, self._nonascii_toks] = float('inf')

        _, topk_indices = (-control_tok_grads).topk(topk, dim=1)
        control_toks = self.prompt.control_toks.to(control_tok_grads.device)
        original_control_toks = control_toks.repeat(batch_size, 1)

        unique_indices = torch.arange(0, len(control_toks), device=control_tok_grads.device)
        new_token_pos = unique_indices.repeat_interleave(batch_size // len(control_toks))

        new_control_toks = torch.gather(
            topk_indices[new_token_pos], 1, 
            torch.randint(0, topk_indices.shape[-1], (new_token_pos.shape[0], 1), device=topk_indices.device)
        )

        return original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_control_toks)

    # Filters candidate controls that do not preserve the # of tokens in the control sequence
    def filter_cand_controls(self, control_cand: Int[Tensor, "batch_size suffix_len"], curr_control=None) -> List[str]:
        cands = []
        for i in range(control_cand.shape[0]):
            decoded_str = self.tokenizer.decode(control_cand[i], skip_special_tokens=True)
            if decoded_str != curr_control and len(self.tokenizer(decoded_str, add_special_tokens=False).input_ids) == len(control_cand[i]):
                cands.append(decoded_str)

        cands = cands + [cands[-1]] * (len(control_cand) - len(cands))
        return cands
    
    def evaluate_control_cand_losses(
        self,
        control_cands: List[str]
    ):
        max_len = self.prompt._control_slice.stop - self.prompt._control_slice.start
        control_cand_toks = torch.stack([
            torch.tensor(
                self.tokenizer(
                    control_cand,
                    add_special_tokens=False
                ).input_ids[:max_len]
            ) for control_cand in control_cands
        ], dim=0)
        cand_input_ids = self.prompt.input_ids.repeat(control_cand_toks.shape[0], 1)
        cand_input_ids[:, self.prompt._control_slice] = control_cand_toks
        logits = self.model(cand_input_ids.to(self.model.device)).logits

        crit = torch.nn.CrossEntropyLoss(reduction='none')
        losses = crit(
            logits[:,self.prompt._loss_slice,:].transpose(1,2),
            cand_input_ids[:,self.prompt._target_slice].to(self.model.device)
        ).mean(dim=-1)
        return losses
        
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
        control_length = len(self.tokenizer(control).input_ids)
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