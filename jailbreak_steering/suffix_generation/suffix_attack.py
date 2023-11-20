import torch
import torch.nn as nn
import time
import os
import gc
import json
import einops

from torch import Tensor
from jaxtyping import Int, Float
from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM

from jailbreak_steering.suffix_generation.prompt_manager import PromptManager
from jailbreak_steering.suffix_generation.suffix_generation_utils import get_nonascii_toks

class SuffixAttack():

    def __init__(self, 
        instruction: str,
        target: str,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        control_init: str,
        system_prompt: Optional[str],
        early_stop_threshold: float,
        candidate_sampling_strategy: str="uniform",
        candidate_sampling_temperature: float=None,
        allow_nonascii: bool=False,
        verbose: bool=False,
        log_path: Optional[str]=None,
        jailbreak_db_path: Optional[str]=None,
    ):
        self.instruction = instruction
        self.target = target
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.early_stop_threshold = early_stop_threshold
        self.candidate_sampling_strategy = candidate_sampling_strategy
        self.candidate_sampling_temperature = candidate_sampling_temperature
        self.verbose = verbose

        self.allow_nonascii = allow_nonascii
        self._nonascii_toks = get_nonascii_toks(self.tokenizer)

        self.llm_jailbreak_folder_path = os.path.dirname(os.path.abspath(__file__))
        self.jailbreak_db_path = jailbreak_db_path or f"{self.llm_jailbreak_folder_path}/jailbreak_db.json"
        self.log_path = log_path or self._generate_log_path()
        self.prompt = PromptManager(instruction, target, tokenizer, system_prompt, control_init)
        self.log = self._initialize_log()

    def _generate_log_path(self):
        timestamp = time.strftime("%Y%m%d-%H:%M:%S")
        log_path = f"{self.llm_jailbreak_folder_path}/logs/attack_{timestamp}.json"
        return log_path

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

    @property
    def control_str(self):
        return self.prompt.control_str
    
    @control_str.setter
    def control_str(self, control):
        self.prompt.control_str = control
    
    @property
    def control_toks(self) -> Int[Tensor, "seq_len"]:
        return self.prompt.control_toks
    
    @control_toks.setter
    def control_toks(self, control: Int[Tensor, "seq_len"]):
        self.prompt.control_toks = control

    def run(
        self, 
        n_steps: int, 
        batch_size: int, 
        topk: int,
    ):
        self._initialize_run_log(n_steps, batch_size, topk)

        loss = best_loss = float('inf')
        best_control = self.control_str

        self.prompt = PromptManager(
            self.instruction,
            self.target,
            self.tokenizer,
            self.system_prompt,
            control_init=self.control_str,
        )

        for step in range(n_steps):
            start = time.time()
            torch.cuda.empty_cache()
            control, loss = self.step(
                batch_size=batch_size, 
                topk=topk,
            )
            
            self._log_step(step, control, loss, time.time() - start)

            self.control_str = control

            if loss < best_loss:
                best_loss = loss
                best_control = control
            if loss < self.early_stop_threshold:
                break

            if (step+1) % 10 == 0:
                self._save_logs(self.log)

        self._save_logs(self.log)
        return self.control_str, best_control, loss, step

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

        control_cands = self.filter_cand_controls(sampled_control_cands, curr_control=self.control_str)

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

        topk_logits, topk_indices = (-control_tok_grads).topk(topk, dim=1)
        control_toks = self.control_toks.to(control_tok_grads.device)
        original_control_toks = control_toks.repeat(batch_size, 1)

        unique_indices = torch.arange(0, len(control_toks), device=control_tok_grads.device)
        new_token_pos = unique_indices.repeat_interleave(batch_size // len(control_toks))

        if self.candidate_sampling_strategy == "softmax":
            new_control_toks = self._sample_softmax(topk_logits, topk_indices, new_token_pos, batch_size)
        elif self.candidate_sampling_strategy == "uniform":
            new_control_toks = self._sample_uniform(topk_indices, new_token_pos)
        else:
            raise ValueError(f"Invalid candidate sampling strategy: {self.candidate_sampling_strategy}")

        return original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_control_toks)

    def _sample_softmax(self, topk_logits, topk_indices, new_token_pos, batch_size):
        assert self.candidate_sampling_temperature is not None, "Must specify temperature for softmax sampling"

        normalized_logits = (topk_logits.size(1)**(0.5)) * topk_logits / topk_logits.norm(dim=-1, keepdim=True)
        probs = torch.softmax(normalized_logits / self.candidate_sampling_temperature, dim=1)
        sampled_indices = einops.rearrange(
            torch.multinomial(probs, batch_size // len(self.control_toks), replacement=False),
            "suffix_len n_replacements -> (suffix_len n_replacements) 1"
        )

        return torch.gather(topk_indices[new_token_pos], 1, sampled_indices)

    def _sample_uniform(self, topk_indices, new_token_pos):
        return torch.gather(
            topk_indices[new_token_pos], 1, 
            torch.randint(0, topk_indices.shape[-1], (new_token_pos.shape[0], 1), device=topk_indices.device)
        )

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

    def _log_step(self, step, control, loss, runtime):
        if self.verbose:
            self._print_step_info(step, control, loss, runtime)
        self._update_log(step, control, loss, runtime)

    def _print_step_info(self, step, control, loss, runtime):
        control_length = len(self.tokenizer(control).input_ids)
        print(f'Current length: {control_length}, Control str: {control}')
        print(f'Step: {step}, Current Loss: {loss:.5f}, Last runtime: {runtime:.5f}')
        print(" ", flush=True)

    def _initialize_run_log(self, n_steps, batch_size, topk):
        self.log.update({
            "n_steps": n_steps,
            "batch_size": batch_size,
            "topk": topk,
            "candidate_sampling_strategy": self.candidate_sampling_strategy,
            "candidate_sampling_temperature": self.candidate_sampling_temperature,
            "steps": [],
            "losses": [],
            "runtimes": [],
            "suffix_lengths": []
        })

    def _update_log(self, step, control, loss, runtime):
        self.log["control_str"] = control
        self.log["steps"].append(step)
        self.log["losses"].append(loss)
        self.log["runtimes"].append(runtime)
        self.log["suffix_lengths"].append(len(self.tokenizer(control).input_ids))