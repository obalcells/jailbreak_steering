import torch
import torch.nn as nn
import time
import os
import json
import time
import einops
import gc

from torch import Tensor
from jaxtyping import Int, Float
from typing import Optional, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

from jailbreak_steering.suffix_gen.prompt_manager import PromptManager

LOG_FLUSH_INTERVAL = 10
MAX_EVAL_BATCH_SIZE = 128 # limits the effective batch size for evaluations, useful to limit memory usage

class SuffixGen():

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer, 
        instruction: str,
        target: str,
        control_init: str,
        system_prompt: Optional[str],
        topk: int,
        batch_size: int,
        max_steps: int,
        early_stop_threshold: float,
        log_path: str,
        allow_nonascii: bool=False,
        verbose: bool=False,
        use_cache: bool=True,
    ):
        if batch_size > MAX_EVAL_BATCH_SIZE and batch_size % MAX_EVAL_BATCH_SIZE != 0:
            raise ValueError(f"batch_size must be a multiple of {MAX_EVAL_BATCH_SIZE} when batch_size > {MAX_EVAL_BATCH_SIZE}")

        self.model = model
        self.tokenizer = tokenizer

        self.prompt = PromptManager(
            instruction,
            target,
            tokenizer,
            system_prompt,
            control_init,
            device=self.model.device,
        )

        self.topk = topk
        self.batch_size = batch_size
        self.max_steps = max_steps

        self.early_stop_threshold = early_stop_threshold
        self.verbose = verbose

        self.allow_nonascii = allow_nonascii
        if not self.allow_nonascii:
            self._nonascii_toks = get_nonascii_toks(self.tokenizer, self.model.device)

        self.log_path = log_path

        self.use_cache = use_cache
        if self.use_cache:
            self.past_key_values = self.model(
                self.prompt.input_ids[:self.prompt._instruction_slice.stop].unsqueeze(0),
                use_cache=True,
            ).past_key_values
            self.past_key_values = self.prepare_past_key_values(batch_size=min(self.batch_size, MAX_EVAL_BATCH_SIZE))

    def run(self):
        self._initialize_log()

        best_loss = float('inf')
        best_control = self.prompt.control_str

        for n_step in range(self.max_steps):

            control, loss = self.step(batch_size=self.batch_size, topk=self.topk)
            self.prompt.control_str = control

            self._log_step(n_step, control, loss)

            if loss < best_loss:
                best_loss, best_control = loss, control
            if loss < self.early_stop_threshold:
                break

            if (n_step+1) % LOG_FLUSH_INTERVAL == 0:
                self._save_log()
                print(f"full_prompt: {repr(self.prompt.full_prompt_str)}")
                print(f"full_prompt: {repr(self.tokenizer.decode(self.prompt.input_ids))}")
                print(f"loss_str: {repr(self.prompt.loss_str)}")
                print(f"loss_str: {repr(self.tokenizer.decode(self.prompt.input_ids[self.prompt._loss_slice]))}")
                print(f"target_str: {repr(self.prompt.target_str)}")
                print(f"target_str: {repr(self.tokenizer.decode(self.prompt.input_ids[self.prompt._target_slice]))}")
                print(f"control_str: {repr(self.prompt.control_str)}")
                print(f"control_str: {repr(self.tokenizer.decode(self.prompt.input_ids[self.prompt._control_slice]))}")

        self._finalize_log(best_control, best_loss)
        self._save_log()
        return best_control, best_loss, n_step

    def step(self, batch_size: int, topk: int):

        # 1. Compute gradients of target loss with respect to each control token
        control_tok_grads = self.compute_control_token_gradients()

        with torch.no_grad():
            # 2. Get candidate control token modifications
            sampled_control_cands_toks = self.sample_control_candidates(control_tok_grads, batch_size, topk, curr_control_toks=self.prompt.control_toks)
            control_cands, control_cands_toks = self.filter_cand_controls(sampled_control_cands_toks, curr_control=self.prompt.control_str)

            # 3. Evaluate each control token modification's target loss, pick the best one
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

        # embed_weights = self.model.model.embed_tokens.weight
        embed_weights = self.model.transformer.wte.weight

        suffix_len = control_slice.stop - control_slice.start
        d_vocab = embed_weights.shape[0]

        # Create one-hot vector for control tokens.
        # We need to compute the gradient of loss w.r.t. these one-hot vectors
        control_toks_one_hot = torch.zeros(suffix_len, d_vocab, dtype=embed_weights.dtype, device=self.model.device)
        control_toks_one_hot.scatter_(1, input_ids[control_slice].unsqueeze(1), 1)
        control_toks_one_hot.requires_grad_()

        # Weave the grad-enabled control embeddings in with the other embeddings
        # input_embeds = self.model.model.embed_tokens(input_ids.unsqueeze(0)).detach()
        input_embeds = self.model.transformer.wte(input_ids.unsqueeze(0)).detach()
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

        cur_prompt_tok_len = self.prompt.input_ids.shape[-1]

        # we need to check that each candidate yields a tokenized prompt length matching the current one

        decoded_cands = self.tokenizer.batch_decode(control_cand, skip_special_tokens=False)

        prompt_cands = [
            PromptManager(
                instruction=self.prompt.instruction,
                target=self.prompt.target,
                tokenizer=self.prompt.tokenizer,
                system_prompt=self.prompt.system_prompt,
                control_init=cand,
                device=self.prompt.device
            ).full_prompt_str for cand in decoded_cands
        ]

        re_encoded_cand_prompts = self.tokenizer(
            prompt_cands,
            add_special_tokens=False,
            padding=False,
        ).input_ids

        valid_cand_idxs = []

        for i in range(control_cand.shape[0]):
            if (decoded_cands[i] != curr_control and # filter duplicates of current candidate
                len(re_encoded_cand_prompts[i]) == cur_prompt_tok_len # ensure that the number of tokens is preserved
            ):
                valid_cand_idxs.append(i)
            # else:
            #     print(f"Filtered out candidate: {repr(decoded_cands[i])} (len {len(re_encoded_cand_prompts[i])} != {cur_prompt_tok_len})")

        cands = [decoded_cands[i] for i in valid_cand_idxs]
        cands_toks = [control_cand[i] for i in valid_cand_idxs]

        if len(cands) == 0:
            cands.append(curr_control)
            cands_toks.append(torch.tensor(self.tokenizer(curr_control, add_special_tokens=False).input_ids).to(self.model.device))

        # pad to batch_size so that it aligns with prev_key_values cache
        cands = cands + [cands[-1]] * (self.batch_size - len(cands))
        cands_toks = cands_toks + [cands_toks[-1]] * (self.batch_size - len(cands_toks))

        cands_toks = torch.stack(cands_toks, dim=0)

        print(f"Max len of cand token: {cands_toks.shape[-1]}")

        return cands, cands_toks

    def evaluate_control_cand_losses(self, control_cands_toks: Int[Tensor, "batch_size suffix_len"]):
        cands_input_ids = self.prompt.input_ids.repeat(control_cands_toks.shape[0], 1)
        cands_input_ids[:, self.prompt._control_slice] = control_cands_toks

        if self.use_cache:
            cache_offset = self.prompt._instruction_slice.stop
            past_key_values = self.past_key_values
        else:
            cache_offset = 0
            past_key_values = None

        if self.batch_size > MAX_EVAL_BATCH_SIZE:
            batched_logits = []
            for i in range(self.batch_size // MAX_EVAL_BATCH_SIZE):
                logits = self.model(
                    cands_input_ids[i*MAX_EVAL_BATCH_SIZE:(i+1)*MAX_EVAL_BATCH_SIZE, cache_offset:],
                    use_cache=self.use_cache,
                    past_key_values=past_key_values
                ).logits
                batched_logits.append(logits)
            logits = torch.cat(batched_logits, dim=0)
        else:
            logits = self.model(
                cands_input_ids[:, cache_offset:],
                use_cache=self.use_cache,
                past_key_values=past_key_values
            ).logits

        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        losses = loss_fn(
            logits[:,self.prompt._loss_slice.start - cache_offset : self.prompt._loss_slice.stop - cache_offset,:].transpose(1,2), # batch_size loss_slice d_vocab -> batch_size d_vocab loss_slice
            cands_input_ids[:, self.prompt._target_slice] # batch_size loss_slice
        ).mean(dim=-1)

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

    def _save_log(self):
        log_dir_path = os.path.dirname(self.log_path)
        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path)
        with open(self.log_path, "w") as file:
            json.dump(self.log, file, indent=4)

    def _initialize_log(self):
        self.log = {
            "instruction": self.prompt.instruction,
            "target": self.prompt.target,
            "system_prompt": self.prompt.system_prompt,
            "control_init": self.prompt.control_str,
            "batch_size": self.batch_size,
            "topk": self.topk,
            "steps": [],
            "use_cache": self.use_cache,
            "allow_nonascii": self.allow_nonascii,
            "start_time": time.time(),
        }

    def _finalize_log(self, best_control: str, best_loss: float):
        self.log["best_control"] = best_control
        self.log["best_loss"] = best_loss
        self.log["end_time"] = time.time()
        self.log["runtime"] = self.log["end_time"] - self.log["start_time"]
        self.log["n_steps"] = len(self.log["steps"])

    def _log_step(self, n_step: int, control: str, loss: float):
        step = {
            "n_step": n_step,
            "control": control,
            "loss": loss,
            "time": time.time(),
        }
        if self.verbose:
            print(step, flush=True)
        self.log["steps"].append(step)

def get_nonascii_toks(tokenizer, device):
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