# This file is mostly copied from https://github.com/nrimsky/SycophancySteering/blob/main/llama_wrapper.py

import torch as t
from jailbreak_steering.utils.load_model import load_llama_2_7b_chat_model, load_llama_2_7b_chat_tokenizer
from jailbreak_steering.utils.tokenize_llama_chat import find_string_in_tokens, tokenize_llama_chat, E_INST
from jailbreak_steering.utils.steering_utils import add_vector_after_position, find_instruction_end_postion
from typing import Tuple, Optional
from torch import Tensor
from jaxtyping import Int, Float


class AttnWrapper(t.nn.Module):
    """
    Wrapper for attention mechanism to save activations
    """

    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        self.activations = output[0]
        return output


class BlockOutputWrapper(t.nn.Module):
    """
    Wrapper for block to save activations and unembed them
    """

    def __init__(self, block, unembed_matrix, norm, tokenizer):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm
        self.tokenizer = tokenizer

        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        self.attn_out_unembedded = None
        self.intermediate_resid_unembedded = None
        self.mlp_out_unembedded = None
        self.block_out_unembedded = None

        self.activations = None
        self.add_activations = None
        self.after_position = None

        self.save_internal_decodings = False
        self.do_projection = False
        self.normalize = True

        self.calc_dot_product_with = None
        self.dot_products = []

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.activations = output[0]
        if self.calc_dot_product_with is not None:
            last_token_activations = self.activations[0, -1, :]
            decoded_activations = self.unembed_matrix(self.norm(last_token_activations))
            top_token_id = t.topk(decoded_activations, 1)[1][0]
            top_token = self.tokenizer.decode(top_token_id)
            dot_product = t.dot(last_token_activations, self.calc_dot_product_with) / (t.norm(
                last_token_activations
            )  * t.norm(self.calc_dot_product_with))
            self.dot_products.append((top_token, dot_product.cpu().item()))
        if self.add_activations is not None:
            augmented_output = add_vector_after_position(
                matrix=output[0],
                vector=self.add_activations,
                position_ids=kwargs["position_ids"],
                after=self.after_position,
                do_projection=self.do_projection,
                normalize=self.normalize
            )
            output = (augmented_output,) + output[1:]

        if not self.save_internal_decodings:
            return output

        # Whole block unembedded
        self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))

        # Self-attention unembedded
        attn_output = self.block.self_attn.activations
        self.attn_out_unembedded = self.unembed_matrix(self.norm(attn_output))

        # Intermediate residual unembedded
        attn_output += args[0]
        self.intermediate_resid_unembedded = self.unembed_matrix(self.norm(attn_output))

        # MLP unembedded
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_out_unembedded = self.unembed_matrix(self.norm(mlp_output))

        return output

    def add(self, activations, do_projection=False, normalize=True):
        self.add_activations = activations
        self.do_projection = do_projection
        self.normalize = normalize

    def reset(self):
        self.add_activations = None
        self.activations = None
        self.block.self_attn.activations = None
        self.after_position = None
        self.do_projection = False
        self.calc_dot_product_with = None
        self.dot_products = []

class LlamaWrapper:
    def __init__(
        self,
        system_prompt,
        add_only_after_end_str=False,
        override_model_weights_path=None,
    ):
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.system_prompt = system_prompt
        self.add_only_after_end_str = add_only_after_end_str

        self.model = load_llama_2_7b_chat_model()
        self.tokenizer = load_llama_2_7b_chat_tokenizer()

        if override_model_weights_path is not None:
            self.model.load_state_dict(t.load(override_model_weights_path))

        self.model = self.model.to(self.device)
        self.END_STR = t.tensor(self.tokenizer.encode("[/INST]")[1:]).to(self.device)
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(
                layer, self.model.lm_head, self.model.model.norm, self.tokenizer
            )

    def set_save_internal_decodings(self, value: bool):
        for layer in self.model.model.layers:
            layer.save_internal_decodings = value

    def set_after_positions(self, pos: Int[Tensor, "batch_size 1"]):
        for layer in self.model.model.layers:
            layer.after_position = pos

    def generate_text(self, prompt: str, max_new_tokens: int = 50) -> str:
        tokens = tokenize_llama_chat(
            self.tokenizer,
            conversation=[(prompt, None)],
            system_prompt=self.system_prompt,
            no_final_eos=True,
        )
        tokens = t.tensor(tokens).unsqueeze(0).to(self.device)
        output_tokens = self.model.generate(tokens, max_new_tokens=max_new_tokens)
        return self.tokenizer.batch_decode(output_tokens)[0]

    def generate_text_with_conversation_history(
        self, history: Tuple[str, Optional[str]], max_new_tokens=50
    ) -> str:
        tokens = tokenize_llama_chat(
            self.tokenizer,
            conversation=history,
            system_prompt=self.system_prompt,
            no_final_eos=True
        )
        tokens = t.tensor(tokens).unsqueeze(0).to(self.device)
        output_tokens = self.model.generate(tokens, max_new_tokens=max_new_tokens)
        return self.tokenizer.batch_decode(output_tokens)[0]

    def generate(self, tokens, **kwargs):
        kwargs["max_new_tokens"] = 50 if kwargs.get("max_new_tokens") is None else kwargs.get("max_new_tokens")

        with t.no_grad():
            if self.add_only_after_end_str:
                instr_pos = [find_instruction_end_postion(tokens[i], self.END_STR) for i in range(tokens.size(0))]
                instr_pos = t.tensor(instr_pos).unsqueeze(-1).to(self.device)

                # We shift the instruction end index by the number of padding tokens added
                pad_token_id = kwargs.get("pad_token_id", self.tokenizer.pad_token_id)
                instr_pos -= (tokens == pad_token_id).sum(dim=-1, keepdim=True)
                # The vector is added starting from the last token in the end instruction string ('[/INST]')
                self.set_after_positions(instr_pos - 1)
            else:
                instr_pos = None
            output_tokens = self.model.generate(tokens, **kwargs)
            return output_tokens

    def get_logits(self, tokens):
        with t.no_grad():
            if self.add_only_after_end_str:
                instr_pos = [find_instruction_end_postion(tokens[i], self.END_STR) for i in range(tokens.size(0))]
                instr_pos = t.tensor(instr_pos).unsqueeze(-1).to(self.device)
                self.set_after_positions(instr_pos - 1)
            else:
                instr_pos = None
            logits = self.model(tokens).logits
            return logits

    def get_last_activations(self, layer):
        return self.model.model.layers[layer].activations

    def set_add_activations(self, layer, activations, do_projection=False, normalize=True):
        self.model.model.layers[layer].add(activations, do_projection, normalize)

    def set_calc_dot_product_with(self, layer, vector):
        self.model.model.layers[layer].calc_dot_product_with = vector

    def get_dot_products(self, layer):
        return self.model.model.layers[layer].dot_products

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()
