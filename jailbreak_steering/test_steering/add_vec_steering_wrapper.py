# This file is mostly copied from https://github.com/nrimsky/SycophancySteering/blob/main/llama_wrapper.py

import torch
from jailbreak_steering.utils.load_model import load_llama_2_7b_chat_model, load_llama_2_7b_chat_tokenizer
from jailbreak_steering.utils.tokenize_llama_chat import find_string_in_tokens, tokenize_llama_chat, E_INST
from jailbreak_steering.steering_wrapper.abstract_steering_wrapper import AbstractModelWrapper

class ModuleWrapper(torch.nn.Module):
    def __init__(self, module, end_of_instruction_tokens):
        super().__init__()
        self.module = module
        self.END_STR = end_of_instruction_tokens

        self.initial_input_ids = None
 
        self.do_steering = False
        self.steering_vector = None
        # we only add the steering vector at the last 3 positions
        self.prompt_steering_positions = [-3, -2, -1]
        self.steer_at_generation_tokens = True

        self.normalize = True
        self.multiplier = 1.0

    def forward(self, *args, **kwargs):
        output = self.module(*args, **kwargs)

        if not self.do_steering:
            return output

        if isinstance(output, tuple):
            activations = output[0]
        else:
            activations = output

        position_ids = kwargs.get("position_ids", None)

        assert position_ids is not None, "Position ids must be provided"
        assert self.steering_vector is not None, "Steering vector must be set"

        if self.steering_vector.dtype != activations.dtype:
            self.steering_vector = self.steering_vector.type(activations.dtype)

        if len(self.steering_vector.shape) == 1:
            self.steering_vector = self.steering_vector.reshape(1, 1, -1)

        if self.steering_vector.device != activations.device:
            self.steering_vector = self.steering_vector.to(activations.device)

        if activations.shape[1] == 1 and self.steer_at_generation_tokens:
            # steering at generation tokens
            token_positions = [0]
        elif activations.shape[1] > 1:
            # steering at prompt tokens
            token_positions = self.prompt_steering_positions

        norm_pre = torch.norm(activations, dim=-1, keepdim=True)

        activations[:, token_positions] = activations[:, token_positions] + self.steering_vector * self.multiplier

        norm_post = torch.norm(activations, dim=-1, keepdim=True)

        activations = activations / norm_post * norm_pre

        if isinstance(output, tuple):
            output = (activations,) + output[1:] 
        else:
            output = modified

        return output
class AddVecModelWrapper(AbstractModelWrapper):
    def __init__(self):
        super().__init__()

    def prepare

    def wrap_model(self, hf_model, tokenizer):
        self.tokenizer = tokenizer
        self.END_STR = torch.tensor(self.tokenizer.encode("[/INST]")[1:])

        self.model = hf_model

        self.model.model.layers[19] = ModuleWrapper(
            self.model.model.layers[19], self.END_STR
        )

        self.model.model.layers[19].steering_vector = self.jailbreak_vectors[19]
        self.model.model.layers[19].do_steering = False

    def unwrap_model(self):
        self.model.model.layers[19] = self.model.model.layers[19].module
        assert not isinstance(self.model.model.layers[19], ModuleWrapper)



class ModelWrapper(abc):
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def wrap_model(self, hf_model, tokenizer):
        self.tokenizer = tokenizer
        self.END_STR = torch.tensor(self.tokenizer.encode("[/INST]")[1:])

        self.model = hf_model

        # we only wrap layer 19 at the moment
        self.model.model.layers[19] = ModuleWrapper(
            self.model.model.layers[19], self.END_STR
        )
        self.model.model.layers[19].steering_vector = self.jailbreak_vectors[19]
        self.model.model.layers[19].do_steering = False
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def unwrap_model(self):
        self.model.model.layers[19] = self.model.model.layers[19].module
        assert not isinstance(self.model.model.layers[19], ModuleWrapper)

@stub.local_entrypoint()
def run_benchmark():
    benchmark = Benchmark()
    model_wrapper = ModelWrapper()

    jailbreaks = json.load(open("./jailbreak_vector/advbench_jailbreaks.json", "r"))

    data = [
        {
            "instruction": f"{jailbreaks[i]['instruction']} + {jailbreaks[i]['control_str']}",
            "output": "", 
        }
        for i in range(len(jailbreaks))
    ]

    benchmark_output = benchmark.generation.remote(model_wrapper, data=data)

    print("Benchmark output:")
    print(benchmark_output)

    with open("benchmark_output.json", "w") as f:
        json.dump(benchmark_output, f, indent=4)

