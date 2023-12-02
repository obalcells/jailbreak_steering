from dataclasses import dataclass
from typing import Optional
import os

from jailbreak_steering.utils.tokenize_llama_chat import DEFAULT_SYSTEM_PROMPT
from jailbreak_steering.suffix_gen.run_suffix_gen import DEFAULT_VECTORS_DIR, DEFAULT_VECTOR_LABEL

@dataclass
class SteeringSettings:
    """
    max_new_tokens: Maximum number of tokens to generate.
    type: Type of test to run. One of "in_distribution", "out_of_distribution", "truthful_qa".
    few_shot: Whether to test with few-shot examples in the prompt. One of "positive", "negative", "none".
    do_projection: Whether to project activations onto orthogonal complement of steering vector.
    override_vector: If not None, the steering vector generated from this layer's activations will be used at all layers. Use to test the effect of steering with a different layer's vector.
    override_vector_model: If not None, steering vectors generated from this model will be used instead of the model being used for generation - use to test vector transference between models.
    use_base_model: Whether to use the base model instead of the chat model.
    model_size: Size of the model to use. One of "7b", "13b".
    n_test_datapoints: Number of datapoints to test on. If None, all datapoints will be used.
    add_every_token_position: Whether to add steering vector to every token position (including question), not only to token positions corresponding to the model's response to the user
    override_model_weights_path: If not None, the model weights at this path will be used instead of the model being used for generation - use to test using activation steering on top of fine-tuned model.

    layers: Layers to use for steering.
    multipliers: Multipliers to use for steering.
    normalize: Whether to normalize the residual stream to its previous norm after adding the steering vector.
    system_prompt: System prompt to use for generation. If None then no system prompt will be used.
    dataset: The name of the dataset to use for generation. It's always an instruction dataset from HF. Either "advbench" or "alpaca" at the moment.
    vector_label: The label of the steering vectors that will be used to steer.
    """

    max_new_tokens: int = 256 
    type: str = "in_distribution"
    few_shot: str = "none"
    do_projection: bool = False
    override_vector: Optional[int] = None
    override_vector_model: Optional[str] = None
    use_base_model: bool = False
    model_size: str = "7b"
    n_test_datapoints: Optional[int] = None
    add_every_token_position: bool = False
    override_model_weights_path: Optional[str] = None

    layers: List[int] = []
    multipliers: List[int] = []
    normalize: bool = True 
    system_prompt: str
    dataset: str = "advbench"
    vector_label: str = DEFAULT_VECTOR_LABEL

    def make_result_save_suffix(self):
        # we don't want to save the system prompt in the suffix
        if self.system_prompt == DEFAULT_SYSTEM_PROMPT:
            system_prompt_short = "default"
        elif self.system_prompt is None:
            system_prompt_short = "none"
        else:
            system_prompt_short = "custom"

        # we don't want to save the vectors dir in the suffix
        if self.vectors_dir == DEFAULT_VECTORS_DIR:
            vectors_dir_short = "default" 
        else:
            vectors_dir_short = self.vectors_dir.split("/")[:10]

        elements = {
            "layers": self.layers,
            "multipliers": self.multipliers,
            "dataset": self.dataset,
            "few_shot": self.few_shot,
            "do_projection": self.do_projection,
            "normalize": self.normalize,
            "max_new_tokens": self.max_new_tokens,
            "n_test_datapoints": self.n_test_datapoints,
            "add_every_token_position": self.add_every_token_position,
            "system_prompt": system_prompt_short,
            "vectors_dir": vectors_dir_short
        }

        return "_".join([f"{k}={str(v).replace('/', '-')[:10]}" for k, v in elements.items() if v is not None])

    def filter_result_files_by_suffix(
        self,
        directory: str,
        layer: Optional[int] = None,
        multiplier: Optional[int] = None,
    ):
        # we don't want to save the system prompt in the suffix
        if self.system_prompt == DEFAULT_SYSTEM_PROMPT:
            system_prompt_short = "default"
        elif self.system_prompt is None:
            system_prompt_short = "none"
        else:
            system_prompt_short = "custom"

        # we don't want to save the vectors dir in the suffix
        if self.vectors_dir == DEFAULT_VECTORS_DIR:
            vectors_dir_short = "default" 
        else:
            vectors_dir_short = self.vectors_dir.split("/")[:10]

        elements = {
            "layers": self.layers,
            "multipliers": self.multipliers,
            "dataset": self.dataset,
            "few_shot": self.few_shot,
            "do_projection": self.do_projection,
            "normalize": self.normalize,
            "max_new_tokens": self.max_new_tokens,
            "n_test_datapoints": self.n_test_datapoints,
            "add_every_token_position": self.add_every_token_position,
            "system_prompt": system_prompt_short,
            "vectors_dir": vectors_dir_short
        }

        filtered_elements = {k: v for k, v in elements.items() if v is not None}
        remove_elements = {k for k, v in elements.items() if v is None}

        matching_files = []

        print(self.override_model_weights_path)

        for filename in os.listdir(directory):
            if all(f"{k}={str(v).replace('/', '-')}" in filename for k, v in filtered_elements.items()):
                # ensure remove_elements are *not* present
                if all(f"{k}=" not in filename for k in remove_elements):
                    matching_files.append(filename)

        return [os.path.join(directory, f) for f in matching_files]

    def get_settings_dict(self):
        return {
            "layers": self.layers,
            "multipliers": self.multipliers,
            "dataset": self.dataset,
            "few_shot": self.few_shot,
            "do_projection": self.do_projection,
            "normalize": self.normalize,
            "max_new_tokens": self.max_new_tokens,
            "n_test_datapoints": self.n_test_datapoints,
            "add_every_token_position": self.add_every_token_position,
            "system_prompt": self.system_prompt,
            "vectors_dir": self.vectors_dir
        }
