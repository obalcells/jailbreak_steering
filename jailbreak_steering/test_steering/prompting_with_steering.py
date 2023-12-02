import argparse
import json
import os
import numpy as np
from datasets import load_dataset
import tqdm
from dataclasses import dataclass

from jailbreak_steering.utils.load_model import load_llama_2_7b_chat_tokenizer
from jailbreak_steering.utils.steering_settings import SteeringSettings

from jailbreak_steering.utils.tokenize_llama_chat import DEFAULT_SYSTEM_PROMPT
from jailbreak_steering.suffix_gen.run_suffix_gen import DEFAULT_VECTORS_DIR

# TODO: Remove this 
# import importlib
# from steering_wrappers.add_vec_steering_wrapper import AddVecSteeringWrapper 
# Function to import module at the runtime
# def dynamic_import(module):
#     return importlib.import_module(module)

# DEFAULT_STEERING_WRAPPER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "steering_wrappers/add_vec_steering_wrapper.py")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "steered_generations")
DEFAULT_HF_DATASET = "obalcells/advbench"
SYSTEM_PROMPT = None

def load_instructions_from_hf_dataset(
    hf_dataset: str,
    n_test_datapoints: int
) -> List[str]:

    if hf_dataset == "advbench":
        dataset = load_dataset('obalcells/advbench', split="train")
        instructions = [dataset[i]["goal"] for i in range(len(dataset))]
        instructions = instructions[:n_test_datapoints]
      
    elif hf_dataset == "alpaca":
        dataset = load_dataset('tatsu-lab/alpaca', split='train')
        # some prompts provide extra context (`input`) related to the instruction
        # we discard all such entries
        instructions = [
            dataset[i]["instruction"]
            for i in range(len(dataset)) if len(dataset[i]["input"]) == 0
        ]
        instructions = instructions[:n_test_datapoints]
    
    else:
        raise NotImplementedError("Only advbench and alpaca datasets are supported for now")

    return instructions 

def generate(
    self,
    model,
    tokenizer,
    prompts: List[str],
    batch_size=64,
    max_new_tokens=256
) -> List[str]:
    generations = []

    for i in tqdm.tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        batch = tokenizer(batch_prompts, padding=True, return_tensors="pt")
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].half().cuda()
        num_input_tokens = input_ids.shape[1]

        output_tokens = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True
        )

        batch_generations = tokenizer.batch_decode(
            output_tokens[:, num_input_tokens:],
            skip_special_tokens=True
        )
        generations.extend(batch_generations)

        del input_ids, attention_mask, output_tokens
        gc.collect()

    return generations

def run_local_steered_text_generation(steering_settings: SteeringSettings):
    model = LlamaWrapper(
        SYSTEM_PROMPT,
        size=steering_settings.model_size,
        use_chat=not steering_settings.use_base_model,
        add_only_after_end_str=not steering_settings.add_every_token_position,
        override_model_weights_path=steering_settings.override_model_weights_path,
    )
    model.set_save_internal_decodings(False)

    hf_dataset = steering_settings.dataset
    max_num_instructions = steering_settings.n_test_datapoints

    instructions = load_instructions_from_hf_dataset(
        hf_dataset,
        max_num_instructions
    )

    prompts = [
        format_instruction_answer_llama_chat(
            tokenizer,
            [(instruction, None)],
            system_prompt,
            no_final_eos=True
        )
        for instruction in instructions
    ]

    generations = generate(
        model,
        tokenizer,
        instructions,
    )

    results = {
        "steering_settings": steering_settings.get_settings_dict(),
        "generations": []
    }

    for i in range(len(data)):
        results["generations"].append({
            "instruction": instructions[i],
            "generation": generations[i]
        })

    return results 


if __name__ == "__main__":
    """
    python3 -m jailbreak_steering.test_steering.prompting_with_steering.py \
        --vectors_dir <path_to_vectors_dir> \
        # TODO: Fix this
        # --steering_wrapper_path <path_to_model_wrapper_class> \
        # parser.add_argument("--layers", nargs="+", type=int, required=True)
        # parser.add_argument("--multipliers", nargs="+", type=float, required=True)
        # parser.add_argument("--do_projection", action="store_true", default=False)
        --system_prompt <system_prompt> \
        --hf_dataset <hf_dataset_name> \
        --max_new_tokens <number_of_generated_tokens_per_prompt> \
        --run_in_modal <True/False> \
        --output_dir <path_to_results_dir>
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--multipliers", nargs="+", type=float, required=True)
    parser.add_argument("--vectors_dir", type=str, default=DEFAULT_VECTORS_DIR)
    parser.add_argument("--hf_dataset", type=str, choices=["advbench"])
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--run_locally', action='store_true', default=True)
    parser.add_argument("--do_projection", action="store_true", default=False)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--system_prompt", type=str, default=SYSTEM_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--n_test_datapoints", type=int, default=100)

    args = parser.parse_args()

    assert args.hf_dataset == "obalcells/advbench", "Only advbench dataset is supported for now"
    assert args.run_locally == True, "Option to run in Modal not implemented yet"

    # attack_lib = dynamic_import(steering_wrapper_path)

    steering_settings = SteeringSettings()

    steering_settings.layers = args.layers
    steering_settings.multipliers = args.multipliers
    steering_settings.do_projection = args.do_projection
    steering_settings.normalize = args.normalize
    steering_settings.system_prompt = args.system_prompt
    steering_settings.dataset = args.hf_dataset
    steering_settings.max_new_tokens = args.max_new_tokens
    steering_settings.n_test_datapoints = args.n_test_datapoints
    steering_settings.add_every_token_position = args.add_every_token_position
    steering_settings.vectors_dir = args.vector_dir
    
    results = run_local_steered_text_generation(steering_settings)

    results_suffix = steering_settings.make_result_save_suffix()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    results_path = os.path.join(
        args.output_dir,
        f"results_{results_suffix}.json"
    )

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
