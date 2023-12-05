import argparse
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import gc
import math
import time
from typing import Dict, List, Optional
import numpy as np
import torch
import tqdm
from datasets import load_dataset

from jailbreak_steering.suffix_gen.run_suffix_gen import DEFAULT_DATASET_PATH, load_config, load_dataset
from jailbreak_steering.utils.llama_wrapper import LlamaWrapper
from jailbreak_steering.utils.tokenize_llama_chat import DEFAULT_SYSTEM_PROMPT, format_instruction_answer_llama_chat
from jailbreak_steering.vector_gen.run_vector_gen import DEFAULT_LABEL, DEFAULT_VECTORS_DIR, make_tensor_save_suffix

DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DEFAULT_STEERING_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "add_layer_19.json")
DEFAULT_MAX_NEW_TOKENS = 50 

def make_config_suffix(config: Dict) -> str:
    if config["system_prompt"] is None or len(config["system_prompt"]) == 0:
        system_prompt_short = "none"
    if config["system_prompt"] == DEFAULT_SYSTEM_PROMPT:
        system_prompt_short = "default"
    else:
        system_prompt_short = "custom"

    elements = {
        "layers": config["layers"],
        "multipliers": config["multipliers"],
        "vector_label": config["vector_label"],
        "do_projection": config["do_projection"],
        "normalize": config["normalize"],
        "add_every_token_position": config["add_every_token_position"],
        "system_prompt": system_prompt_short,
    }

    timestamp = time.strftime("%Y%m%d-%H:%M")

    suffix = timestamp + "_" + "_".join([f"{k}={v}" for k, v in elements.items() if v is not None])

    return suffix

def get_steering_vector(label, layer, vectors_dir):
    return torch.load(
        os.path.join(
            vectors_dir,
            f"vec_layer_{make_tensor_save_suffix(layer, label)}.pt",
        )
    )

def load_instructions_from_hf_dataset(
    hf_dataset: str,
    n_test_datapoints: int
) -> List[str]:

    if hf_dataset == "obalcells/advbench":
        dataset = load_dataset('obalcells/advbench', split="train")
        instructions = [dataset[i]["goal"] for i in range(len(dataset))]
        instructions = instructions[:n_test_datapoints]
      
    elif hf_dataset == "tatsu-lab/alpaca":
        dataset = load_dataset('tatsu-lab/alpaca', split='train')
        # The prompts providing extra context (`input`) are discarded
        instructions = [
            dataset[i]["instruction"]
            for i in range(len(dataset)) if len(dataset[i]["input"]) == 0
        ]
        instructions = instructions[:n_test_datapoints]
    
    else:
        raise NotImplementedError("Only advbench and alpaca datasets are supported for now")

    return instructions 

def generate(
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

def run_local_steered_text_generation(
    instructions: List[str],
    max_new_tokens: int,
    steering_config: Dict,
    steering_vectors: List[torch.Tensor]
) -> List[str]:

    model = LlamaWrapper(
        steering_config["system_prompt"],
        add_only_after_end_str=not steering_config["add_every_token_position"],
    )
    model.set_save_internal_decodings(False)
    model.reset_all()

    for i, layer in enumerate(steering_config["layers"]):
        vector = steering_vectors[i].to(model.device)
        multiplier = steering_config["multipliers"][i]

        model.set_add_activations(
            layer,
            multiplier * vector,
            do_projection=steering_config["do_projection"],
            normalize=steering_config["normalize"]
        )

    prompts = [
        format_instruction_answer_llama_chat(
            model.tokenizer,
            [(instruction, None)],
            steering_config["system_prompt"],
            no_final_eos=True
        )
        for instruction in instructions
    ]

    return generate(
        model,
        model.tokenizer,
        prompts,
        max_new_tokens=max_new_tokens
    )

def run_modal_steered_text_generation(instructions: List[str], max_new_tokens: int, steering_config: Dict, steering_vectors: List[torch.Tensor]) -> Dict:
    raise NotImplementedError("Modal not implemented yet")

def prompting_with_steering(
    local_data_path: str,
    vectors_dir: str,
    config_path: str,
    results_dir: str,
    hf_dataset: str = None,
    num_test_datapoints: int = None, # None means use all datapoints
    max_new_tokens: int = 256,
    run_locally: bool = True,
):
    steering_config = load_config(config_path)

    steering_vector_label = steering_config["vector_label"]

    # Load the steering vectors locally
    steering_vectors = []
    for layer in steering_config["layers"]:
        vec = get_steering_vector(steering_vector_label, layer, vectors_dir)
        vec = vec.cpu().type(torch.float16)
        steering_vectors.append(vec)

    # Load the list of instructions
    if hf_dataset is not None:
        instructions = load_instructions_from_hf_dataset(hf_dataset, num_test_datapoints)
    else:
        instructions, _ = load_dataset(local_data_path, n=num_test_datapoints)

    start_time = time.time()

    generations = run_local_steered_text_generation(
        instructions, max_new_tokens, steering_config, steering_vectors
    )

    runtime = round(time.time() - start_time)
    
    results = {}

    results["config"] = steering_config
    results["vectors_dir"] = vectors_dir
    results["local_data_path"] = local_data_path
    results["num_test_datapoints"] = num_test_datapoints
    results["max_new_tokens"] = max_new_tokens
    results["runtime"] = runtime

    results["generations"] = []
    for instruction, generation in zip(instructions, generations):
        results["generations"].append({ "instruction": instruction, "generation": generation })

    results_suffix = make_config_suffix(steering_config)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results_path = os.path.join(
        results_dir,
        f"results_{results_suffix}.json"
    )

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Finished running {num_test_datapoints} prompts in {runtime} seconds")
    print(f"Saved results to: {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--hf_dataset", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--vectors_dir", type=str, default=DEFAULT_VECTORS_DIR)
    parser.add_argument("--config_path", type=str, default=DEFAULT_STEERING_CONFIG_PATH)
    parser.add_argument('--run_locally', action='store_true', default=True)
    parser.add_argument("--num_test_datapoints", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=50)

    args = parser.parse_args()

    assert args.run_locally == True, "Option to run in Modal not implemented yet"

    prompting_with_steering(
        args.dataset_path,
        args.vectors_dir,
        args.config_path,
        args.results_dir,
        hf_dataset=args.hf_dataset,
        num_test_datapoints=args.num_test_datapoints,
        max_new_tokens=args.max_new_tokens,
        run_locally=args.run_locally,
    )
