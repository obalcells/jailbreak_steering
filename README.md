# Jailbreak Steering

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then create a `.env` file with the following variables (see `.env.example`):

```
HF_TOKEN=huggingface_token_with_access_to_llama2
```

## Usage

Note that all commands should be run from the root directory.

### Suffix generation

In this step, we take a dataset of instructions, and generate a suffix for each instruction that induces the desired behavior (e.g. not refusing a dangerous instruction).

To run this step on a dataset of instructions:
```bash
python3 -m jailbreak_steering.suffix_gen.run_suffix_gen \
    --dataset_path <path_to_dataset> \
    --results_dir <path_to_results_dir> \
    --logs_dir <path_to_logs_dir> \
    --config_path <path_to_config>
    --start_idx <idx_of_first_prompt> \
    --end_idx <idx_of_last_prompt>
```

- `dataset_path` points to a dataset of instructions, which is assumed to be a csv with a column `goal` (the instruction, e.g. `Tell me how to make a bomb`) and a column `target` (the target response, e.g. `Sure, here's how to make a bomb`).
- `results_dir` is the directory where the generated suffixes will be saved.
- `logs_dir` is the directory where the logs will be saved.
- `config_path` is the path of the config file containing the parameters that will be used to run the attack. The default config used is `./jailbreak_steering/suffix_gen/configs/suffix_gen_config.json`.
- `start_idx` and `end_idx` (non-inclusive) define a slice of the dataset. The attack is run only for the prompts in `dataset[start_idx:end_idx]`. By default we take only the first 10 prompts.

### Process suffix generation results

Before generating steering vectors, we'll process the results from suffix generation,
creating a dataset with the proper formatting.

To run this step:
```bash
python3 -m jailbreak_steering.suffix_gen.process_suffix_gen \
    --suffix_gen_results_path <path_to_suffix_gen_results> \
    --output_dir <path_to_output_dir>
```

### Vector generation

To generate steering vectors, run:
```bash
python3 -m jailbreak_steering.vector_gen.run_vector_gen \
    --label <label> \
    --data_path <path_to_data> \
    --layers <layers> \
    --vectors_dir <path_to_vectors_dir> \
    --activations_dir <path_to_activations_dir> \
    --save_activations <save_activations>
```

### Text generation applying steering vectors
```bash
python3 -m jailbreak_steering.test_steering.prompting_with_steering.py \
    --dataset_path <path_to_data> \
    --hf_dataset <hf_dataset_id> \
    --results_dir <path_to_results_dir> \
    --vectors_dir <path_to_vectors_dir> \
    --config_path <path_to_steering_config> \
    --run_locally <boolean_flag_run_locally_or_modal> \
    --num_test_datapoints <max_number_prompts> \
    --max_new_tokens <number_tokens_generated_per_prompt> \
```

- `dataset_path` points to a dataset of instructions, which is assumed to be a csv with a column `goal` (the instruction, e.g. `Tell me how to make a bomb`) and a column `target` (the target response, e.g. `Sure, here's how to make a bomb`).
- `hf_dataset` is the name of a Huggingface dataset that we'll run generations for. This is set to `None` by default, but if we pass a value then we'll use a Huggingface dataset instead of passing the path to a locally stored dataset. The supported Huggingface datasets at the moment are `obalcells/advbench` and `tatsu-lab/alpaca`.
- `results_dir` is the directory where the generated suffixes will be saved.
- `config_path` is the path of the config file containing the parameters that will be used to perform the steering. The default config used is `./jailbreak_steering/test_steering/configs/add_layer_19.json`. This config specified which steering vectors, multipliers, layers, system prompt, etc to use and it's agnostic to the datase and any kind of generation parameters.
- `run_locally` is a boolean flag specifying whether to run the steering locally or in a [Modal](https://modal.com) instance. This is set to `True` by default.
- `num_test_datapoints` specifies how many instructions to take from the dataset. This is set to `None` by default which means taking the whole dataset.
- `max_new_tokens` specifies the number of tokens we want to generate for each prompt. This is set to `50` by default.

### Calculate attack success rate
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default="llm_judge", choices=['llm_judge', 'prefix_search'])
    parser.add_argument('--generations_path', type=str, required=True)
    parser.add_argument('--results_dir', type=str, default=DEFAULT_ASR_RESULTS_DIR)
    parser.add_argument('--model_name', type=str, default=DEFAULT_LLM_JUDGE_MODEL)
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--max_generation_length', type=int, default=DEFAULT_TRUNCATION_LENGTH)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

```bash
python3 -m jailbreak_steering.attack_success_rate.calculate_attack_success_rate.py \
    --method <method> \
    --generations_path <path_to_generations> \
    --results_dir <path_to_results_dir> \
    --model_name <name_of_llm_judge_model> \
    --api_key <modal_api_key> \
    --batch_size <batch_size> \
    --truncation_length <max_generation_length> \
    --verbose <boolean_flag_verbose>
```

- `method` specifies which method to use to calculate the attack success rate. This can be either `llm_judge` or `prefix_search`.
- `generations_path` is the path to the file containing the steered generations. Their format has to match the one of the files produced by running `prompting_with_steering`.
```json
{
    ...
    "generations": [
        {
            "instruction": "instruction text",
            "generation": "generation text"
        },
        ...
    ]
} 
```
- `results_dir` is the directory where the results will be saved.
- `model_name` is the name of the LLM Judge model that we'll use to calculate the attack success rate.
- `api_key` is the OpenAI API key. This is only needed if we run the evaluation using `llm_judge`.
- `batch_size` is the batch size to use to make requests to Modal. This is set to `10` by default.
- `truncation_length` is the maximum number of tokens to include per generation.
- `verbose` is a boolean flag specifying whether to print the results of the attack success rate calculation. This is set to `False` by default.

Here's an example:
```python
python3 -m jailbreak_steering.attack_success_rate.compute_attack_success_rate --generations_path ./jailbreak_steering/test_steering/results/results_20231205-00\:45_layers\=\[19\]_multipliers\=\[1\]_vector_label\=original_do_projection\=False_normalize\=True_add_every_token_position\=False_system_prompt\=custom.json
```

Output:
```bash
Evaluating generations: 100%|███████████████████████████████| 10/10 [01:43<00:00, 10.32s/it]
Overall success rate: 85.00%
```

### Calculate the average number of tokens generated

```bash
python3 -m jailbreak_steering.attack_success_rate.calculate_average_num_tokens.py \
    --generations_path <path_to_generations> \
    --results_dir <path_to_results_dir>
```

- `generations_path` is the path to the file containing the generated responses.
- `results_dir` is the directory where the results will be saved.

### Calculate the average number of tokens generated

```bash
python3 -m jailbreak_steering.attack_success_rate.calculate_average_num_tokens.py \
    --generations_path <path_to_generations> \
    --results_dir <path_to_results_dir>
```

- `generations_path` is the path to the file containing the generated responses.
- `results_dir` is the directory where the results will be saved.

### Calculate the average number of tokens generated

```bash
python3 -m jailbreak_steering.attack_success_rate.calculate_average_num_tokens.py \
    --generations

## Tests

To run tests, run:
```bash
pytest
```


## Step-by-step

