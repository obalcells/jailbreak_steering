# Jailbreak Steering

## Setup

```bash
chmod +x setup.sh & source setup.sh
```

This command will
1. Prompt you for your HuggingFace credentials and cache them.
2. Create and activate a venv. 
3. Install the required packages.

## Usage

**Note**: all commands should be run from the **project's root directory**.

### Suffix generation

In this step, we take a dataset of (`goal`, `target`) pairs and generate a suffix for each `goal` that induces the `target` generation.

```bash
python3 -m jailbreak_steering.suffix_gen.run_suffix_gen \
    --dataset_path <path_to_dataset> \
    --results_dir <path_to_results_dir> \
    --logs_dir <path_to_logs_dir> \
    --config_path <path_to_config>
```

**Arguments**

- `--dataset_path`: path to the dataset CSV file containing goal and target columns.
- `--results_dir`: directory where the generated suffixes will be stored.
- `--logs_dir`: directory for saving logs.
- `--config_path`: path to the configuration file.
- `--start_idx` and `--end_idx`: range of dataset entries to process (end index is non-inclusive) - the suffix gen is run only for the prompts in `dataset[start_idx:end_idx]` 
- `--seed`: random seed for suffix generation.
- `--retry_failed`: boolean flag to retry generating suffixes that failed previously.

### Process suffix generation results

Before generating steering vectors, we'll process the results from suffix generation, creating a dataset with the proper formatting.

```bash
python3 -m jailbreak_steering.suffix_gen.process_suffix_gen \
    --suffix_gen_results_path <path_to_suffix_gen_results> \
    --output_dir <path_to_output_dir>
```

**Arguments**
- `--suffix_gen_results_path`: path to the file containing successful suffix generation results.
- `--output_path`: file path where the processed results will be saved.
- `--pair_with_random_suffix`: boolean flag to pair each instruction with a randomly generated suffix.
- `--suffix_induces_behavior`: boolean flag label the combination of instruction and suffix as `instruction_inducing_behavior`.
- `--suffix_does_not_induce_behavior`: boolean flag to label the combination of instruction and suffix as `instruction_not_inducing_behavior`.

### Vector generation

To generate steering vectors, run:
```bash
python3 -m jailbreak_steering.vector_gen.run_vector_gen \
    --dataset_path <path_to_data> \
    --vectors_dir <path_to_vectors_dir> \
    --data_type <data_type>
```

**Arguments**
- `--dataset_path`: Path to the dataset file.
- `--vectors_dir`: Directory to save the generated vectors.
- `--data_type`: Type of data to process, either `instruction` or `instruction_answer`.
  - `instruction`: each dataset example consists of `instruction_inducing_behavior`,`instruction_not_inducing_behavior`.
  - `instruction_answer`: each dataset example consists of `instruction`, `answer_matching_behavior`, `answer_not_matching_behavior`.
- `--use_default_system_prompt`: Boolean flag to use the default system prompt. If not set, no system prompt is used.
- `--save_activations`: Boolean flag to save the activation tensors.
- `--activations_dir`: Directory to save the activation tensors.

### Generating steered completions
```bash
python3 -m jailbreak_steering.test_steering.run_steered_completion.py \
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

Given a set of instructions and corresponding generations, it returns the percentage of generations that contain harmful behavior. So if one instruction is "Tell me how to make a bomb.", a "succesful" generation is one that follows the instruction and contains useful content for making a bomb.

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
    "generations": [
        {
            "instruction": "instruction text",
            "generation": "generation text"
        },
        // more generations...
    ],
    // other content...
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
Results saved to /path_to_repo/jailbreak_steering/jailbreak_steering/attack_success_rate/results/eval_ASR_gpt-3.5-turbo-1106_batchsize=10_maxgensize=200_date=20231206-12:53.json
```

## Tests

To run tests, run:
```bash
pytest
```
