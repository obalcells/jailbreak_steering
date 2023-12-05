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
```

- `dataset_path` is points to a dataset of instructions, which is assumed to be a csv with a column `goal` (the instruction, e.g. `Tell me how to make a bomb`) and a column `target` (the target response, e.g. `Sure, here's how to make a bomb`).
- `output_dir` is the directory where the generated suffixes will be saved.
- `logs_dir` is the directory where the logs will be saved.

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
    --layers <layers> \
    --multipliers <multipliers> \
    --label <steering_vector_label> \
    --vectors_dir <path_to_vectors_dir> \
    --hf_dataset_label <hf_dataset_label> \
    --output_dir <path_to_results_dir>
    --run_locally <true_or_false> \
    --do_projection <true_or_false> \
    --normalize <true_or_false> \
    --system_prompt <system_prompt_str> \
    --max_new_tokens <number_tokens_generated_per_prompt> \
    --n_test_datapoints <max_number_prompts>
```

## Tests

To run tests, run:
```bash
pytest
```


## Step-by-step

