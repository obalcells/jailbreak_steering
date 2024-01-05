### Suffix generations

For `start_idx in [0, 20, ..., 400]`, `end_idx = start_idx + 20`, and `run_name = f"{start_idx}_{end_idx}"`:

```bash
python3 -m jailbreak_steering.suffix_gen.run_suffix_gen \
    --dataset_path ./datasets/unprocessed/advbench/harmful_behaviors_train.csv \
    --results ./jailbreak_steering/suffix_gen/runs/{run_name}/results \
    --logs_dir ./jailbreak_steering/suffix_gen/runs/{run_name}/logs \
    --config_path ./jailbreak_steering/suffix_gen/configs/suffix_gen_config.json \
    --start_idx {start_idx} \
    --end_idx {end_idx}
```

### Aggregate suffix generations

```bash
python3 -m jailbreak_steering.suffix_gen.runs.aggregate_results
```

### Process suffix generations

```bash
python3 -m jailbreak_steering.suffix_gen.process_suffix_gen \
    --suffix_gen_results_path ./jailbreak_steering/suffix_gen/runs/aggregated_results/successful_results.json \
    --output_path ./datasets/processed/advbench/advbench_suffix.json \
    --suffix_induces_behavior
```

### Vector generation

```bash
python3 -m jailbreak_steering.vector_gen.run_vector_gen \
    --dataset_path ./datasets/processed/advbench/advbench_suffix.json \
    --vectors_dir ./results/advbench_suffix_sys/vectors \
    --data_type instruction \
    --use_default_system_prompt
```

### Steered completion

Adding the vector extracted from suffix pairs:

```bash
python3 -m jailbreak_steering.steered_completion.run_steered_completion \
    --dataset_path ./datasets/unprocessed/advbench/harmful_behaviors_eval.csv \
    --results_path ./results/advbench_suffix_sys/steered_completion/layer_19.json \
    --vectors_dir ./results/advbench_suffix_sys/vectors \
    --config_path ./jailbreak_steering/steered_completion/configs/add_layer_19.json \
    --use_default_system_prompt \
    --run_locally \
    --max_new_tokens 100
```

Baseline (not adding any vectors):

```bash
python3 -m jailbreak_steering.steered_completion.run_steered_completion \
    --dataset_path ./datasets/unprocessed/advbench/harmful_behaviors_eval.csv \
    --results_path ./results/baseline_sys/steered_completion/baseline.json \
    --config_path ./jailbreak_steering/steered_completion/configs/baseline_config.json \
    --use_default_system_prompt \
    --run_locally \
    --max_new_tokens 100
```

### Evaluate completions

```bash
python3 -m jailbreak_steering.eval.compute_attack_success_rate \
    --method prefix_search \
    --generations_path ./results/advbench_suffix_sys/steered_completion/layer_19.json \
    --results_path ./results/advbench_suffix_sys/eval/layer_19.json \
    --verbose
```

```bash
python3 -m jailbreak_steering.eval.compute_attack_success_rate \
    --method prefix_search \
    --generations_path ./results/baseline_sys/steered_completion/baseline.json \
    --results_path ./results/baseline_sys/eval/baseline.json \
    --verbose
```