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
--output_path ./datasets/processed/advbench/harmful_behaviors_train.json \
--suffix_does_not_induce_behavior
```

### Vector generation

```bash
```
