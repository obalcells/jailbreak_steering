### Suffix generations

```bash
python3 -m jailbreak_steering.suffix_gen.run_suffix_gen \
    --dataset_path ./datasets/unprocessed/custom/harmful_instructions.csv \
    --results ./jailbreak_steering/suffix_gen/runs/qwen/results \
    --logs_dir ./jailbreak_steering/suffix_gen/runs/qwen/logs \
    --config_path ./jailbreak_steering/suffix_gen/configs/custom_suffix_gen_config.json \
    --start_idx 0 \
    --end_idx 4
```