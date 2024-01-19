### Suffix generations

```bash
python3 -m jailbreak_steering.suffix_gen.run_suffix_gen \
    --dataset_path ./datasets/unprocessed/custom/harmful_instructions.csv \
    --results ./jailbreak_steering/suffix_gen/runs/qwen_0/results \
    --logs_dir ./jailbreak_steering/suffix_gen/runs/qwen_0/logs \
    --config_path ./jailbreak_steering/suffix_gen/configs/custom_suffix_gen_config.json \
    --start_idx 0 \
    --end_idx 8
```

```bash
nohup python3 -m jailbreak_steering.suffix_gen.run_suffix_gen \
    --dataset_path ./datasets/unprocessed/custom/harmful_instructions.csv \
    --results ./jailbreak_steering/suffix_gen/runs/qwen_0/results \
    --logs_dir ./jailbreak_steering/suffix_gen/runs/qwen_0/logs \
    --config_path ./jailbreak_steering/suffix_gen/configs/custom_suffix_gen_config.json \
    --start_idx 0 \
    --end_idx 8 > out.txt 2>&1 &
```

(venv) root@C.8351845:~/jailbreak_steering$ nohup python3 -m jailbreak_steering.suffix_gen.run_suffix_gen \
>     --dataset_path ./datasets/unprocessed/custom/harmful_instructions.csv \
>     --results ./jailbreak_steering/suffix_gen/runs/qwen_0/results \
>     --logs_dir ./jailbreak_steering/suffix_gen/runs/qwen_0/logs \
>     --config_path ./jailbreak_steering/suffix_gen/configs/custom_suffix_gen_config.json \
>     --start_idx 0 \
>     --end_idx 8 > out.txt 2>&1 &
[1] 3735