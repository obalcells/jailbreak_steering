
## Original attack parameters

```json
{
    "system_prompt": "default",
    "max_steps": 500,
    "batch_size": 512,
    "topk": 256,
    "control_len": 20,
}
```

The ASR for the individual prompt attack is 56%

The ASR for the multiprompt attack is 86%

## Configs

| config idx | batch size | topk | control length | number of steps |
|-|-|-|-|-|
| 0 | 512 | 256 | 20 | 256 |
| 1 | 512 | 256 | 30 | 256 |
| 2 | 512 | 64 | 20 | 256 |
| 3 | 512 | 64 | 30 | 256 |
| 4 | 256 | 128 | 20 | 512 |
| 5 | 256 | 128 | 30 | 512 |
| 6 | 256 | 32 | 20 | 512 |
| 7 | 256 | 32 | 30 | 512 |
| 8 | 128 | 64 | 20 | 1024 |
| 9 | 128 | 64 | 30 | 1024 |
| 10 | 128 | 16 | 20 | 1024 |
| 11 | 128 | 16 | 30 | 1024 |


## Commands (first run)

CUDA_VISIBLE_DEVICES=0 nohup python3 benchmarking/run_benchmarks.py --configs 0 1 > output0.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python3 benchmarking/run_benchmarks.py --configs 2 3 > output1.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python3 benchmarking/run_benchmarks.py --configs 4 5 > output2.out 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python3 benchmarking/run_benchmarks.py --configs 6 7 > output3.out 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python3 benchmarking/run_benchmarks.py --configs 8 9 > output4.out 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python3 benchmarking/run_benchmarks.py --configs 10 11 > output5.out 2>&1 &

The tests of the even-numbered configs worked fine, but they failed afterwards. The reason for this (I think) was because I changed a few things in the repo without knowing that this would affect the ongoing tests. This is because we run each attack as a bash subcommand instead of calling the function directly.

## Commands (after first run)

CUDA_VISIBLE_DEVICES=0 nohup python3 benchmarking/run_benchmarks.py --configs 1 > output0_again.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python3 benchmarking/run_benchmarks.py --configs 3 > output1_again.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python3 benchmarking/run_benchmarks.py --configs 5 > output2_again.out 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python3 benchmarking/run_benchmarks.py --configs 7 > output3_again.out 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup python3 benchmarking/run_benchmarks.py --configs 11 > output3_again.out 2>&1 &

I then ran again the attacks for the missing configs, except for config 9 because that one hadn't failed (the command for that was still processing config 8).

After running them I noticed that an error had occurred for both config 1 and 11.

Weirdly, for config 1 all instructions except the last one had been processed correctly. The error is a CUDA memory error. Logs [here](https://github.com/obalcells/jailbreak_steering/blob/master/output0_again.out).

For config 11, also a weird error occurred (see logs [here](https://github.com/obalcells/jailbreak_steering/blob/master/output5.out)). I think this might be an actual bug in the implementation.

## Commands (third run)

CUDA_VISIBLE_DEVICES=0 nohup python3 benchmarking/run_benchmarks.py --configs 1 > output0.out 2>&1 &

(I will rename output0.out to output_config_1.out so that the previous .out doesn't get replaced)

CUDA_VISIBLE_DEVICES=1 nohup python3 benchmarking/run_benchmarks.py --configs 11 > output_config_11.out 2>&1 &
