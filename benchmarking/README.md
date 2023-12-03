
## Optimization:

1) Batch size
    - original: 512
2) Length of the control
    - original: 20 
3) Topk
    - original: 256

Original: 500 steps
Original ASR (of suffix_gen with this config): 56% (multiple 86%)

## Excel

config idx	batch size	topk	control length	number of steps
0	512	256	20	256
1	512	256	30	256
2	512	64	20	256
3	512	64	30	256
4	256	128	20	512
5	256	128	30	512
6	256	32	20	512
7	256	32	30	512
8	128	64	20	1024
9	128	64	30	1024
10	128	16	20	1024
11	128	16	30	1024

## Commands

CUDA_VISIBLE_DEVICES=0 nohup python3 benchmarking/run_benchmarks.py --configs 0 1 > output0.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python3 benchmarking/run_benchmarks.py --configs 2 3 > output1.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python3 benchmarking/run_benchmarks.py --configs 4 5 > output2.out 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python3 benchmarking/run_benchmarks.py --configs 6 7 > output3.out 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python3 benchmarking/run_benchmarks.py --configs 8 9 > output4.out 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python3 benchmarking/run_benchmarks.py --configs 10 11 > output5.out 2>&1 &

## Commands (after first run)

CUDA_VISIBLE_DEVICES=0 nohup python3 benchmarking/run_benchmarks.py --configs 1 > output0_again.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python3 benchmarking/run_benchmarks.py --configs 3 > output1_again.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python3 benchmarking/run_benchmarks.py --configs 5 > output2_again.out 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python3 benchmarking/run_benchmarks.py --configs 7 > output3_again.out 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup python3 benchmarking/run_benchmarks.py --configs 11 > output3_again.out 2>&1 &