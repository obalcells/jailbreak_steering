import sys

def main():
    gpu_number = sys.argv[1]
    start_idx = sys.argv[2]
    end_idx = sys.argv[3]
    run_name = f"{start_idx}_{end_idx}"
    command = f"CUDA_VISIBLE_DEVICES={gpu_number} nohup python3 -m jailbreak_steering.suffix_gen.run_suffix_gen --results ./jailbreak_steering/suffix_gen/runs/{run_name}/results --logs_dir ./jailbreak_steering/suffix_gen/runs/{run_name}/logs --start_idx {start_idx} --end_idx {end_idx} > ./jailbreak_steering/suffix_gen/runs/{run_name}_output.out 2>&1 &"
    print(command)

if __name__ == "__main__":
    main()