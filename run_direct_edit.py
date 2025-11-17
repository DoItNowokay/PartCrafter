#!/usr/bin/env python3

import subprocess
import re
import time
import os
import shlex
import sys

# --- Configuration ---

# The command you want to run.
# It's split using shlex.split() to handle arguments correctly.
COMMAND_TO_RUN = (
    "bash scripts/train_partcrafter_edit.sh "
    "--config configs/mp8_nt512_edit.yaml "
    "--gradient_accumulation_steps 4 "
    "--output_dir output_partcrafter "
    "--tag direct_tweak_latent_editing_direct_text_mp8_nt512_chair "
    "--text_conditioning direct_text "
    "--editing direct_tweak_latent"
)

# Your memory threshold.
MEM_THRESHOLD_MIB = 95000

# How many seconds to wait before re-checking for a free GPU.
CHECK_INTERVAL_SECONDS = 60

# --- End of Configuration ---


def find_available_gpu(mem_threshold_mib):
    """
    Checks GPUs 0-7 individually using 'nvidia-smi --query-gpu'
    to find one with memory usage below the threshold.
    """
    
    # We will check GPUs 0 through 7
    for gpu_id in range(8):
        try:
            # This query asks for *only* the memory used number,
            # with no header ("noheader") and no units ("nounits").
            # The output will be a single number, e.g., "133004".
            command = [
                'nvidia-smi',
                f'--id={gpu_id}',
                '--query-gpu=memory.used',
                '--format=csv,noheader,nounits'
            ]
            
            # Run the command
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )
            
            output = result.stdout.strip()
            
            # Parse the memory usage
            try:
                mem_used = int(output)
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not parse memory for GPU {gpu_id}. Output: '{output}'. Error: {e}")
                continue # Try the next GPU

            print(f"Checking GPU {gpu_id}: Used memory {mem_used}MiB")
            
            if mem_used < mem_threshold_mib:
                print(f"  -> Found suitable GPU {gpu_id} with {mem_used}MiB used.")
                return gpu_id

        except FileNotFoundError:
            print("Error: 'nvidia-smi' command not found.", file=sys.stderr)
            print("Please ensure the NVIDIA driver is installed and 'nvidia-smi' is in your PATH.", file=sys.stderr)
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            # This will happen if a GPU ID is invalid or not responding.
            # We can just ignore it and continue to the next one.
            print(f"Warning: Could not query GPU {gpu_id}. Error: {e.stderr.strip()}", file=sys.stderr)
        except Exception as e:
            print(f"An unexpected error occurred while checking GPU {gpu_id}: {e}", file=sys.stderr)
            
    # If we looped through all GPUs and found none
    return None


def run_command_on_gpu(base_command_list, gpu_id):
    """
    Runs the specified command on the given GPU ID using subprocess.Popen
    and streams its output to stdout.
    """
    print(f"\n--- Starting command on GPU {gpu_id} ---")
    oom_detected = False
    
    # Copy the current environment variables
    my_env = os.environ.copy()
    # Set CUDA_VISIBLE_DEVICES for the subprocess
    my_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        # Use subprocess.Popen to run the command and stream output
        with subprocess.Popen(
            base_command_list,
            env=my_env,
            stdout=subprocess.PIPE,       # Capture standard output
            stderr=subprocess.STDOUT,      # Redirect standard error to standard output
            text=True,
            bufsize=1,                   # Line-buffered
            universal_newlines=True
        ) as process:
            
            if process.stdout:
                # Read output line by line in real-time
                for line in iter(process.stdout.readline, ''):
                    print(line, end='') # Print the line to our stdout
                    
                    # Check for OOM error
                    if "CUDA out of memory" in line or "OOM" in line:
                        oom_detected = True
            
            # Wait for the process to terminate and get the exit code
            exit_code = process.wait() 
        
        return oom_detected, exit_code

    except FileNotFoundError:
        print(f"Error: Command not found. Is '{base_command_list[0]}' in your PATH?", file=sys.stderr)
        return False, -1 # Return a non-zero exit code
    except Exception as e:
        print(f"Error starting subprocess: {e}", file=sys.stderr)
        return False, -1 # Return a non-zero exit code


def main():
    """
    Main loop to find a GPU and run the command.
    """
    # Parse the command string into a list
    try:
        command_list = shlex.split(COMMAND_TO_RUN)
    except Exception as e:
        print(f"Error: Could not parse COMMAND_TO_RUN: {e}", file=sys.stderr)
        print("Please check the command string for formatting errors.", file=sys.stderr)
        sys.exit(1)

    try:
        while True:
            gpu_id = find_available_gpu(MEM_THRESHOLD_MIB)
            
            if gpu_id is not None:
                # Found a GPU, run the command
                oom_detected, exit_code = run_command_on_gpu(command_list, gpu_id)
                
                if oom_detected:
                    print(f"\n*** Process on GPU {gpu_id} failed with CUDA OOM. ***")
                    print("Restarting search for an available GPU...\n")
                    time.sleep(10) # Wait 10s before retrying
                elif exit_code == 0:
                    print(f"\n*** Command completed successfully on GPU {gpu_id}. Exiting. ***")
                    break # Success! Exit the while loop.
                else:
                    print(f"\n*** Command on GPU {gpu_id} failed with exit code {exit_code} (Not OOM). ***")
                    print(f"Waiting {CHECK_INTERVAL_SECONDS}s before retrying...\n")
                    time.sleep(CHECK_INTERVAL_SECONDS)
            
            else:
                # No suitable GPU found
                print(f"No GPU found with memory usage below {MEM_THRESHOLD_MIB}MiB.")
                print(f"Waiting {CHECK_INTERVAL_SECONDS} seconds before re-checking...")
                time.sleep(CHECK_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nScript interrupted by user. Exiting.")
        sys.exit(0)

if __name__ == "__main__":
    main()