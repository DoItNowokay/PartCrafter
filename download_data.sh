#!/bin/bash

# Name of the Python script to run
python_file="download_data.py"

echo "Target Script: $python_file"
echo "Passing Arguments: $@"

# Loop until the Python file completes successfully
while true; do
    echo "Starting Python script with 2.5-minute timeout..."

    # Run the script with a 150-second timeout
    # "$@" passes all arguments (e.g., --dataset abo) to the python script
    timeout 15000s python "$python_file" "$@"
    
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "Python script completed successfully!"
        break
    elif [ $exit_code -eq 124 ]; then
        echo "Python script timed out after 150 seconds. Restarting after 40 seconds..."
    else
        echo "Python script failed with exit code $exit_code. Retrying after 40 seconds..."
    fi

    sleep 40
done