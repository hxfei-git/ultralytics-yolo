#!/bin/bash

# Training task scheduler
# Monitors the 'train_queue' directory and automatically starts training
# when a .yaml model config file is placed inside.
#
# Usage:
#   chmod +x scheduler_train.sh
#   ./scheduler_train.sh
#
# To submit a new job:
#   Copy your model config (e.g., myexp.yaml) into ./train_queue/
#   The experiment name will be derived from the filename (without .yaml)

QUEUE_DIR="train_queue"

mkdir -p "$QUEUE_DIR"

echo "Training scheduler started. Monitoring directory: $(realpath "$QUEUE_DIR")"
echo "To add a task: copy a .yaml file into $QUEUE_DIR"
echo

while true; do
    # Check if any .yaml files exist
    yaml_files=("$QUEUE_DIR"/*.yaml)
    if [ -e "${yaml_files[0]}" ]; then
        yaml_path="${yaml_files[0]}"
        filename=$(basename "$yaml_path")
        name="${filename%.yaml}"  # Remove .yaml suffix

        echo "Starting training task: $name"
        echo "Model config: $yaml_path"

        if python train.py --yaml "$yaml_path" --name "$name"; then
            echo "Training succeeded. Removing task file: $filename"
            rm -f "$yaml_path"
        else
            echo "Training failed. Keeping task file for inspection: $filename"
        fi

        echo
    else
        echo "No tasks in queue. Exiting."
        exit 0
    fi
done