#!/bin/bash

# Define the base paths for train and dev data directories
TRAIN_DATA_PATH="/home/dan/TemporalUnderstandingInLLMs/data/preprocessed/train/"
DEV_DATA_PATH="/home/dan/TemporalUnderstandingInLLMs/data/preprocessed/dev/packed/"

# Define the path to your Python script
PYTHON_SCRIPT_PATH="/home/dan/TemporalUnderstandingInLLMs/instruction_tuning/IT_w_command_line_args.py"

# Output directory for training logs
OUTPUT_DIR="/home/dan/TemporalUnderstandingInLLMs/model_outputs/"

# Model contexts
CONTEXTS=("no_context" "random_context" "rel_context" "WD_context")

# Models
MODELS=("GEMMA" "Llama")

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Loop through models and contexts to run the script with all combinations
for MODEL in "${MODELS[@]}"; do
    for CONTEXT in "${CONTEXTS[@]}"; do
        TRAIN_FILE="train_${MODEL}_${CONTEXT}_packed.jsonl"
        EVAL_FILE="dev_${MODEL}_${CONTEXT}_packed.jsonl"
        
        # Output file path
        OUTPUT_FILE="${OUTPUT_DIR}${MODEL}_${CONTEXT}_training.out"

        # Debug print
        echo "Output to: $OUTPUT_FILE"

        # Construct the full paths to the train and eval files
        FULL_TRAIN_PATH="${TRAIN_DATA_PATH}${TRAIN_FILE}"
        FULL_EVAL_PATH="${DEV_DATA_PATH}${EVAL_FILE}"

        # Check if both train and eval data files exist before running the script
        if [[ -f "$FULL_TRAIN_PATH" && -f "$FULL_EVAL_PATH" ]]; then
            # Call the Python script with command line arguments and save output to a file
            python "$PYTHON_SCRIPT_PATH" --llama_or_gemma "$MODEL" \
                                         --train_data_file "$FULL_TRAIN_PATH" \
                                         --eval_data_file "$FULL_EVAL_PATH" \
                                         --model_context "$CONTEXT" > "$OUTPUT_FILE" 2>&1
            echo "$MODEL with $CONTEXT context has finished processing."
        else
            echo "Training or evaluation file not found for $MODEL with $CONTEXT."
        fi
    done
done
