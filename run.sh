#!/bin/bash

# Check if exactly four arguments are given ($# is the number of arguments)
if [ "$#" -ne 4 ]; then
    echo "Usage: ./run.sh /path/to/Taiwan-LLaMa-folder /path/to/peft-folder /path/to/input.josn /path/to/output.json"
    exit 1
fi

BASE_MODEL="$1"
PEFT_CONFIG="$2"
INPUT_PATH="$3"
OUTPUT_PATH="$4"

# Run the Python program with the provided arguments
python ./src/inference.py --base_model "$BASE_MODEL" --peft_config "$PEFT_CONFIG" --input_file "$INPUT_PATH" --output_file "$OUTPUT_PATH"

