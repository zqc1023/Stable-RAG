#!/bin/bash

# Select visible GPUs
export CUDA_VISIBLE_DEVICES=0

# Example input and model paths (replace with your own)
INPUT_FILE=path/to/dataset/test.json
MODEL=path/to/merged_model

echo "Running QA script with fixed parameters:"
echo "Model: $MODEL"
echo "Input file: $INPUT_FILE"

python Stable-RAG/inference.py \
  --input "$INPUT_FILE" \
  --model "$MODEL"