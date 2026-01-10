#!/bin/bash

# Use specific GPUs
export CUDA_VISIBLE_DEVICES=0,1

# Paths (replace with your own environment paths)
INPUT_FILE=/path/to/your/data/all_dpo_train.json
MODEL=/path/to/your/models/Llama-3-8B-Instruct

# Model and dataset configuration
MODEL_NAME=Llama-3-8B-Instruct
DATASET=nq        # Options: nq, tqa, hotpotqa
RETRIEVER=contriever   # Options: contriever, dpr

# Print configuration
echo "Running QA script with fixed parameters:"
echo "Model: $MODEL"
echo "Input file: $INPUT_FILE"

# Run the training script
python /path/to/your/code/train_dpo.py \
    --input "$INPUT_FILE" \
    --model "$MODEL" \
    --model_name "$MODEL_NAME" \
    --dataset "$DATASET" \
    --retriever "$RETRIEVER"