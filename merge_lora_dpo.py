import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# Optional: restrict visible GPUs via environment variable
# Example: os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def merge_lora(base_model_path, lora_model_path, merged_model_path):
    """
    Merge a LoRA adapter into the base model and save the merged model.
    """
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base_model, lora_model_path)
    model = model.merge_and_unload()

    os.makedirs(merged_model_path, exist_ok=True)
    model.save_pretrained(merged_model_path, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(merged_model_path)


if __name__ == "__main__":
    # Example paths (replace with your own)
    experiment_dir = "path/to/lora/checkpoint"

    base_model_path = "path/to/base_model"
    lora_model_path = experiment_dir
    merged_model_path = os.path.join(experiment_dir, "merged")

    merge_lora(base_model_path, lora_model_path, merged_model_path)
