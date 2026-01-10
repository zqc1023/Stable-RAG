import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from datasets import Dataset
from peft import get_peft_model, LoraConfig
import json
from datetime import datetime
import random
import argparse
import logging
import sys


class LoggerWriter:
    """Redirect stdout/stderr to logging, flush immediately."""

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        message = message.rstrip()
        if message:
            self.logger.log(self.level, message)
            for handler in self.logger.handlers:
                handler.flush()

    def flush(self):
        for handler in self.logger.handlers:
            handler.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="Train DPO")
    parser.add_argument("--input", type=str, required=True, help="input_file")
    parser.add_argument("--model", type=str, help="model")
    parser.add_argument("--model_name", type=str, help="model_name")
    parser.add_argument("--retriever", type=str, help="retriever", default="dpr")
    parser.add_argument("--dataset", type=str, help="dataset_name")
    return parser.parse_args()


def create_path(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder {folder_path} Created")
    else:
        print(f"Folder {folder_path} Existed")


def setup_logger(output_dir):
    log_file = os.path.join(output_dir, "train.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info(f"Logger initialized. Logs will be saved to {log_file}")

    sys.stdout = LoggerWriter(logger, logging.INFO)

    return logger


def load_json(file_path, max_samples=18000):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data[:max_samples] if len(data) > max_samples else data


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_dpo():
    args = parse_args()

    set_seed(42)
    train_file = args.input
    example_data = load_json(train_file)

    random.shuffle(example_data)
    split_idx = int(0.85 * len(example_data))
    train_data = example_data[:split_idx]
    val_data = example_data[split_idx:]

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = (
        f"./saves/{args.dataset}/{args.model_name}/{args.retriever}/{current_time}"
    )
    create_path(output_dir)

    logging_dir = os.path.join(output_dir, "logs")
    create_path(logging_dir)

    setup_logger(output_dir)

    model_name = args.model
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    ref_model.eval()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=128,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=128,
        lora_dropout=0,
        bias="none",
    )
    model = get_peft_model(model, peft_config)

    dpo_config = DPOConfig(
        output_dir=output_dir,
        logging_dir=logging_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=5e-6,
        warmup_ratio=0.1,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        seed=42,
        beta=0.4,
        max_length=2048,
        max_prompt_length=2048,
        max_grad_norm=1.0,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Training finished. TensorBoard logs are in {logging_dir}")
    print("Run: tensorboard --logdir {}".format(logging_dir))


if __name__ == "__main__":
    train_dpo()
