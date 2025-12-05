# /// script
# dependencies = [
#     "trl>=0.12.0",
#     "peft>=0.7.0",
#     "transformers>=4.36.0",
#     "accelerate>=0.24.0",
#     "trackio",
#     "bitsandbytes",
#     "qwen2-rust @ git+https://github.com/snowmead/Qwen2-7b-rust-sft.git",
# ]
# ///

"""
SFT fine-tuning of Qwen2-7B-Instruct on Fortytwo-Network/Strandset-Rust-v1.
Quick test run with 1000 examples.
"""

import argparse
import random


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2-7B on Rust code dataset"
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        default=False,
        help="Push the trained model to Hugging Face Hub (default: False)",
    )
    return parser.parse_args()


args = parse_args()

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from qwen2_rust import format_for_sft

# Load dataset
print("Loading dataset...")
dataset = load_dataset("Fortytwo-Network/Strandset-Rust-v1", split="train")
print(f"Full dataset: {len(dataset)} examples")

# Take subset for quick test
shuffle_seed = random.randint(0, 2**32 - 1)
print(f"Shuffle seed: {shuffle_seed}")
dataset = dataset.shuffle(seed=shuffle_seed).select(range(min(1000, len(dataset))))
print(f"Using subset: {len(dataset)} examples")


print("Formatting dataset...")
dataset = dataset.map(format_for_sft, remove_columns=dataset.column_names)
print("Dataset formatted")

# Create train/eval split
print("Creating train/eval split...")
dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"]
print(f"   Train: {len(train_dataset)} examples")
print(f"   Eval: {len(eval_dataset)} examples")

# Training configuration
config = SFTConfig(
    output_dir="qwen2-7b-rust-sft",
    push_to_hub=args.push_to_hub,
    hub_model_id="snowmead/qwen2-7b-rust-sft" if args.push_to_hub else None,
    hub_strategy="every_save" if args.push_to_hub else "end",
    hub_private_repo=False,
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    eval_steps=100,
    save_total_limit=2,
    eval_strategy="steps",
    report_to="trackio",
    project="qwen2-rust-finetune",
    run_name="quick-test-1k",
    # Hyperparameters
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    max_length=2048,
    gradient_checkpointing=True,
    bf16=True,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
)

# LoRA configuration for 7B model
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

print("Initializing trainer...")
trainer = SFTTrainer(
    model="Qwen/Qwen2-7B-Instruct",
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=config,
    peft_config=peft_config,
)

print("Starting training...")
trainer.train()

if args.push_to_hub:
    print("Pushing to Hub...")
    trainer.push_to_hub()
    print("Complete! Model at: https://huggingface.co/snowmead/qwen2-7b-rust-sft")
else:
    print("Complete! Model saved locally to: qwen2-7b-rust-sft/")
