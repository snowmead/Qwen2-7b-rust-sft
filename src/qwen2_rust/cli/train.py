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

"""SFT fine-tuning of Qwen3-4B-Instruct on Fortytwo-Network/Strandset-Rust-v1."""

import argparse
import random
from datetime import UTC, datetime

# Presets based on hyperparameter experiments (10k examples, A100-large)
# Results: small-batch achieved best eval loss (0.766), aggressive-lr close second (0.767)
PRESETS = {
    "small-batch": {
        # Best generalization - 4x more gradient updates with implicit regularization
        "batch_size": 2,
        "grad_accum": 2,
        "lr": 2e-5,
        "warmup_ratio": 0.1,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
    },
    "aggressive-lr": {
        # Fast convergence - nearly same quality as small-batch but 4x fewer steps
        "batch_size": 4,
        "grad_accum": 4,
        "lr": 5e-5,
        "warmup_ratio": 0.2,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
    },
    "high-capacity": {
        # For larger datasets (100k+) - more adapter capacity
        "batch_size": 4,
        "grad_accum": 4,
        "lr": 2e-5,
        "warmup_ratio": 0.1,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
    },
}


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-4B on Rust code dataset"
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(PRESETS.keys()),
        default=None,
        help=f"Use preset config: {', '.join(PRESETS.keys())} (individual args override)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        default=False,
        help="Push the trained model to Hugging Face Hub (default: False)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for dataset shuffling (default: random)",
    )
    parser.add_argument(
        "-n",
        "--num-examples",
        type=int,
        default=1000,
        help="Number of examples to train on (default: 1000)",
    )
    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-device training batch size (default: 8)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio (default: 0.1)",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="cosine",
        help="Learning rate scheduler type (default: cosine)",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Logging frequency in steps (default: 10)",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Save checkpoint frequency in steps (default: 500)",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Evaluation frequency in steps (default: 500)",
    )
    # LoRA hyperparameters
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (default: 0.05)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom run name for Trackio (default: auto-generated)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Model to fine-tune (default: Qwen/Qwen3-4B-Instruct-2507)",
    )
    args = parser.parse_args()

    # Apply preset if specified (CLI args override preset values)
    if args.preset:
        preset = PRESETS[args.preset]
        print(f"Using preset: {args.preset}")
        # Only apply preset values if user didn't explicitly set them
        defaults = parser.parse_args([])
        if args.batch_size == defaults.batch_size:
            args.batch_size = preset["batch_size"]
        if args.grad_accum == defaults.grad_accum:
            args.grad_accum = preset["grad_accum"]
        if args.lr == defaults.lr:
            args.lr = preset["lr"]
        if args.warmup_ratio == defaults.warmup_ratio:
            args.warmup_ratio = preset["warmup_ratio"]
        if args.lora_r == defaults.lora_r:
            args.lora_r = preset["lora_r"]
        if args.lora_alpha == defaults.lora_alpha:
            args.lora_alpha = preset["lora_alpha"]
        if args.lora_dropout == defaults.lora_dropout:
            args.lora_dropout = preset["lora_dropout"]

    import trackio
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    from qwen2_rust import format_for_sft

    # Load dataset first (needed for num_examples calculation)
    print("Loading dataset...")
    dataset = load_dataset("Fortytwo-Network/Strandset-Rust-v1", split="train")
    print(f"Full dataset: {len(dataset)} examples")

    # Generate run name early so we can use it for both trackio and SFTConfig
    shuffle_seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    num_examples = min(args.num_examples, len(dataset))
    run_name = (
        args.run_name
        or f"sft-{num_examples}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    )

    # Build hyperparameters config for logging
    hyperparams = {
        # Model
        "model": args.model,
        "dataset": "Fortytwo-Network/Strandset-Rust-v1",
        "num_examples": num_examples,
        "seed": shuffle_seed,
        # Training
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "learning_rate": args.lr,
        "max_length": args.max_length,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler": args.lr_scheduler,
        # LoRA
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        # Preset (if used)
        "preset": args.preset,
    }

    # Initialize Trackio with config before TRL takes over
    print(f"Initializing Trackio run: {run_name}")
    print(f"Hyperparameters: {hyperparams}")
    trackio.init(
        project="qwen2-rust-finetune",
        name=run_name,
        space_id="snowmead/trackio",
        dataset_id="snowmead/trackio-dataset",
        config=hyperparams,
    )

    # Take subset (using pre-computed values)
    print(f"Shuffle seed: {shuffle_seed}")
    dataset = dataset.shuffle(seed=shuffle_seed).select(range(num_examples))
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
        output_dir="qwen3-4b-rust-sft",
        push_to_hub=args.push_to_hub,
        hub_model_id="snowmead/qwen3-4b-rust-sft" if args.push_to_hub else None,
        hub_strategy="every_save" if args.push_to_hub else "end",
        hub_private_repo=False,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=2,
        eval_strategy="steps",
        report_to="trackio",
        project="qwen2-rust-finetune",
        run_name=run_name,
        # Hyperparameters
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_length,
        gradient_checkpointing=True,
        bf16=True,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler,
    )

    # LoRA configuration for 4B model
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
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

    print(f"Initializing trainer with model: {args.model}")
    trainer = SFTTrainer(
        model=args.model,
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
        print("Complete! Model at: https://huggingface.co/snowmead/qwen3-4b-rust-sft")
    else:
        print("Complete! Model saved locally to: qwen3-4b-rust-sft/")


if __name__ == "__main__":
    main()
