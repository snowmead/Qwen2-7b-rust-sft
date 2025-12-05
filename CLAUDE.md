# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fine-tuning utilities for Qwen2-7B-Instruct on the Fortytwo-Network/Strandset-Rust-v1 dataset. Uses TRL (Transformer Reinforcement Learning) with LoRA for parameter-efficient fine-tuning.

## Commands

### Package Management (uv)

```bash
uv sync                     # Install dependencies
uv add <package>            # Add dependency
uv add --dev <package>      # Add dev dependency
```

### CLI Tools

```bash
uv run qwen2-train                     # Train with defaults (1000 examples)
uv run qwen2-train -n 2000             # Specify example count
uv run qwen2-train --push-to-hub       # Push to HuggingFace Hub
uv run qwen2-train --seed 42           # Reproducible shuffling

uv run qwen2-validate                  # Validate dataset format
uv run qwen2-evaluate                  # Evaluate trained model

uv run qwen2-fetch-metrics --run "sft-2000-20251205-175811"  # Fetch training metrics
uv run qwen2-fetch-metrics --run "..." --stdout              # Print to stdout
```

### Linting & Formatting (ruff)

```bash
uv run ruff check .         # Lint
uv run ruff check --fix .   # Lint with auto-fix
uv run ruff format .        # Format
```

## Architecture

```
src/qwen2_rust/
├── __init__.py           # Exports format_for_sft, parse_json_field
├── format_utils.py       # Dataset formatting: converts raw examples to chat messages
└── cli/
    ├── train.py          # SFT training (PEP 723 UV script for HF Jobs)
    ├── fetch_metrics.py  # Fetch Trackio metrics from HF Spaces
    ├── evaluate.py       # Model evaluation
    └── validate.py       # Dataset validation
```

### Key Components

- **`format_for_sft()`** (`format_utils.py`): Converts dataset examples to chat message format for SFTTrainer. Handles 14 task types (code_completion, bug_detection, test_generation, etc.).

- **`train.py`**: PEP 723 UV script with inline dependencies for HuggingFace Jobs. Uses LoRA config targeting attention and MLP layers with `r=16, lora_alpha=32`.

- **`.data/`**: Git-ignored directory for training metrics JSON files.
