# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0.0",
#     "transformers>=4.45.0",
#     "peft>=0.13.0",
#     "accelerate>=0.34.0",
#     "huggingface-hub>=0.25.0",
#     "safetensors>=0.4.0",
#     "llama-cpp-python>=0.3.0",
# ]
# ///
"""
Merge PEFT LoRA adapter with base model and convert to GGUF.

This script:
1. Loads a base model and LoRA adapter
2. Merges them into a standalone model
3. Converts to GGUF format for LM Studio / llama.cpp
4. Optionally pushes to HuggingFace Hub

Usage (local):
    uv run src/qwen2_rust/cli/merge_and_gguf.py \
        --base-model Qwen/Qwen3-0.6B \
        --adapter snowmead/qwen3-0.6b-rust-sft \
        --output-repo snowmead/qwen3-0.6b-rust-sft-GGUF \
        --quant Q8_0 \
        --push-to-hub

Usage (HF Jobs):
    hf jobs uv run --flavor l4x1 --timeout 1h --secrets HF_TOKEN \
        "https://raw.githubusercontent.com/snowmead/Qwen2-7b-rust-sft/main/src/qwen2_rust/cli/merge_and_gguf.py" \
        -- --base-model Qwen/Qwen3-0.6B \
           --adapter snowmead/qwen3-0.6b-rust-sft \
           --output-repo snowmead/qwen3-0.6b-rust-sft-GGUF \
           --quant Q8_0 \
           --push-to-hub
"""

import argparse
import os
import subprocess
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi, upload_file


def main():
    parser = argparse.ArgumentParser(
        description="Merge PEFT LoRA adapter with base model and convert to GGUF"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Base model name or path (e.g., Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Adapter model name or path (e.g., snowmead/qwen3-0.6b-rust-sft)",
    )
    parser.add_argument(
        "--output-repo",
        type=str,
        required=True,
        help="Output repository name for GGUF model",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push GGUF model to HuggingFace Hub",
    )
    parser.add_argument(
        "--quant",
        type=str,
        default="Q8_0",
        choices=["F16", "F32", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"],
        help="GGUF quantization type (default: Q8_0 for best quality)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="gguf-output",
        help="Local output directory",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and merge
    print("=" * 60)
    print("Step 1: Loading and merging model")
    print("=" * 60)

    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading tokenizer from: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )

    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter,
        device_map="auto",
    )

    print("Merging adapter with base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {merged_dir}")
    model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    # Step 2: Convert to GGUF using llama.cpp's convert script
    print("=" * 60)
    print("Step 2: Converting to GGUF")
    print("=" * 60)

    # Clone llama.cpp if not present
    llama_cpp_dir = output_dir / "llama.cpp"
    if not llama_cpp_dir.exists():
        print("Cloning llama.cpp...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git", str(llama_cpp_dir)],
            check=True,
        )

    # Install conversion dependencies
    print("Installing conversion dependencies...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(llama_cpp_dir / "requirements.txt")],
        check=True,
    )

    # Convert to GGUF
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    model_name = args.adapter.split("/")[-1]
    gguf_filename = f"{model_name}-{args.quant}.gguf"
    gguf_path = output_dir / gguf_filename

    print(f"Converting to GGUF ({args.quant})...")
    subprocess.run(
        [
            sys.executable,
            str(convert_script),
            str(merged_dir),
            "--outfile", str(gguf_path),
            "--outtype", args.quant.lower(),
        ],
        check=True,
    )

    print(f"GGUF file created: {gguf_path}")
    print(f"Size: {gguf_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Step 3: Push to Hub
    if args.push_to_hub:
        print("=" * 60)
        print("Step 3: Pushing to HuggingFace Hub")
        print("=" * 60)

        api = HfApi()

        # Create repo if it doesn't exist
        try:
            api.create_repo(args.output_repo, exist_ok=True)
        except Exception as e:
            print(f"Repo creation note: {e}")

        # Upload GGUF file
        print(f"Uploading {gguf_filename} to {args.output_repo}...")
        api.upload_file(
            path_or_fileobj=str(gguf_path),
            path_in_repo=gguf_filename,
            repo_id=args.output_repo,
        )

        # Create a simple README
        readme_content = f"""---
license: apache-2.0
base_model: {args.base_model}
tags:
  - gguf
  - llama.cpp
  - lm-studio
  - qwen3
  - rust
  - fine-tuned
---

# {model_name} (GGUF)

This is a GGUF quantized version of [{args.adapter}](https://huggingface.co/{args.adapter}).

## Model Details
- **Base Model**: [{args.base_model}](https://huggingface.co/{args.base_model})
- **LoRA Adapter**: [{args.adapter}](https://huggingface.co/{args.adapter})
- **Quantization**: {args.quant}
- **File**: `{gguf_filename}`

## Usage

### LM Studio
1. Download `{gguf_filename}`
2. Place in your LM Studio models folder
3. Load and chat!

### llama.cpp
```bash
./llama-cli -m {gguf_filename} -c 2048 -cnv
```

## Training
This model was fine-tuned on the [Strandset-Rust-Think](https://huggingface.co/datasets/snowmead/Strandset-Rust-Think) dataset
for Rust code generation with reasoning capabilities.
"""
        readme_path = output_dir / "README.md"
        readme_path.write_text(readme_content)

        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=args.output_repo,
        )

        print(f"Complete! GGUF model available at: https://huggingface.co/{args.output_repo}")
    else:
        print(f"Complete! GGUF file saved to: {gguf_path}")


if __name__ == "__main__":
    main()
