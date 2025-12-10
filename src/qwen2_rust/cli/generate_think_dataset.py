"""Generate reasoning-augmented Rust dataset using Anthropic API.

This script uses the Anthropic API to generate reasoning traces for Rust
programming tasks. Given a task and its ground truth solution, Claude generates
the thinking process that would lead to that solution.

Supports two modes:
1. Real-time API: Immediate results, good for testing (default)
2. Batch API: 50% cost savings, recommended for large runs (5000+ examples)

Usage:
    # Real-time generation (for testing)
    uv run qwen2-generate-think generate -n 10 --seed 42

    # Batch workflow (50% cheaper, recommended for production)
    uv run qwen2-generate-think batch-prepare -n 5000 --seed 42
    uv run qwen2-generate-think batch-submit --batch-file .data/think-batch/batch_5000_42.jsonl
    uv run qwen2-generate-think batch-status --batch-id <id>
    uv run qwen2-generate-think batch-process --batch-id <id> --samples-file .data/think-batch/samples_5000_42.jsonl

    # Push completed dataset to HuggingFace Hub
    uv run qwen2-generate-think push --input .data/think-dataset/results.jsonl --hub-repo snowmead/Strandset-Rust-Think
"""

import argparse
import asyncio
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

import anthropic

# Load .env file if present (for ANTHROPIC_API_KEY)
from dotenv import load_dotenv

load_dotenv()

# System prompt for generating thinking traces (following Claude 4.5 best practices)
SYSTEM_PROMPT = """You are an expert Rust programmer generating thinking traces for a training dataset.

<task_context>
You will receive a Rust programming task and its correct response. Your job is to generate the internal thinking trace that would have occurred BEFORE writing that response - the stream-of-consciousness reasoning that leads to the solution.

These thinking traces will be used to train a smaller language model to develop better reasoning capabilities. The model will learn to think through problems before responding.
</task_context>

<output_format>
Write the thinking trace as internal monologue - first person, present tense, as thoughts occur:
- "Let me look at this code..."
- "I notice that..."
- "The key issue here is..."
- "I should consider..."
- "This means I need to..."

Do NOT explain or analyze the solution. Generate the thinking that would PRECEDE and LEAD TO the solution.
</output_format>

<quality_standards>
- The thinking must naturally arrive at the given response
- Include the problem-solving process: observation, analysis, consideration of options, decision
- Cover relevant Rust concepts only as they inform the solution
- Keep it concise and focused - this is thinking, not teaching
</quality_standards>"""

# Follow-up prompt to request thinking trace after showing the solution
REASONING_REQUEST_PROMPT = """Now generate the thinking trace that would have preceded this response.

<instructions>
Write the internal monologue that would occur while working through this problem BEFORE arriving at the response above. This is the thinking process, not an explanation.

Format as stream-of-consciousness reasoning:
- First person, present tense
- Show the problem-solving process as it unfolds
- Include observations, considerations, and decisions
- End with the decision/approach that leads directly to the response

Do not explain the response. Generate the thinking that would produce it.
</instructions>"""


def format_example_for_thinking(example: dict) -> tuple[str, str]:
    """Extract user prompt and ground truth response from example.

    Returns:
        tuple: (user_prompt, ground_truth) where user_prompt is the task/question
               and ground_truth is the verified correct solution from the dataset.
    """
    from qwen2_rust import format_for_sft

    # Get the formatted messages from our existing formatter
    formatted = format_for_sft(example)
    user_prompt = formatted["messages"][0]["content"]
    ground_truth = formatted["messages"][1]["content"]
    return user_prompt, ground_truth


# Default model
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"

# Model pricing (per million tokens)
MODEL_PRICING = {
    "claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
    "claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
}


def get_model_pricing(model: str) -> tuple[float, float]:
    """Get pricing for a model (input_price, output_price) per million tokens."""
    if model in MODEL_PRICING:
        p = MODEL_PRICING[model]
        return p["input"], p["output"]
    # Default to Sonnet pricing for unknown models
    return 3.0, 15.0


async def process_single_example(
    client: anthropic.AsyncAnthropic,
    example: dict,
    idx: int,
    semaphore: asyncio.Semaphore,
    model: str = DEFAULT_MODEL,
) -> dict | None:
    """Process a single example using Anthropic API.

    Generates reasoning traces that justify the ground truth response from the dataset.
    Uses a multi-turn conversation: show the task and solution, then ask for reasoning.
    """
    user_prompt, ground_truth = format_example_for_thinking(example)

    # Multi-turn conversation: show task + solution, then request reasoning
    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": ground_truth},
        {"role": "user", "content": REASONING_REQUEST_PROMPT},
    ]

    async with semaphore:
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=16000,
                messages=messages,
                system=SYSTEM_PROMPT,
            )

            # Extract text content from response
            text_content = []
            for block in response.content:
                if block.type == "text":
                    text_content.append(block.text)

            reasoning = "\n".join(text_content)

            # Calculate cost based on model
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            # Get model-specific pricing
            input_price, output_price = get_model_pricing(model)
            cost_usd = (
                input_tokens * input_price + output_tokens * output_price
            ) / 1_000_000

            return {
                "idx": idx,
                "crate_name": example.get("crate_name", ""),
                "task_category": example.get("task_category", ""),
                "original_input": example.get("input_data", ""),
                "original_output": example.get("output_data", ""),
                "user_prompt": user_prompt,
                "ground_truth": ground_truth,
                "reasoning": reasoning,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_cost_usd": cost_usd,
            }

        except anthropic.APIError as e:
            print(f"API error processing example {idx}: {e}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"Error processing example {idx}: {e}", file=sys.stderr)
            return None


def load_checkpoint(checkpoint_file: Path) -> set[int]:
    """Load processed indices from checkpoint file."""
    if not checkpoint_file.exists():
        return set()

    processed = set()
    with open(checkpoint_file) as f:
        for line in f:
            try:
                row = json.loads(line)
                processed.add(row["idx"])
            except (json.JSONDecodeError, KeyError):
                continue
    return processed


async def generate_dataset(args):
    """Generate reasoning-augmented dataset using Anthropic API."""
    from datasets import load_dataset

    print("Loading Strandset-Rust-v1 dataset...")
    dataset = load_dataset("Fortytwo-Network/Strandset-Rust-v1", split="train")
    print(f"Full dataset: {len(dataset)} examples")

    # Shuffle and sample
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    print(f"Using seed: {seed}")

    dataset = dataset.shuffle(seed=seed)
    num_examples = min(args.num_examples, len(dataset))
    dataset = dataset.select(range(num_examples))
    print(f"Selected {num_examples} examples")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    output_file = output_dir / f"think-dataset-{num_examples}-{seed}-{timestamp}.jsonl"
    checkpoint_file = output_dir / f"checkpoint-{num_examples}-{seed}.jsonl"

    # Load checkpoint if resuming
    processed_indices = set()
    if args.resume and Path(args.resume).exists():
        checkpoint_file = Path(args.resume)
        processed_indices = load_checkpoint(checkpoint_file)
        print(f"Resuming from checkpoint: {len(processed_indices)} already processed")

    # Filter out already processed examples
    examples_to_process = [
        (idx, example)
        for idx, example in enumerate(dataset)
        if idx not in processed_indices
    ]
    print(f"Examples to process: {len(examples_to_process)}")

    if not examples_to_process:
        print("All examples already processed!")
        return

    # Create async Anthropic client
    client = anthropic.AsyncAnthropic()

    # Concurrency control - limit parallel requests
    semaphore = asyncio.Semaphore(args.concurrency)

    # Process examples
    print(f"Model: {args.model}")
    print(f"Processing with concurrency={args.concurrency}...")

    results = []
    processed = 0
    errors = 0

    # Open checkpoint file for appending
    with open(checkpoint_file, "a") as checkpoint_f:
        # Process in batches to show progress
        batch_size = args.concurrency * 2
        for batch_start in range(0, len(examples_to_process), batch_size):
            batch = examples_to_process[batch_start : batch_start + batch_size]

            tasks = [
                process_single_example(
                    client,
                    example,
                    idx,
                    semaphore,
                    args.model,
                )
                for idx, example in batch
            ]

            batch_results = await asyncio.gather(*tasks)

            for result in batch_results:
                if result is not None:
                    results.append(result)
                    # Write to checkpoint immediately
                    checkpoint_f.write(json.dumps(result) + "\n")
                    checkpoint_f.flush()
                    processed += 1
                    print(
                        f"[{processed}/{len(examples_to_process)}] "
                        f"idx={result['idx']} ✓ reasoning={len(result['reasoning'])} chars"
                    )
                else:
                    errors += 1

    print("\nProcessing complete!")
    print(f"  Successful: {processed}")
    print(f"  Errors: {errors}")

    # Write final output
    print(f"\nWriting final dataset to {output_file}")
    with open(output_file, "w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")

    # Stats
    if results:
        avg_reasoning_len = sum(len(r["reasoning"]) for r in results) / len(results)
        print(f"\nAvg reasoning length: {avg_reasoning_len:.0f} chars")

        total_cost = sum(r.get("total_cost_usd", 0) or 0 for r in results)
        print(f"Cost: ${total_cost:.4f}")

    print(f"\nCheckpoint saved to: {checkpoint_file}")
    print(f"Final dataset: {output_file}")


# =============================================================================
# Batch API Functions (50% cost savings)
# =============================================================================


def format_example_for_batch(
    example: dict,
    idx: int,
    model: str = DEFAULT_MODEL,
) -> dict:
    """Convert a Strandset example to Anthropic batch format.

    Uses multi-turn conversation: show task + ground truth solution, then request reasoning.
    """
    user_prompt, ground_truth = format_example_for_thinking(example)

    # Multi-turn conversation: show task + solution, then request reasoning
    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": ground_truth},
        {"role": "user", "content": REASONING_REQUEST_PROMPT},
    ]

    return {
        "custom_id": f"rust-think-{idx}",
        "params": {
            "model": model,
            "max_tokens": 16000,
            "system": SYSTEM_PROMPT,
            "messages": messages,
        },
    }


def batch_prepare(args):
    """Prepare batch JSONL file for submission to Anthropic Batch API."""
    from datasets import load_dataset

    print("Loading Strandset-Rust-v1 dataset...")
    dataset = load_dataset("Fortytwo-Network/Strandset-Rust-v1", split="train")
    print(f"Full dataset: {len(dataset)} examples")

    # Shuffle and sample
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    print(f"Using seed: {seed}")

    dataset = dataset.shuffle(seed=seed)
    num_examples = min(args.num_examples, len(dataset))
    dataset = dataset.select(range(num_examples))
    print(f"Selected {num_examples} examples")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the sampled examples for later reference (needed for processing results)
    samples_file = output_dir / f"samples_{num_examples}_{seed}.jsonl"
    print(f"Saving sampled examples to {samples_file}")

    batch_requests = []
    with open(samples_file, "w") as f:
        for idx, example in enumerate(dataset):
            # Save original example
            f.write(json.dumps(dict(example)) + "\n")
            # Create batch request
            batch_requests.append(format_example_for_batch(example, idx, args.model))

    # Save batch requests
    batch_file = output_dir / f"batch_{num_examples}_{seed}.jsonl"
    print(f"Saving batch requests to {batch_file}")

    with open(batch_file, "w") as f:
        for req in batch_requests:
            f.write(json.dumps(req) + "\n")

    print(f"\n✅ Batch file ready: {batch_file}")
    print(f"   Samples file: {samples_file}")
    print(f"   Total requests: {len(batch_requests)}")
    print(f"   Model: {args.model}")

    # Estimate cost (with 50% batch discount)
    # Rough estimate: ~500 input tokens, ~2000 output tokens per example
    input_price, output_price = get_model_pricing(args.model)
    est_input = num_examples * 500
    est_output = num_examples * 2000
    est_cost = (est_input * input_price + est_output * output_price) / 1_000_000 * 0.5
    print(f"\n💰 Estimated cost (with 50% batch discount): ${est_cost:.2f}")

    print("\nNext step - submit the batch:")
    print(f"  uv run qwen2-generate-think batch-submit --batch-file {batch_file}")


def batch_submit(args):
    """Submit batch to Anthropic Batch API."""
    batch_file = Path(args.batch_file)
    if not batch_file.exists():
        print(f"Error: Batch file not found: {batch_file}")
        return

    print(f"Loading batch requests from {batch_file}")
    with open(batch_file) as f:
        requests = [json.loads(line) for line in f]
    print(f"Total requests: {len(requests)}")

    client = anthropic.Anthropic()
    print("Submitting batch to Anthropic...")
    batch = client.messages.batches.create(requests=requests)

    print("\n✅ Batch submitted!")
    print(f"   Batch ID: {batch.id}")
    print(f"   Status: {batch.processing_status}")
    print("\nNext step - check status:")
    print(f"  uv run qwen2-generate-think batch-status --batch-id {batch.id}")

    # Save batch ID for convenience
    batch_id_file = batch_file.parent / "latest_batch_id.txt"
    with open(batch_id_file, "w") as f:
        f.write(batch.id)
    print(f"\nBatch ID saved to: {batch_id_file}")


def batch_status(args):
    """Check batch status."""
    client = anthropic.Anthropic()
    batch = client.messages.batches.retrieve(args.batch_id)

    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.processing_status}")
    print(f"Created: {batch.created_at}")

    if batch.request_counts:
        total = (
            batch.request_counts.processing
            + batch.request_counts.succeeded
            + batch.request_counts.errored
            + batch.request_counts.canceled
            + batch.request_counts.expired
        )
        print(f"\nProgress: {batch.request_counts.succeeded}/{total}")
        print(f"  Processing: {batch.request_counts.processing}")
        print(f"  Succeeded: {batch.request_counts.succeeded}")
        print(f"  Errored: {batch.request_counts.errored}")
        print(f"  Canceled: {batch.request_counts.canceled}")
        print(f"  Expired: {batch.request_counts.expired}")

    if batch.processing_status == "ended":
        print("\n✅ Batch complete!")
        print("\nNext step - process results:")
        print(
            f"  uv run qwen2-generate-think batch-process --batch-id {args.batch_id} --samples-file <samples_file>"
        )


def batch_process(args):
    """Process batch results and create dataset."""
    from datasets import Dataset

    client = anthropic.Anthropic()

    print(f"Fetching results for batch {args.batch_id}...")
    batch = client.messages.batches.retrieve(args.batch_id)

    if batch.processing_status != "ended":
        print(f"Batch not complete. Status: {batch.processing_status}")
        return

    # Load original samples
    samples_file = Path(args.samples_file)
    if not samples_file.exists():
        print(f"Error: Samples file not found: {samples_file}")
        return

    print(f"Loading original samples from {samples_file}")
    with open(samples_file) as f:
        original_samples = [json.loads(line) for line in f]
    print(f"Loaded {len(original_samples)} original samples")

    # Fetch batch results
    print("Streaming batch results...")
    results = {}
    total_input_tokens = 0
    total_output_tokens = 0

    for result in client.messages.batches.results(args.batch_id):
        custom_id = result.custom_id
        idx = int(custom_id.split("-")[-1])

        if result.result.type == "succeeded":
            message = result.result.message

            # Extract text content
            text_content = []
            for block in message.content:
                if block.type == "text":
                    text_content.append(block.text)

            input_tokens = message.usage.input_tokens
            output_tokens = message.usage.output_tokens
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            results[idx] = {
                "reasoning": "\n".join(text_content),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        else:
            print(f"Warning: Request {custom_id} failed: {result.result.type}")

    print(f"Successfully processed {len(results)} results")

    # Create dataset rows
    dataset_rows = []
    for idx, sample in enumerate(original_samples):
        if idx not in results:
            continue

        user_prompt, ground_truth = format_example_for_thinking(sample)
        result = results[idx]

        dataset_rows.append(
            {
                "idx": idx,
                "crate_name": sample.get("crate_name", ""),
                "task_category": sample.get("task_category", ""),
                "original_input": sample.get("input_data", ""),
                "original_output": sample.get("output_data", ""),
                "user_prompt": user_prompt,
                "ground_truth": ground_truth,
                "reasoning": result["reasoning"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
            }
        )

    print(f"Created {len(dataset_rows)} dataset rows")

    # Save locally
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    output_file = (
        output_dir / f"think-dataset-{len(dataset_rows)}-batch-{timestamp}.jsonl"
    )

    print(f"Saving to {output_file}")
    with open(output_file, "w") as f:
        for row in dataset_rows:
            f.write(json.dumps(row) + "\n")

    # Push to HuggingFace if requested
    if args.push_to_hub:
        print("\nPushing to HuggingFace Hub...")
        hf_dataset = Dataset.from_list(dataset_rows)
        repo_id = args.hub_repo or f"snowmead/Strandset-Rust-Think-{len(dataset_rows)}"
        print(f"Pushing to {repo_id}")
        hf_dataset.push_to_hub(repo_id, private=False)
        print(f"Dataset available at: https://huggingface.co/datasets/{repo_id}")

    # Print stats
    print("\n📊 Results:")
    print(f"  Total examples: {len(dataset_rows)}")
    if dataset_rows:
        avg_reasoning = sum(len(r["reasoning"]) for r in dataset_rows) / len(
            dataset_rows
        )
        print(f"  Avg reasoning length: {avg_reasoning:.0f} chars")

    # Cost (with 50% batch discount)
    cost_usd = (total_input_tokens * 3 + total_output_tokens * 15) / 1_000_000 * 0.5
    print("\n💰 Cost (with 50% batch discount):")
    print(f"  Input tokens: {total_input_tokens:,}")
    print(f"  Output tokens: {total_output_tokens:,}")
    print(f"  Total: ${cost_usd:.4f}")

    print(f"\n✅ Dataset saved to: {output_file}")


def push_to_hub(args):
    """Push generated dataset to HuggingFace Hub."""
    from datasets import Dataset

    input_file = Path(args.input)
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return

    print(f"Loading dataset from {input_file}")
    with open(input_file) as f:
        rows = [json.loads(line) for line in f]
    print(f"Loaded {len(rows)} rows")

    if not rows:
        print("No rows to push!")
        return

    # Create HuggingFace dataset
    print("\nCreating HuggingFace dataset...")
    hf_dataset = Dataset.from_list(rows)

    repo_id = args.hub_repo
    print(f"Pushing to {repo_id}")
    hf_dataset.push_to_hub(repo_id, private=args.private)
    print(f"Dataset available at: https://huggingface.co/datasets/{repo_id}")

    # Print stats
    print("\nDataset stats:")
    print(f"  Total rows: {len(rows)}")
    if rows and "reasoning" in rows[0]:
        avg_reasoning = sum(len(r.get("reasoning", "")) for r in rows) / len(rows)
        print(f"  Avg reasoning length: {avg_reasoning:.0f} chars")


def stats(args):
    """Show statistics for a generated dataset."""
    input_file = Path(args.input)
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return

    print(f"Loading dataset from {input_file}")
    with open(input_file) as f:
        rows = [json.loads(line) for line in f]
    print(f"Total rows: {len(rows)}")

    if rows and "reasoning" in rows[0]:
        reasoning_lengths = [len(r.get("reasoning", "")) for r in rows]
        print("\nReasoning length (chars):")
        print(f"  Min: {min(reasoning_lengths)}")
        print(f"  Max: {max(reasoning_lengths)}")
        print(f"  Avg: {sum(reasoning_lengths) / len(reasoning_lengths):.0f}")

    # Task category distribution
    categories = {}
    for r in rows:
        cat = r.get("task_category", "unknown")
        if cat not in categories:
            categories[cat] = 0
        categories[cat] += 1

    print("\nTask category distribution:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    # Cost
    total_cost = sum(r.get("total_cost_usd", 0) or 0 for r in rows)
    if total_cost > 0:
        print(f"\nTotal cost: ${total_cost:.4f}")
        print(f"Cost per example: ${total_cost / len(rows):.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate reasoning-augmented Rust dataset using Anthropic API"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ==========================================================================
    # Real-time API commands (for testing)
    # ==========================================================================

    # Generate command (real-time, for testing)
    generate = subparsers.add_parser(
        "generate", help="Generate dataset using real-time API (for testing)"
    )
    generate.add_argument(
        "-n", "--num-examples", type=int, default=100, help="Number of examples"
    )
    generate.add_argument("--seed", type=int, default=42, help="Random seed")
    generate.add_argument(
        "--output-dir",
        type=str,
        default=".data/think-dataset",
        help="Output directory",
    )
    generate.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Max concurrent requests (default: 3)",
    )
    generate.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint file to resume from",
    )
    generate.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Claude model to use (default: {DEFAULT_MODEL})",
    )

    # ==========================================================================
    # Batch API commands (50% cost savings, for production)
    # ==========================================================================

    # Batch prepare command
    batch_prep = subparsers.add_parser(
        "batch-prepare",
        help="Prepare batch file for Anthropic Batch API (50%% cheaper)",
    )
    batch_prep.add_argument(
        "-n", "--num-examples", type=int, default=5000, help="Number of examples"
    )
    batch_prep.add_argument("--seed", type=int, default=42, help="Random seed")
    batch_prep.add_argument(
        "--output-dir",
        type=str,
        default=".data/think-batch",
        help="Output directory",
    )
    batch_prep.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Claude model to use (default: {DEFAULT_MODEL})",
    )

    # Batch submit command
    batch_sub = subparsers.add_parser(
        "batch-submit", help="Submit batch to Anthropic Batch API"
    )
    batch_sub.add_argument(
        "--batch-file", type=str, required=True, help="Batch JSONL file"
    )

    # Batch status command
    batch_stat = subparsers.add_parser("batch-status", help="Check batch status")
    batch_stat.add_argument("--batch-id", type=str, required=True, help="Batch ID")

    # Batch process command
    batch_proc = subparsers.add_parser(
        "batch-process", help="Process batch results into dataset"
    )
    batch_proc.add_argument("--batch-id", type=str, required=True, help="Batch ID")
    batch_proc.add_argument(
        "--samples-file", type=str, required=True, help="Original samples JSONL"
    )
    batch_proc.add_argument(
        "--output-dir",
        type=str,
        default=".data/think-dataset",
        help="Output directory",
    )
    batch_proc.add_argument(
        "--push-to-hub", action="store_true", help="Push to HuggingFace Hub"
    )
    batch_proc.add_argument("--hub-repo", type=str, help="HuggingFace repo ID")

    # ==========================================================================
    # Utility commands
    # ==========================================================================

    # Push command
    push = subparsers.add_parser("push", help="Push dataset to HuggingFace Hub")
    push.add_argument("--input", type=str, required=True, help="Input JSONL file")
    push.add_argument("--hub-repo", type=str, required=True, help="HuggingFace repo ID")
    push.add_argument("--private", action="store_true", help="Make dataset private")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show dataset statistics")
    stats_parser.add_argument(
        "--input", type=str, required=True, help="Input JSONL file"
    )

    args = parser.parse_args()

    if args.command == "generate":
        asyncio.run(generate_dataset(args))
    elif args.command == "batch-prepare":
        batch_prepare(args)
    elif args.command == "batch-submit":
        batch_submit(args)
    elif args.command == "batch-status":
        batch_status(args)
    elif args.command == "batch-process":
        batch_process(args)
    elif args.command == "push":
        push_to_hub(args)
    elif args.command == "stats":
        stats(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
