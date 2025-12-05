# /// script
# dependencies = [
#     "datasets",
#     "qwen2-rust @ git+https://github.com/snowmead/Qwen2-7b-rust-sft.git",
# ]
# ///

"""Validate format_for_sft function by examining examples from each task category."""

from collections import defaultdict

from datasets import load_dataset

from qwen2_rust import format_for_sft

DATASET = "Fortytwo-Network/Strandset-Rust-v1"


def main():
    print("Loading dataset...")
    dataset = load_dataset(DATASET, split="train")
    print(f"Total examples: {len(dataset)}")

    # Group examples by task category (just 1 per category)
    by_category = defaultdict(list)
    for i, example in enumerate(dataset):
        task = example.get("task_category", "unknown")
        if len(by_category[task]) < 1:
            by_category[task].append((i, example))

    print(f"\nFound {len(by_category)} task categories\n")
    print("=" * 80)

    # Examine each category - show full content
    for task, examples in sorted(by_category.items()):
        print(f"\n{'=' * 80}")
        print(f"TASK CATEGORY: {task}")
        print(f"{'=' * 80}")

        for _, (i, example) in enumerate(examples):
            print(f"\n--- Example (index {i}) ---")
            print(f"Crate: {example.get('crate_name', 'unknown')}")

            # Show formatted result - FULL content
            formatted = format_for_sft(example)
            print("\n[USER MESSAGE]:")
            print(formatted["messages"][0]["content"])

            print("\n[ASSISTANT MESSAGE]:")
            print(formatted["messages"][1]["content"])

            print()

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
