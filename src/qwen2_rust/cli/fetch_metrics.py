#!/usr/bin/env python3
"""Fetch Trackio training metrics from HuggingFace Spaces for conversational analysis."""

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import HfApi, list_datasets


def discover_trackio_dataset(
    username: str = "snowmead", project: str = "qwen2-rust-finetune"
) -> str | None:
    """
    Discover Trackio dataset location on HuggingFace Hub.

    Try multiple strategies:
    1. Search for datasets containing project name
    2. Look for common Trackio dataset naming patterns
    3. List user's datasets and find matches

    Args:
        username: HuggingFace username
        project: Trackio project name

    Returns:
        Dataset ID if found, None otherwise
    """
    HfApi()

    # Strategy 1: Search for datasets with project name
    print(f"Searching for datasets with project '{project}'...")
    try:
        datasets = list(list_datasets(author=username, limit=100))
        for ds in datasets:
            if project in ds.id.lower() or "trackio" in ds.id.lower():
                print(f"Found candidate dataset: {ds.id}")
                return ds.id
    except Exception as e:
        print(f"Dataset search failed: {e}")

    # Strategy 2: Try common Trackio patterns
    common_patterns = [
        f"{username}/trackio",
        f"{username}/trackio_dataset",
        f"{username}/{project}",
        f"{username}/{project}_dataset",
    ]

    for pattern in common_patterns:
        try:
            # Try to access the dataset
            load_dataset(pattern, split="train", streaming=True)
            print(f"Found dataset: {pattern}")
            return pattern
        except Exception:
            continue

    return None


def fetch_run_metrics(
    run_name: str, dataset_id: str, project: str = "qwen2-rust-finetune"
) -> dict[str, Any]:
    """
    Fetch all metrics for a specific run from Trackio dataset.

    Args:
        run_name: Run name (e.g., "sft-2000-20251205-175811")
        dataset_id: HuggingFace dataset ID
        project: Trackio project name

    Returns:
        Dictionary with training history and summary stats
    """
    print(f"Loading dataset {dataset_id}...")
    ds = load_dataset(dataset_id, split="train")

    # Filter for this specific run
    print(f"Filtering for run: {run_name}")
    run_data = ds.filter(lambda x: x.get("run_name") == run_name)

    if len(run_data) == 0:
        raise ValueError(f"No data found for run: {run_name}")

    print(f"Found {len(run_data)} metric records for run {run_name}")

    # Convert to list and sort by step
    metrics_list = []
    for record in run_data:
        metrics_list.append(record)

    metrics_list.sort(key=lambda x: x.get("step", 0))

    # Extract metrics history
    history = []
    for record in metrics_list:
        step_metrics = {"step": record.get("step", 0)}

        # Extract common metrics (Trackio uses prefixes like train/loss)
        if "train/loss" in record and record["train/loss"] is not None:
            step_metrics["train_loss"] = float(record["train/loss"])
        if "eval/loss" in record and record["eval/loss"] is not None:
            step_metrics["eval_loss"] = float(record["eval/loss"])
        if (
            "train/learning_rate" in record
            and record["train/learning_rate"] is not None
        ):
            step_metrics["learning_rate"] = float(record["train/learning_rate"])
        if "train/grad_norm" in record and record["train/grad_norm"] is not None:
            step_metrics["grad_norm"] = float(record["train/grad_norm"])
        if "train/epoch" in record and record["train/epoch"] is not None:
            step_metrics["epoch"] = float(record["train/epoch"])
        if "timestamp" in record:
            step_metrics["timestamp"] = record["timestamp"]

        history.append(step_metrics)

    # Calculate summary statistics
    train_losses = [m["train_loss"] for m in history if "train_loss" in m]
    eval_losses = [m["eval_loss"] for m in history if "eval_loss" in m]

    summary = {
        "total_steps": history[-1]["step"] if history else 0,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_eval_loss": eval_losses[-1] if eval_losses else None,
        "best_eval_loss": min(eval_losses) if eval_losses else None,
        "best_eval_step": (
            history[[m.get("eval_loss") for m in history].index(min(eval_losses))][
                "step"
            ]
            if eval_losses
            else None
        ),
    }

    return {
        "run_id": run_name,
        "project": project,
        "dataset": dataset_id,
        "summary": summary,
        "metrics_history": history,
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch Trackio metrics for conversational analysis"
    )
    parser.add_argument(
        "--run",
        type=str,
        required=True,
        help="Run name (e.g., sft-2000-20251205-175811)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="qwen2-rust-finetune",
        help="Trackio project name",
    )
    parser.add_argument(
        "--username", type=str, default="snowmead", help="HuggingFace username"
    )
    parser.add_argument(
        "--dataset", type=str, help="Dataset ID (auto-discover if not provided)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file (default: .data/{run_id}.json, or stdout if --stdout)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print to stdout instead of saving to file",
    )

    args = parser.parse_args()

    # Discover dataset if not provided
    dataset_id = args.dataset
    if not dataset_id:
        print("Discovering Trackio dataset...")
        dataset_id = discover_trackio_dataset(args.username, args.project)
        if not dataset_id:
            print("ERROR: Could not find Trackio dataset")
            print("Please specify dataset ID with --dataset")
            return 1

    # Fetch metrics
    try:
        metrics = fetch_run_metrics(args.run, dataset_id, args.project)

        # Format output
        output = json.dumps(metrics, indent=2)

        # Determine output path
        if args.stdout:
            # Print to stdout
            print("\n" + "=" * 80)
            print(output)
            print("=" * 80)
        else:
            # Save to file
            if args.output:
                output_path = Path(args.output)
            else:
                # Default: .data/{run_id}.json
                data_dir = Path(".data")
                data_dir.mkdir(exist_ok=True)
                output_path = data_dir / f"{args.run}.json"

            with open(output_path, "w") as f:
                f.write(output)
            print(f"Metrics written to {output_path}")

        return 0

    except Exception as e:
        print(f"ERROR: Failed to fetch metrics: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
