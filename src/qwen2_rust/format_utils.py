"""Shared formatting utilities for SFT training and evaluation."""

import ast
import json


def parse_json_field(field):
    """Parse a JSON or Python dict string field, returning dict."""
    if isinstance(field, dict):
        return field
    if not isinstance(field, str):
        return {"raw": str(field)} if field is not None else {}

    # Try JSON first
    try:
        return json.loads(field)
    except json.JSONDecodeError:
        pass

    # Try Python literal (handles single quotes, None, etc.)
    try:
        result = ast.literal_eval(field)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass

    return {"raw": field}


def format_for_sft(example):
    """Convert dataset format to chat messages for instruction tuning.

    Returns a dict with 'messages' containing user and assistant messages,
    matching the format expected by SFTTrainer.
    """
    input_data = parse_json_field(example["input_data"])
    output_data = parse_json_field(example["output_data"])

    task = example.get("task_category", "code_completion")
    crate = example.get("crate_name", "unknown")

    # Task-specific formatting
    if task == "api_usage_prediction":
        code = input_data.get("code", "")
        context = input_data.get("code_context", "")
        user_content = f"Predict the next API call for this Rust code from the {crate} crate:\n\n```rust\n{code}\n```"
        if context:
            user_content += f"\n\nContext:\n```rust\n{context}\n```"
        assistant_content = output_data.get("next_api_call", "")

    elif task == "bug_detection":
        buggy_code = input_data.get("buggy_code", "")
        context = input_data.get("code_context", "")
        user_content = f"Find and fix the bug in this Rust code from the {crate} crate:\n\n```rust\n{buggy_code}\n```"
        if context:
            user_content += f"\n\nContext:\n```rust\n{context}\n```"
        bug_desc = output_data.get("bug_description", "")
        fixed_code = output_data.get("fixed_code", "")
        assistant_content = f"{bug_desc}\n\nFixed code:\n```rust\n{fixed_code}\n```"

    elif task == "code_completion":
        prefix = input_data.get("prefix", "")
        suffix = input_data.get("suffix", "")
        user_content = f"Complete the following Rust code from the {crate} crate:\n\nPrefix:\n```rust\n{prefix}\n```\n\nSuffix:\n```rust\n{suffix}\n```"
        assistant_content = f"```rust\n{output_data.get('completion', '')}\n```"

    elif task == "code_explanation":
        code = input_data.get("code", "")
        context = input_data.get("code_context", "")
        user_content = f"Explain what this Rust code from the {crate} crate does:\n\n```rust\n{code}\n```"
        if context:
            user_content += f"\n\nContext:\n```rust\n{context}\n```"
        assistant_content = output_data.get("explanation", "")

    elif task == "code_generation":
        title = input_data.get("title", "")
        description = input_data.get("description", "")
        context = input_data.get("code_context", "")
        signature = input_data.get("function_signature", "")
        user_content = (
            f"Generate Rust code for the {crate} crate:\n\n**{title}**\n\n{description}"
        )
        if signature:
            user_content += f"\n\nFunction signature:\n```rust\n{signature}\n```"
        if context:
            user_content += f"\n\nContext:\n```rust\n{context}\n```"
        assistant_content = f"```rust\n{output_data.get('code', '')}\n```"

    elif task == "code_optimization":
        code_before = input_data.get("code_before", "")
        context = input_data.get("code_context", "")
        user_content = f"Optimize this Rust code from the {crate} crate:\n\n```rust\n{code_before}\n```"
        if context:
            user_content += f"\n\nContext:\n```rust\n{context}\n```"
        rationale = output_data.get("rationale", "")
        code_after = output_data.get("code_after", "")
        assistant_content = (
            f"{rationale}\n\nOptimized code:\n```rust\n{code_after}\n```"
        )

    elif task == "code_refactoring":
        code_before = input_data.get("code_before", "")
        context = input_data.get("code_context", "")
        user_content = f"Refactor this Rust code from the {crate} crate:\n\n```rust\n{code_before}\n```"
        if context:
            user_content += f"\n\nContext:\n```rust\n{context}\n```"
        rationale = output_data.get("rationale", "")
        code_after = output_data.get("code_after", "")
        assistant_content = (
            f"{rationale}\n\nRefactored code:\n```rust\n{code_after}\n```"
        )

    elif task == "code_review":
        code_before = input_data.get("code_before", "")
        context = input_data.get("code_context", "")
        user_content = f"Review this Rust code from the {crate} crate and suggest improvements:\n\n```rust\n{code_before}\n```"
        if context:
            user_content += f"\n\nContext:\n```rust\n{context}\n```"
        comment = output_data.get("review_comment", "")
        code_after = output_data.get("code_after", "")
        assistant_content = f"{comment}\n\nImproved code:\n```rust\n{code_after}\n```"

    elif task == "code_search":
        query = input_data.get("query", input_data.get("raw", ""))
        context = input_data.get("code_context", "")
        user_content = f"Find Rust code that matches this query: {query}"
        if context:
            user_content += f"\n\nContext:\n```rust\n{context}\n```"
        snippet = output_data.get("code_snippet", output_data.get("raw", ""))
        assistant_content = f"```rust\n{snippet}\n```"

    elif task == "code_summarization":
        code = input_data.get("code", "")
        context = input_data.get("code_context", "")
        user_content = f"Summarize what this Rust code from the {crate} crate does:\n\n```rust\n{code}\n```"
        if context:
            user_content += f"\n\nContext:\n```rust\n{context}\n```"
        assistant_content = output_data.get("summary", "")

    elif task == "comment_generation":
        code = input_data.get("code", "")
        context = input_data.get("code_context", "")
        user_content = f"Add inline comments to this Rust code from the {crate} crate:\n\n```rust\n{code}\n```"
        if context:
            user_content += f"\n\nContext:\n```rust\n{context}\n```"
        assistant_content = f"```rust\n{output_data.get('commented_code', '')}\n```"

    elif task == "docstring_generation":
        code = input_data.get("code", "")
        context = input_data.get("code_context", "")
        user_content = f"Generate documentation comments for this Rust code from the {crate} crate:\n\n```rust\n{code}\n```"
        if context:
            user_content += f"\n\nContext:\n```rust\n{context}\n```"
        assistant_content = output_data.get("docstring", "")

    elif task == "function_naming":
        code = input_data.get("code", "")
        context = input_data.get("code_context", "")
        user_content = f"Suggest a descriptive name for the `__placeholder__` function in this Rust code from the {crate} crate:\n\n```rust\n{code}\n```"
        if context:
            user_content += f"\n\nContext:\n```rust\n{context}\n```"
        assistant_content = output_data.get("function_name", "")

    elif task == "test_generation":
        code_context = input_data.get("code_context", "")
        code_to_test = input_data.get("code_to_test", "")
        test_context = input_data.get("test_context", "")
        user_content = f"Generate unit tests for this Rust code from the {crate} crate:\n\n```rust\n{code_to_test}\n```"
        if code_context:
            user_content += f"\n\nContext:\n```rust\n{code_context}\n```"
        if test_context:
            user_content += f"\n\nTest context:\n```rust\n{test_context}\n```"
        test_cases = output_data.get("test_cases", [])
        if isinstance(test_cases, list):
            assistant_content = "\n\n".join(test_cases)
        else:
            assistant_content = str(test_cases)

    elif task == "variable_naming":
        code = input_data.get("code", "")
        context = input_data.get("code_context", "")
        user_content = f"Suggest a descriptive name for the `__placeholder__` variable in this Rust code from the {crate} crate:\n\n```rust\n{code}\n```"
        if context:
            user_content += f"\n\nContext:\n```rust\n{context}\n```"
        assistant_content = output_data.get("variable_name", "")

    else:
        # Fallback for unknown task types
        user_content = f"Process this Rust code from the {crate} crate ({task}):\n\n{json.dumps(input_data, indent=2)}"
        assistant_content = json.dumps(output_data, indent=2)

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def format_for_sft_think(example):
    """Convert Strandset-Rust-Think dataset to chat messages with <think> tags.

    Expected columns: user_prompt, ground_truth, reasoning
    The reasoning is wrapped in <think> tags and prepended to the ground truth.
    """
    user_content = example["user_prompt"]
    reasoning = example.get("reasoning", "")
    ground_truth = example["ground_truth"]

    # Prepend reasoning in <think> tags
    if reasoning:
        assistant_content = f"<think>\n{reasoning}\n</think>\n\n{ground_truth}"
    else:
        assistant_content = ground_truth

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }
