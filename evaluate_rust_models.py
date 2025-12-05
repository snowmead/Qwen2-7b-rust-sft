# /// script
# dependencies = [
#     "transformers>=4.36.0",
#     "accelerate>=0.24.0",
#     "peft>=0.7.0",
#     "datasets",
#     "torch",
#     "huggingface_hub",
#     "qwen2-rust @ git+https://github.com/snowmead/Qwen2-7b-rust-sft.git",
# ]
# ///

"""
Evaluate base Qwen2-7B-Instruct vs fine-tuned snowmead/qwen2-7b-rust-sft.

Compares performance on the test split of the Rust dataset using:
- Perplexity on held-out test examples
- Token-based code similarity (F1)
- Qualitative comparison of generated outputs
"""

import json

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from qwen2_rust import format_for_sft

# Configuration
BASE_MODEL = "Qwen/Qwen2-7B-Instruct"
FINETUNED_MODEL = "snowmead/qwen2-7b-rust-sft"
DATASET = "Fortytwo-Network/Strandset-Rust-v1"
NUM_EVAL_SAMPLES = 100
MAX_NEW_TOKENS = 256


def load_base_model():
    """Load the base Qwen2-7B-Instruct model."""
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


def load_finetuned_model():
    """Load the fine-tuned model with LoRA adapter."""
    print("Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)
    return model, tokenizer


def compute_perplexity(model, tokenizer, texts, max_length=512):
    """Compute average perplexity over a list of texts."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(model.device)

            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity


def generate_response(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS):
    """Generate a response from the model."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=generation_config)

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return response


def simple_code_similarity(generated, expected):
    """Simple token-based similarity for code."""
    gen_tokens = set(generated.split())
    exp_tokens = set(expected.split())

    if not exp_tokens:
        return 0.0

    intersection = gen_tokens & exp_tokens
    precision = len(intersection) / len(gen_tokens) if gen_tokens else 0
    recall = len(intersection) / len(exp_tokens) if exp_tokens else 0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def main():
    print("=" * 60)
    print("EVALUATION: Base vs Fine-tuned Qwen2-7B on Rust Tasks")
    print("=" * 60)

    # Load test dataset
    print("\nLoading test dataset...")
    dataset = load_dataset(DATASET, split="test")
    print(f"Test set size: {len(dataset)} examples")

    # Use subset for evaluation
    eval_samples = min(NUM_EVAL_SAMPLES, len(dataset))
    dataset = dataset.shuffle(seed=42).select(range(eval_samples))
    print(f"Evaluating on {eval_samples} samples")

    # Prepare evaluation data (format once per example)
    formatted = [format_for_sft(ex) for ex in dataset]
    prompts = [f["messages"][0]["content"] for f in formatted]
    expected_outputs = [f["messages"][1]["content"] for f in formatted]
    full_texts = [f"{p}\n\n{e}" for p, e in zip(prompts, expected_outputs, strict=True)]

    # Evaluate base model
    print("\n" + "=" * 40)
    print("EVALUATING BASE MODEL")
    print("=" * 40)
    base_model, base_tokenizer = load_base_model()

    print("\nComputing perplexity...")
    base_ppl = compute_perplexity(base_model, base_tokenizer, full_texts[:20])
    print(f"Base model perplexity: {base_ppl:.2f}")

    print("\nGenerating sample outputs...")
    base_outputs = []
    base_similarities = []
    for i in range(min(10, len(prompts))):
        output = generate_response(base_model, base_tokenizer, prompts[i])
        base_outputs.append(output)
        sim = simple_code_similarity(output, expected_outputs[i])
        base_similarities.append(sim)

    avg_base_sim = sum(base_similarities) / len(base_similarities)
    print(f"Base model avg code similarity: {avg_base_sim:.4f}")

    # Free memory
    del base_model
    torch.cuda.empty_cache()

    # Evaluate fine-tuned model
    print("\n" + "=" * 40)
    print("EVALUATING FINE-TUNED MODEL")
    print("=" * 40)
    ft_model, ft_tokenizer = load_finetuned_model()

    print("\nComputing perplexity...")
    ft_ppl = compute_perplexity(ft_model, ft_tokenizer, full_texts[:20])
    print(f"Fine-tuned model perplexity: {ft_ppl:.2f}")

    print("\nGenerating sample outputs...")
    ft_outputs = []
    ft_similarities = []
    for i in range(min(10, len(prompts))):
        output = generate_response(ft_model, ft_tokenizer, prompts[i])
        ft_outputs.append(output)
        sim = simple_code_similarity(output, expected_outputs[i])
        ft_similarities.append(sim)

    avg_ft_sim = sum(ft_similarities) / len(ft_similarities)
    print(f"Fine-tuned model avg code similarity: {avg_ft_sim:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\n{'Metric':<30} {'Base Model':<15} {'Fine-tuned':<15} {'Delta':<10}")
    print("-" * 70)
    print(
        f"{'Perplexity':<30} {base_ppl:<15.2f} {ft_ppl:<15.2f} {ft_ppl - base_ppl:<10.2f}"
    )
    print(
        f"{'Code Similarity (F1)':<30} {avg_base_sim:<15.4f} {avg_ft_sim:<15.4f} {avg_ft_sim - avg_base_sim:<10.4f}"
    )

    ppl_improvement = ((base_ppl - ft_ppl) / base_ppl) * 100
    sim_improvement = (
        ((avg_ft_sim - avg_base_sim) / avg_base_sim) * 100 if avg_base_sim > 0 else 0
    )

    print(
        f"\nPerplexity improvement: {ppl_improvement:.1f}% {'(lower is better)' if ppl_improvement > 0 else '(degradation)'}"
    )
    print(
        f"Similarity improvement: {sim_improvement:.1f}% {'(higher is better)' if sim_improvement > 0 else '(degradation)'}"
    )

    # Show qualitative examples
    print("\n" + "=" * 60)
    print("QUALITATIVE COMPARISON (First 3 examples)")
    print("=" * 60)

    for i in range(min(3, len(prompts))):
        print(f"\n--- Example {i + 1} ---")
        print(f"Task: {dataset[i].get('task_category', 'unknown')}")
        print(f"Crate: {dataset[i].get('crate_name', 'unknown')}")
        print(f"\nPrompt (truncated): {prompts[i][:200]}...")
        print(f"\nExpected (truncated): {expected_outputs[i][:200]}...")
        print(f"\nBase output (truncated): {base_outputs[i][:200]}...")
        print(f"\nFine-tuned output (truncated): {ft_outputs[i][:200]}...")
        print(f"\nBase similarity: {base_similarities[i]:.4f}")
        print(f"Fine-tuned similarity: {ft_similarities[i]:.4f}")

    # Save results
    results = {
        "base_model": BASE_MODEL,
        "finetuned_model": FINETUNED_MODEL,
        "dataset": DATASET,
        "num_samples": eval_samples,
        "metrics": {
            "base_perplexity": base_ppl,
            "finetuned_perplexity": ft_ppl,
            "base_similarity": avg_base_sim,
            "finetuned_similarity": avg_ft_sim,
            "perplexity_improvement_pct": ppl_improvement,
            "similarity_improvement_pct": sim_improvement,
        },
    }

    print("\n" + "=" * 60)
    print("RESULTS JSON")
    print("=" * 60)
    print(json.dumps(results, indent=2))

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
