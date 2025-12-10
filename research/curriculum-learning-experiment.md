# Curriculum Learning Experiment: Task-Based Progressive Training

This document captures the hypothesis, methodology, and results for training a small language model (Qwen3-0.6B) using task-based curriculum learning on the Strandset-Rust-v1 dataset.

## Motivation

After extensive hyperparameter tuning, the Qwen3-0.6B model plateaued around **1.0 eval loss** on the full mixed-task dataset. Despite trying:
- Different batch sizes (small-batch preset)
- Higher LoRA capacity (r=32, α=64)
- Extended training (2 epochs)
- LR scheduler variations (cosine_with_restarts)

The model struggled to break below 1.0 eval loss. This suggests a **capacity limitation** rather than a hyperparameter issue.

## Hypothesis

**Task-based curriculum learning will help the small model learn more effectively by:**

1. Building foundational skills first (understanding code)
2. Progressively adding complexity (describing → generating code)
3. Allowing knowledge transfer between related task types

Just as humans learn to read before they write, models may benefit from learning to analyze code before generating it.

## Curriculum Design

### Phase 1: Understanding (Read & Analyze)

Train the model to comprehend code structure and identify patterns.

| Task Type | Description | Count |
|-----------|-------------|-------|
| code_explanation | Explain what code does | 26,687 |
| code_summarization | Summarize code functionality | - |
| code_review | Review code for issues | 27,037 |
| bug_detection | Find bugs in code | 27,175 |

**Why first**: These tasks require reading and analyzing code without generating new code. The model learns syntax, patterns, and common idioms.

### Phase 2: Annotation (Describe & Label)

Train the model to describe code elements and suggest names.

| Task Type | Description | Count |
|-----------|-------------|-------|
| docstring_generation | Write documentation | 13,395 |
| comment_generation | Add inline comments | 13,387 |
| function_naming | Suggest function names | 13,442 |
| variable_naming | Suggest variable names | 13,423 |

**Why second**: These tasks bridge understanding and generation. The model must understand code to describe it, but outputs are constrained (names, short descriptions).

### Phase 3: Generation (Write & Transform)

Train the model to write and modify code.

| Task Type | Description | Count |
|-----------|-------------|-------|
| code_generation | Generate new code | 13,372 |
| code_completion | Complete partial code | 13,396 |
| code_refactoring | Improve code structure | 13,495 |
| code_optimization | Optimize performance | 13,386 |
| test_generation | Write unit tests | 13,542 |
| code_search | Find relevant code | - |
| api_usage_prediction | Predict API usage | - |

**Why third**: These are the most complex tasks. By this phase, the model has learned to read, analyze, and describe code—making generation more tractable.

## Implementation

### Checkpoint-Based Sequential Training

Each phase trains independently and saves a checkpoint to HuggingFace Hub:

```bash
# Phase 1: Understanding
hf jobs uv run ... train.py \
  --curriculum-phase understanding \
  --output-repo snowmead/qwen3-0.6b-rust-phase1 \
  ...

# Phase 2: Annotation (resume from Phase 1)
hf jobs uv run ... train.py \
  --curriculum-phase annotation \
  --resume-from snowmead/qwen3-0.6b-rust-phase1 \
  --output-repo snowmead/qwen3-0.6b-rust-phase2 \
  ...

# Phase 3: Generation (resume from Phase 2)
hf jobs uv run ... train.py \
  --curriculum-phase generation \
  --resume-from snowmead/qwen3-0.6b-rust-phase2 \
  --output-repo snowmead/qwen3-0.6b-rust-phase3 \
  ...
```

### CLI Arguments Added

```python
--curriculum-phase  # Filter to specific phase: understanding, annotation, generation
--resume-from       # HuggingFace Hub repo to resume from
--output-repo       # HuggingFace Hub repo to push checkpoint
```

## Baseline Comparisons

Before curriculum training, we established these baselines on Qwen3-0.6B:

| Run Name | Examples | Epochs | Eval Loss | Notes |
|----------|----------|--------|-----------|-------|
| exp-qwen3-0.6b-5k | 5,000 | 1 | 1.088 | Initial baseline |
| exp-qwen3-0.6b-20k-r32-1024-a100 | 20,000 | 1 | 1.004 | 4x data, ~0.08 improvement |
| exp-qwen3-0.6b-20k-2ep-restarts | 20,000 | 2 | 0.972* | 2 epochs + restarts |

*Training in progress

## Experiment Plan

### Phase 1 Training

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen3-0.6B |
| Dataset | understanding tasks only |
| Examples | All available (~80k) or subset |
| Epochs | 1 |
| Batch Size | 2 |
| Grad Accum | 2 |
| Learning Rate | 2e-5 |
| LoRA r/α | 32/64 |
| Max Length | 1024 |

**Expected outcome**: Lower eval loss on understanding tasks, improved code comprehension.

### Phase 2 Training

Same hyperparameters, but:
- Resume from Phase 1 checkpoint
- Train on annotation tasks only

**Expected outcome**: Model leverages understanding skills to improve annotation quality.

### Phase 3 Training

Same hyperparameters, but:
- Resume from Phase 2 checkpoint
- Train on generation tasks only

**Expected outcome**: Generation quality improved by foundation built in earlier phases.

## Success Criteria

1. **Per-phase eval loss**: Each phase should show healthy convergence on its task subset
2. **Final eval loss**: Phase 3 model should achieve lower loss on full dataset than baseline (~1.0)
3. **Qualitative improvement**: Model should produce more coherent code generations

## Results

*To be filled after experiments complete*

### Phase 1 Results

| Metric | Value |
|--------|-------|
| Eval Loss | TBD |
| Training Steps | TBD |
| Time | TBD |

### Phase 2 Results

| Metric | Value |
|--------|-------|
| Eval Loss | TBD |
| Training Steps | TBD |
| Time | TBD |

### Phase 3 Results

| Metric | Value |
|--------|-------|
| Eval Loss | TBD |
| Training Steps | TBD |
| Time | TBD |

### Comparison with Baseline

| Model | Eval Loss (Full) | Training Approach |
|-------|------------------|-------------------|
| Baseline (mixed) | ~1.0 | All tasks simultaneously |
| Curriculum (phase3) | TBD | Progressive phases |

## Conclusions

*To be written after experiments*

---

*Research document created December 2024. Training infrastructure: HuggingFace Jobs. Monitoring: Trackio.*
