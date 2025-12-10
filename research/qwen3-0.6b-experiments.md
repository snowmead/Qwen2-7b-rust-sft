# Qwen3-0.6B Fine-Tuning Experiments

This document captures the experiments conducted to fine-tune the Qwen3-0.6B model on the Strandset-Rust-v1 dataset, including observations and conclusions.

## Objective

Determine how far we can push eval loss on a small (0.6B parameter) model through hyperparameter tuning and training strategies.

## Model & Dataset

- **Base Model**: Qwen/Qwen3-0.6B
- **Dataset**: Fortytwo-Network/Strandset-Rust-v1 (191,008 examples)
- **Task**: Supervised fine-tuning on 14 Rust programming task types
- **Hardware**: A100-large (80GB)

## Experiments

### Experiment 1: Initial Baseline (5K Examples)

**Run Name**: `exp-qwen3-0.6b-5k`

| Parameter | Value |
|-----------|-------|
| Examples | 5,000 |
| Epochs | 1 |
| Batch Size | 2 |
| Grad Accum | 2 |
| Learning Rate | 2e-5 |
| LoRA r/α | 16/32 |
| LoRA Dropout | 0.1 |
| Max Length | 2048 |
| LR Scheduler | cosine |

**Results**:

| Metric | Value |
|--------|-------|
| Eval Loss | **1.088** |
| Train Loss | ~1.0 |

**Observations**:
- Model learns quickly in early steps
- Eval loss plateaus around 1.0
- Small dataset limits generalization

---

### Experiment 2: Scaled Data + Higher Capacity (20K Examples)

**Run Name**: `exp-qwen3-0.6b-20k-r32-1024-a100`

| Parameter | Value | Change from Exp 1 |
|-----------|-------|-------------------|
| Examples | 20,000 | 4x more |
| Epochs | 1 | — |
| Batch Size | 2 | — |
| Grad Accum | 2 | — |
| Learning Rate | 2e-5 | — |
| LoRA r/α | 32/64 | 2x capacity |
| LoRA Dropout | 0.1 | — |
| Max Length | 1024 | Reduced (96.4% coverage) |
| LR Scheduler | cosine | — |

**Results**:

| Metric | Value |
|--------|-------|
| Eval Loss | **1.004** |
| Train Loss | ~0.95 |
| Improvement | -0.084 from baseline |

**Observations**:
- 4x more data yielded only 0.084 improvement
- Diminishing returns suggest capacity limitation
- Higher LoRA rank (32 vs 16) didn't dramatically help
- Reducing max_length to 1024 was safe (captures 96.4% of examples)
- Model appears to be hitting a floor around 1.0

---

### Experiment 3: Extended Training + LR Restarts (20K, 2 Epochs)

**Run Name**: `exp-qwen3-0.6b-20k-2ep-restarts`

| Parameter | Value | Change from Exp 2 |
|-----------|-------|-------------------|
| Examples | 20,000 | — |
| Epochs | 2 | 2x training |
| Batch Size | 2 | — |
| Grad Accum | 2 | — |
| Learning Rate | 2e-5 | — |
| LoRA r/α | 32/64 | — |
| LoRA Dropout | 0.1 | — |
| Max Length | 1024 | — |
| LR Scheduler | cosine_with_restarts | Changed |
| Warmup Ratio | 0.1 | — |

**Results**:

| Metric | Value |
|--------|-------|
| Eval Loss | **0.963** |
| Train Loss | 0.943 |
| Best Eval Step | 866 / 891 |
| Improvement | -0.041 from Exp 2, -0.125 from baseline |

**Observations**:
- **Broke below 1.0** for the first time
- cosine_with_restarts helped escape local minima
- Second epoch provided meaningful improvement
- LR "reheating" at epoch boundary gave fresh momentum
- Train/eval gap is small (0.02) - no overfitting

---

## Summary Comparison

| Experiment | Examples | Epochs | Scheduler | Eval Loss | Δ from Baseline |
|------------|----------|--------|-----------|-----------|-----------------|
| Exp 1 (baseline) | 5,000 | 1 | cosine | 1.088 | — |
| Exp 2 (scaled) | 20,000 | 1 | cosine | 1.004 | -0.084 |
| **Exp 3 (restarts)** | 20,000 | 2 | cosine_with_restarts | **0.963** | **-0.125** |

## Key Insights

### 1. Data Scaling Has Diminishing Returns

Going from 5K → 20K examples (4x) only improved eval loss by 0.084. This suggests the 0.6B model has limited capacity to absorb more data without architectural changes.

### 2. Extended Training + LR Restarts Works

The combination of 2 epochs with cosine_with_restarts was the most effective intervention:
- 2 epochs = more gradient updates
- Restarts = LR "reheats" after first epoch, allowing escape from local minima

### 3. LoRA Capacity Matters Less Than Expected

Doubling LoRA rank (16→32) didn't provide dramatic gains. The bottleneck is likely the base model's 0.6B parameters, not the adapter capacity.

### 4. The ~1.0 Loss Floor

There appears to be a natural floor around 1.0 eval loss for this model size on this dataset. Breaking below 1.0 required:
- Extended training (2 epochs)
- LR scheduler that prevents premature convergence

### 5. Max Length Reduction is Safe

Reducing max_length from 2048 → 1024 had no negative impact. The dataset analysis showed 96.4% of examples fit within 1024 tokens.

## Training Dynamics

All three experiments showed similar patterns:
1. **Rapid early learning**: Loss drops quickly in first 50-100 steps
2. **Gradual refinement**: Slow improvement through remaining training
3. **Stable gradients**: No spikes or instability (grad_norm typically 1.5-3.0)
4. **No overfitting**: Train/eval gap stays small (~0.02-0.05)

## Recommendations for Further Improvement

Based on these experiments, potential next steps:

1. **Curriculum Learning**: Train on easier tasks first (understanding → annotation → generation)
2. **More Epochs**: Try 3-4 epochs with restarts to see if trend continues
3. **Larger Model**: Test Qwen3-4B to see if capacity is the true bottleneck
4. **Data Quality**: Filter for highest-quality examples rather than more data
5. **Knowledge Distillation**: Train 0.6B model to match larger teacher model

## Conclusion

The Qwen3-0.6B model can achieve **0.963 eval loss** on the Strandset-Rust-v1 dataset with:
- 20,000 examples
- 2 epochs
- cosine_with_restarts LR scheduler
- LoRA r=32, α=64

Further improvements likely require either more training time, curriculum-based approaches, or a larger base model.

---

*Experiments conducted December 2024. Training infrastructure: HuggingFace Jobs (A100-large). Monitoring: Trackio.*
