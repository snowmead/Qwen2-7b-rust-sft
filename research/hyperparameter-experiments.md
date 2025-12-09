# Fine-Tuning a Model to Excel at Rust: A Research Journey

This document captures the end-to-end research process for fine-tuning a language model on Rust code, including experiment design, hyperparameter exploration, and the reasoning behind each decision.

## Project Goal

Train a language model to be proficient at Rust programming tasks using the **Fortytwo-Network/Strandset-Rust-v1** dataset. The model should handle diverse Rust-related tasks including code completion, bug detection, test generation, documentation, and more.

## The Dataset: Strandset-Rust-v1

The dataset contains examples across 14 different Rust programming task types:

- Code completion
- Bug detection
- Test generation
- Documentation generation
- Code explanation
- Refactoring suggestions
- Error handling
- Type inference
- Macro expansion
- Trait implementation
- Lifetime annotation
- Performance optimization
- API design
- Unsafe code review

This diversity requires the model to develop a broad understanding of Rust idioms, patterns, and best practices.

## Training Approach: LoRA + SFT

### Why LoRA (Low-Rank Adaptation)?

Instead of fine-tuning all model parameters (billions of weights), LoRA adds small trainable "adapter" matrices to specific layers. Benefits:

1. **Memory efficient**: Only ~0.1-1% of parameters are trainable
2. **Fast training**: Fewer gradients to compute
3. **Modular**: Can swap adapters without touching base model
4. **Prevents catastrophic forgetting**: Base model knowledge preserved

### Why SFT (Supervised Fine-Tuning)?

SFT trains the model to follow a specific format—in our case, converting Rust programming tasks into chat-style messages with system prompts, user queries, and expected responses.

## Understanding the Hyperparameters

Before running experiments, it's crucial to understand what each hyperparameter controls:

### Learning Rate (lr)

The learning rate controls how much the model's weights change with each update.

- **Too high**: Model overshoots optimal values, training becomes unstable
- **Too low**: Training progresses slowly, may get stuck in local minima
- **Typical range for LoRA**: 1e-5 to 5e-5

### Warmup Ratio

Training doesn't start at full learning rate. Instead, it gradually "warms up" from near-zero to the target LR.

- **Purpose**: Prevents early instability when model hasn't seen much data
- **How it works**: If warmup_ratio=0.1 with 1000 steps, LR linearly increases from ~0 to target over first 100 steps
- **Higher warmup**: More conservative start, better for aggressive LRs

### Learning Rate Scheduler

After warmup, the scheduler controls how LR changes over remaining training:

- **Cosine**: Smooth decay following cosine curve—gentle decline, then faster near end
- **Linear**: Steady decrease from max to zero
- **Constant**: No decay after warmup

Cosine is popular because it allows the model to make large improvements early, then fine-tune with smaller updates.

### LoRA Rank (r)

The rank determines the "capacity" of the adapter—how much new information it can learn.

- **Low rank (8-16)**: Efficient, good for small adaptations
- **High rank (32-64)**: More capacity, better for large datasets or complex tasks
- **Trade-off**: Higher rank = more parameters = more memory + compute

### LoRA Alpha

A scaling factor applied to the adapter's contribution. The effective scaling is `alpha / r`.

- **Convention**: Often set to 2× the rank (e.g., r=16, alpha=32)
- **Higher alpha**: Adapter has stronger influence on output
- **Lower alpha**: More conservative, relies more on base model

### LoRA Dropout

Dropout randomly "turns off" a fraction of adapter neurons during training.

- **Purpose**: Prevents overfitting by forcing redundancy
- **How it works**: Each forward pass, random neurons output zero
- **Typical values**: 0.05-0.1
- **Higher dropout**: More regularization, helps with small datasets

### Batch Size & Gradient Accumulation

These together determine the "effective batch size"—how many examples contribute to each weight update.

- **Batch size**: Examples processed in parallel (limited by GPU memory)
- **Gradient accumulation**: How many batches of gradients to sum before updating
- **Effective batch = batch_size × gradient_accumulation_steps**

Example: batch_size=4, grad_accum=4 → effective batch of 16

**Why it matters:**

- **Large batches**: Stable gradients, efficient GPU use, fewer updates per epoch
- **Small batches**: Noisier gradients (can help generalization), more updates per epoch

### Steps vs Epochs

- **Epoch**: One complete pass through the training data
- **Step**: One weight update (after accumulating gradients)

With batch_size=4 and grad_accum=4 on 9000 training examples:

- Examples per step: 16
- Steps per epoch: 9000 / 16 = 562 steps

### The Implicit Regularization of Small Batches

This is a key insight that informed our experiments. When using smaller batches:

1. Each gradient estimate is based on fewer examples
2. This introduces "noise" into the gradient direction
3. The noise prevents the model from memorizing specific examples
4. Result: Better generalization to unseen data

**Analogy**: Imagine navigating a valley in fog. With large batches (clear view), you march straight toward the deepest point—but might get stuck in a narrow pit. With small batches (foggy), you wander a bit, but you're more likely to find a broad, flat valley floor—which represents a solution that works well on diverse inputs.

## Experiment Design

### Constraints

- **Examples**: 10,000 (validation scale before committing to full dataset)
- **Seed**: 42 (reproducibility across all runs)
- **Hardware**: A100-large (consistent compute environment)
- **Monitoring**: Trackio for real-time metrics comparison
- **Push to Hub**: Disabled (quick iteration)

### The 5 Experiments

#### Run 1: Baseline

**Purpose**: Establish reference metrics with sensible defaults.

| Parameter    | Value  |
| ------------ | ------ |
| epochs       | 1      |
| lr           | 2e-5   |
| batch_size   | 4      |
| grad_accum   | 4      |
| warmup_ratio | 0.1    |
| lr_scheduler | cosine |
| lora_r       | 16     |
| lora_alpha   | 32     |
| lora_dropout | 0.05   |

**Effective batch size**: 16
**Run name**: `exp-baseline-10k`

---

#### Run 2: Conservative Learning Rate

**Hypothesis**: Lower LR with more epochs allows gradual, stable convergence without overshooting.

| Parameter    | Value | Change from Baseline |
| ------------ | ----- | -------------------- |
| epochs       | 2     | +1                   |
| lr           | 1e-5  | ÷2                   |
| batch_size   | 4     | —                    |
| grad_accum   | 4     | —                    |
| warmup_ratio | 0.1   | —                    |
| lora_r       | 16    | —                    |
| lora_alpha   | 32    | —                    |
| lora_dropout | 0.05  | —                    |

**Run name**: `exp-conservative-lr-10k`

**Reasoning**:

- Half the learning rate = smaller weight updates = less risk of overshooting
- Double epochs compensates for slower progress
- Should produce smoother loss curves

---

#### Run 3: High Capacity

**Hypothesis**: More adapter capacity captures complex code patterns; higher dropout prevents overfitting.

| Parameter    | Value | Change from Baseline |
| ------------ | ----- | -------------------- |
| epochs       | 1     | —                    |
| lr           | 2e-5  | —                    |
| batch_size   | 4     | —                    |
| grad_accum   | 4     | —                    |
| warmup_ratio | 0.1   | —                    |
| lora_r       | 32    | ×2                   |
| lora_alpha   | 64    | ×2                   |
| lora_dropout | 0.1   | ×2                   |

**Run name**: `exp-high-capacity-10k`

**Reasoning**:

- Higher rank = adapter can learn more nuanced patterns
- Maintains alpha/r ratio of 2×
- Double dropout counteracts overfitting risk from added capacity

---

#### Run 4: Aggressive Learning Rate

**Hypothesis**: Higher LR achieves faster convergence; extended warmup prevents early instability.

| Parameter    | Value | Change from Baseline |
| ------------ | ----- | -------------------- |
| epochs       | 1     | —                    |
| lr           | 5e-5  | ×2.5                 |
| batch_size   | 4     | —                    |
| grad_accum   | 4     | —                    |
| warmup_ratio | 0.2   | ×2                   |
| lora_r       | 16    | —                    |
| lora_alpha   | 32    | —                    |
| lora_dropout | 0.05  | —                    |

**Run name**: `exp-aggressive-lr-10k`

**Reasoning**:

- Higher LR = bigger steps = faster progress
- 20% warmup (vs 10%) gives model time to stabilize before full LR
- Cosine decay then smoothly reduces to prevent late-training instability

---

#### Run 5: Small Batch

**Hypothesis**: Smaller effective batch size provides implicit regularization through gradient noise; more updates per epoch.

| Parameter    | Value | Change from Baseline |
| ------------ | ----- | -------------------- |
| epochs       | 1     | —                    |
| lr           | 2e-5  | —                    |
| batch_size   | 2     | ÷2                   |
| grad_accum   | 2     | ÷2                   |
| warmup_ratio | 0.1   | —                    |
| lora_r       | 16    | —                    |
| lora_alpha   | 32    | —                    |
| lora_dropout | 0.1   | ×2                   |

**Effective batch size**: 4 (vs baseline 16)
**Run name**: `exp-small-batch-10k`

**Reasoning**:

- Smaller batches = noisier gradients = implicit regularization
- 4× more parameter updates per epoch (2250 steps vs 562)
- Higher dropout adds explicit regularization
- Research shows small batches often generalize better

## Results

All experiments ran on A100-large hardware with 10,000 examples (9,000 train / 1,000 eval).

### Final Metrics

| Experiment        | Eval Loss | Token Accuracy | Steps | Training Time |
| ----------------- | --------- | -------------- | ----- | ------------- |
| **Small Batch**   | **0.766** | 82.1%          | 2250  | ~45 min       |
| **Aggressive LR** | **0.767** | 82.0%          | 562   | ~15 min       |
| High Capacity     | 0.786     | 81.2%          | 562   | ~18 min       |
| Baseline          | 0.802     | 80.5%          | 562   | ~15 min       |
| Conservative LR   | 0.816     | 79.8%          | 1125  | ~30 min       |

### Key Observations

#### 1. Small Batch Wins on Quality

The small-batch configuration achieved the best eval loss (0.766), validating the hypothesis about implicit regularization. With 4× more gradient updates per epoch, the model had more opportunities to learn while the noisy gradients prevented overfitting.

#### 2. Aggressive LR: Best Time-to-Quality Ratio

Nearly identical eval loss (0.767) but in 4× fewer steps. For rapid iteration, this is the practical choice. The extended warmup (20%) was crucial—it allowed the high learning rate to work without early instability.

#### 3. High Capacity Didn't Help (At This Scale)

Doubling the LoRA rank from 16 to 32 didn't improve results on 10k examples. The additional capacity may become valuable at 100k+ examples where there's more pattern diversity to capture.

#### 4. Conservative LR Underperformed

Despite the intuition that "slower = more careful = better," the conservative approach produced the worst eval loss. Two epochs weren't enough to compensate for the halved learning rate, and the model may have needed even more training time.

#### 5. All Runs Showed Early Plateau

Token accuracy jumped to ~80% within the first 50-100 steps, then slowly improved. This suggests the base model (Qwen) already has strong code understanding—fine-tuning is primarily teaching it the task format and Rust-specific patterns.

### Loss Curves Analysis

All experiments showed healthy training dynamics:

- Training loss decreased smoothly
- Eval loss tracked training loss (no overfitting)
- No sudden spikes or instability

The small-batch run showed slightly more variance in the loss curve (expected from noisier gradients) but achieved the best final result.

## Scaling Considerations

### When to Use Each Configuration

| Scenario                               | Recommended Preset |
| -------------------------------------- | ------------------ |
| **Quick iteration / prototyping**      | aggressive-lr      |
| **Production training / best quality** | small-batch        |
| **Large datasets (100k+)**             | high-capacity      |

### Combining Approaches?

A natural question: can we combine small batches with aggressive LR?

**Answer**: Proceed with caution. Both techniques increase gradient variance:

- Small batches → noisy gradient estimates
- High LR → large weight updates

Combined, this could cause training instability. If attempting this combination:

- Use even longer warmup (25-30%)
- Consider gradient clipping
- Monitor loss curves carefully for spikes

### Scaling to 100k+ Examples

For larger datasets:

1. **Start with high-capacity preset** (r=32)—more data justifies more parameters
2. **Consider increasing rank further** (r=64) if validation loss plateaus
3. **May need multiple epochs** to fully utilize the data
4. **Small-batch benefits may diminish**—with more data, gradient estimates become reliable even with larger batches

## Final Presets

Based on experiment results, we created three reusable presets:

```python
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
```

Usage:

```bash
uv run qwen2-train --preset small-batch -n 10000
uv run qwen2-train --preset aggressive-lr -n 5000 --push-to-hub
```

Individual parameters can still override preset values:

```bash
uv run qwen2-train --preset small-batch --lr 3e-5  # Override just LR
```

## Validation Run: Qwen3-4B with Small-Batch Preset

To validate the presets and test a smaller model, we ran:

```bash
uv run qwen2-train --preset small-batch -n 5000 --run-name exp-small-batch-5k
```

**Model**: Qwen3-4B-Instruct-2507 (4B parameters vs 7B)

### Results

| Metric         | Value      |
| -------------- | ---------- |
| Eval Loss      | 0.779      |
| Token Accuracy | 80.9%      |
| Training Steps | 1125       |
| Hardware       | A100-large |

Despite being a smaller model, Qwen3-4B achieved competitive results:

- Only 0.013 higher eval loss than Qwen2-7B with same preset
- 80.9% token accuracy (vs 82.1% for 7B on 10k examples)
- Faster inference at deployment time

This suggests the small-batch preset generalizes well across model sizes.

## Conclusions

1. **Implicit regularization matters**: Smaller batch sizes improved generalization, outperforming configurations with more compute per step.

2. **Learning rate vs. time trade-off**: Aggressive LR achieves near-optimal results in 25% of the training time—valuable for iteration.

3. **Adapter capacity scales with data**: r=16 was sufficient for 10k examples; r=32 may help at 100k+.

4. **Conservative isn't always better**: Lower learning rates require proportionally more training time to compensate.

5. **Base model quality matters**: The rapid early improvement suggests fine-tuning is augmenting already-capable models, not teaching from scratch.

## Next Steps

1. **Scale to full dataset** using small-batch preset
2. **Evaluate on held-out Rust tasks** not in training data
3. **Compare with code-specific models** (CodeLlama, StarCoder)
4. **Test inference quality** on real Rust programming scenarios
5. **Consider merging adapters** for deployment without LoRA overhead

---

_Research conducted December 2024. Training infrastructure: HuggingFace Jobs. Monitoring: Trackio._
