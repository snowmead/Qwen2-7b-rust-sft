# Dataset Analysis: Fortytwo-Network/Strandset-Rust-v1

Analysis of token lengths to inform training configuration choices.

## Dataset Overview

- **Source**: `Fortytwo-Network/Strandset-Rust-v1`
- **Total Examples**: 191,008
- **Sample Size for Analysis**: 5,000 examples (random seed 42)
- **Tokenizer**: Qwen/Qwen3-0.6B

## Token Length Statistics

| Statistic | Value |
|-----------|-------|
| Minimum | 49 tokens |
| Maximum | 18,736 tokens |
| Mean | 361 tokens |
| Median | 266 tokens |
| Std Dev | 417 tokens |

## Length Distribution

| Max Length | Examples Fitting | Percentage | Truncated |
|------------|------------------|------------|-----------|
| 256 | 2,413 | 48.3% | 51.7% |
| 512 | 3,951 | 79.0% | 21.0% |
| 1024 | 4,818 | **96.4%** | 3.6% |
| 1536 | 4,945 | 98.9% | 1.1% |
| 2048 | 4,978 | **99.6%** | 0.4% |
| 3000 | 4,995 | 99.9% | 0.1% |
| 4096 | 4,996 | 99.9% | 0.1% |
| 8192 | 4,999 | 100.0% | 0.0% |

## Key Findings

### 1. Most Examples Are Short
- **Median: 266 tokens** - half of all examples fit in 256 tokens
- **Mean: 361 tokens** - average is relatively short
- The distribution is right-skewed with a long tail of longer examples

### 2. 2048 Token Limit Is Appropriate
- **99.6% of examples fit within 2048 tokens**
- Only 0.4% of examples are truncated at this limit
- Current default `--max-length 2048` captures nearly all training data

### 3. 1024 Tokens Captures 96.4%
- Reducing to `--max-length 1024` would:
  - Lose only 3.6% of data to truncation
  - Significantly reduce memory usage (~4x less for attention)
  - Allow larger batch sizes on memory-constrained GPUs
  - Enable training on GPUs with less VRAM (e.g., A10G)

### 4. Outliers Exist
- Maximum observed: 18,736 tokens
- These extreme outliers likely contain very long code files
- They will always be truncated regardless of reasonable max_length settings

## Recommendations

### For Standard Training (A100)
Use `--max-length 2048` (default)
- Captures 99.6% of examples fully
- A100 has sufficient memory

### For Memory-Constrained GPUs (A10G, T4)
Use `--max-length 1024`
- Captures 96.4% of examples fully
- Reduces memory usage by ~4x for attention computation
- Enables training with larger batch sizes

### For Fast Iteration
Use `--max-length 512`
- Captures 79% of examples fully
- Fastest training iteration
- Good for quick hyperparameter searches

## Impact on Training

When examples are truncated:
1. The model only sees the beginning of the conversation
2. For code tasks, this may cut off the actual code solution
3. However, with only 0.4% truncated at 2048, impact is minimal

## Visualization

```
Token Length Distribution (5000 samples)
----------------------------------------
0-256:    ████████████████████████ 48.3%
256-512:  ███████████████ 30.7%
512-1024: █████████ 17.3%
1024-1536: █ 2.5%
1536-2048: ▏ 0.7%
2048+:     ▏ 0.4%
```

---

*Analysis conducted December 2024 using Qwen3-0.6B tokenizer.*
