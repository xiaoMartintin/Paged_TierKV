# Paged-TierKV: Block-Level Tiered KV Cache via Mixed-Precision and Sparsification

**Team Members:** Jingwu Wang, Martin Kang
## 1. Introduction
The Key-Value (KV) cache is a major memory bottleneck in LLM serving. Current compression methods—uniform quantization and token sparsification (dropping)—force a harsh trade-off between memory savings, generation quality, and context retention. Based on the highly skewed nature of LLM attention scores, we propose **Paged-TierKV**, a block-level tiered memory manager that dynamically classifies KV blocks into three tiers: **Hot** (high precision), **Warm** (quantized), and **Cold** (evicted/sparsified), maximizing memory reduction while preserving critical context.

## 2. Problem
**Research Question:**  
Can a block-level, tiered KV cache policy combining mixed-precision quantization and selective sparsification outperform single-method compression in memory efficiency and quality?

**Problem Definition:**  
Dynamically changing KV cache precision or dropping tokens introduces severe memory fragmentation and state-management overhead. The core systems-engineering challenge is designing a lightweight, non-fragmenting memory allocator that seamlessly migrates blocks across different states without severe synchronization overhead or complex custom math kernels.

## 3. Status Quo
Existing works typically tackle KV cache optimization in isolation:
- **Quantization:** Compresses uniformly but blindly degrades critical "attention sinks."
- **Sparsification:** Drops tokens to save space but destroys historical context.
- **Paged Attention:** Eliminates memory fragmentation using block-level allocation but operates uniformly in a single precision.

**Our Improvement:**  
Paged-TierKV bridges these paradigms by implementing a hybrid 3-tier policy over a block table. It mitigates accuracy loss by keeping hot blocks precise and mitigates context loss by quantizing warm blocks.

## 4. High-Level Implementation

**Planned Datasets:**  
We will use standard language modeling datasets (e.g., WikiText) for perplexity evaluation, and long-context benchmarks (e.g., subsets of LongBench) to evaluate context retention and accuracy. Synthetic prompts will be used for memory profiling.

**Implementation Plan:**
- **C++ Block Manager:**  
  Use C++ and PyBind11 to build a custom memory router. It maintains a BlockTable and manages physical block pointers across different memory pools (e.g., high-precision vs. compressed) to prevent fragmentation.

- **Model Integration (Python):**  
  Integrate the backend into a lightweight, hackable LLM architecture. A Python-based policy engine periodically evaluates block-level attention scores. Instead of writing custom low-level math kernels, leverage PyTorch’s native tensor operations for quantization, triggering the C++ API to update block routing and states dynamically.

- **Evaluation:**  
  Benchmark Paged-TierKV against:
  - Standard uncompressed baselines
  - Single-method compression baselines (e.g., uniform quantization, token eviction)

  Metrics include:
  - Peak Memory Usage (MB)
  - Throughput (tokens/s)
  - Accuracy