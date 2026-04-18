# Final Evaluation Results

## Environment / Notes
- Codebase is treated as frozen for final evaluation.
- Default TierKV config: `policy_interval=64`, `policy_on_new_block=False`, `policy_mode=hot_warm`.
- Pool layout: compact state-separated HOT/WARM/COLD storage.
- Main result path: TinyLlama staged high-KV-pressure evaluation.
- 7B extension path: Vicuna/LLaMA-compatible MHA experiment, reported separately as supporting evidence.
- Log directory: `final_eval_logs/latest`.

## Commands Run
### Main final benchmark package
Log: `final_eval_logs/latest/main.log`
```bash
modal run main.py
```

### Tier-mode ablation
Log: `final_eval_logs/latest/tier_mode.log`
```bash
modal run main.py --tier-mode-ablation
```

### Quality sanity
Log: `final_eval_logs/latest/quality_sanity.log`
```bash
modal run main.py --quality-sanity
```

### Stage A scaling
Log: `final_eval_logs/latest/stage_a.log`
```bash
TIERKV_STAGED_COUNTS=32,64,96,128,160 TIERKV_STAGED_CONTEXT_TOKENS=1900 TIERKV_STAGED_DECODE_STEPS=16 modal run main.py --staged-scaling
```

### Stage B scaling
Log: `final_eval_logs/latest/stage_b.log`
```bash
TIERKV_STAGED_COUNTS=192,224,256,320 TIERKV_STAGED_CONTEXT_TOKENS=1900 TIERKV_STAGED_DECODE_STEPS=16 modal run main.py --staged-scaling
```

### Boundary run
Log: `final_eval_logs/latest/boundary.log`
```bash
TIERKV_STAGED_COUNTS=384,448,512 TIERKV_STAGED_CONTEXT_TOKENS=1900 TIERKV_STAGED_DECODE_STEPS=16 modal run main.py --staged-scaling
```

### 7B extension scaling at 1024 tokens
Log: `final_eval_logs/latest/extension_1024.log`
```bash
TIERKV_EXTENSION_MODEL_ID=${TIERKV_EXTENSION_MODEL_ID:-lmsys/vicuna-7b-v1.5} TIERKV_EXTENSION_CONTEXT_TOKENS=1024 TIERKV_EXTENSION_COUNTS=1,2,4,6,8,12 TIERKV_EXTENSION_DECODE_STEPS=16 modal run main.py --extension-scaling
```

### 7B extension scaling at 1536 tokens
Log: `final_eval_logs/latest/extension_1536.log`
```bash
TIERKV_EXTENSION_MODEL_ID=${TIERKV_EXTENSION_MODEL_ID:-lmsys/vicuna-7b-v1.5} TIERKV_EXTENSION_CONTEXT_TOKENS=1536 TIERKV_EXTENSION_COUNTS=1,2,4,6,8 TIERKV_EXTENSION_DECODE_STEPS=16 modal run main.py --extension-scaling
```

## 1. Main TinyLlama Results

### 1. Clean Sanity Table
Source log: `final_eval_logs/latest/main.log`

| Configuration | Peak Memory (MB) | Qasper F1 | Avg Input Tokens | Tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 2238.56 | 11.17 | 1900 | 53.29 |
| Pure Quantization | 2220.81 | 10.75 | 1900 | 16.27 |
| Pure Sparsification | 2198.53 | 0.93 | 1900 | 32.94 |
| Paged-TierKV (budget=4) | 2220.46 | 10.55 | 1900 | 16.44 |
| Paged-TierKV (budget=8) | 2222.87 | 10.67 | 1900 | 16.75 |

### Extended table with memory-realism metrics

| Configuration | Peak Mem (MB) | Qasper F1 | Avg Tokens | tok/s | HOT/WARM/COLD | Logical Resident (MB) | Dense Eq KV (MB) | Physical KV (MB) | Logical Compression | Physical Compression | Pool Util |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 2238.56 | 11.17 | 1900 | 53.29 | 0/0/0 | 0.0 | 0.0 | 0.0 | 0.00x | 0.00x | 0.0% |
| Pure Quantization | 2220.81 | 10.75 | 1900 | 16.27 | 88/2574/0 | 22.6 | 41.5 | 23.7 | 1.83x | 1.75x | 96.0% |
| Pure Sparsification | 2198.53 | 0.93 | 1900 | 32.94 | 88/0/2574 | 1.3 | 41.5 | 1.4 | 32.73x | 29.99x | 100.0% |
| Paged-TierKV (budget=4) | 2220.46 | 10.55 | 1900 | 16.44 | 132/2530/0 | 23.0 | 41.5 | 23.5 | 1.81x | 1.77x | 98.5% |
| Paged-TierKV (budget=8) | 2222.87 | 10.67 | 1900 | 16.75 | 220/2442/0 | 23.6 | 41.5 | 24.8 | 1.76x | 1.68x | 95.5% |

| Context Tokens | Baseline tok/s | Pure Quant tok/s | Paged-TierKV tok/s | Paged vs Baseline |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 57.45 | 26.95 | 50.15 | 0.87x |
| 1024 | 57.89 | 26.79 | 49.22 | 0.85x |
| 1536 | 58.23 | 26.67 | 49.63 | 0.85x |
| 1900 | 55.17 | 18.16 | 46.51 | 0.84x |

### 2. Tier-Mode Ablation
Source log: `final_eval_logs/latest/tier_mode.log`

| Mode | Peak Memory (MB) | Qasper F1 | Qasper tok/s |
| --- | ---: | ---: | ---: |
| HOT-only | 2239.72 | 11.16 | 23.53 |
| HOT+WARM | 2222.87 | 10.67 | 12.77 |
| HOT+WARM+COLD | 2208.14 | 1.47 | 19.80 |
| Mode | Peak Mem (MB) | Qasper F1 | tok/s | HOT/WARM/COLD | Logical Resident (MB) | Dense Eq KV (MB) | Physical KV (MB) | Logical Compression | Physical Compression | Pool Util |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HOT-only | 2239.72 | 11.16 | 23.53 | 2662/0/0 | 41.5 | 41.5 | 41.9 | 1.00x | 0.99x | 99.2% |
| HOT+WARM | 2222.87 | 10.67 | 12.77 | 220/2442/0 | 23.6 | 41.5 | 24.8 | 1.76x | 1.68x | 95.5% |
| HOT+WARM+COLD | 2208.14 | 1.47 | 19.80 | 220/704/1738 | 9.2 | 41.5 | 9.4 | 4.52x | 4.40x | 98.5% |
| Context Tokens | HOT-only tok/s | HOT+WARM tok/s | HOT+WARM+COLD tok/s |
| ---: | ---: | ---: | ---: |
| 512 | 24.69 | 31.32 | 40.83 |
| 1024 | 24.88 | 35.73 | 39.65 |
| 1536 | 24.70 | 35.46 | 40.78 |
| 1900 | 15.88 | 31.24 | 40.38 |

### 3. Quality Sanity
Source log: `final_eval_logs/latest/quality_sanity.log`

| Configuration | Peak Memory (MB) | Qasper F1 | Avg Input Tokens | Tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 2238.56 | 11.17 | 1900 | 41.57 |
| Pure Quantization | 2220.81 | 10.65 | 1900 | 12.08 |
| Pure Sparsification | 2198.53 | 0.93 | 1900 | 25.05 |
| Paged-TierKV (budget=4) | 2220.46 | 10.55 | 1900 | 12.41 |
| Paged-TierKV (budget=8) | 2222.87 | 10.67 | 1900 | 12.59 |

### Extended table with memory-realism metrics

| Configuration | Peak Mem (MB) | Qasper F1 | Avg Tokens | tok/s | HOT/WARM/COLD | Logical Resident (MB) | Dense Eq KV (MB) | Physical KV (MB) | Logical Compression | Physical Compression | Pool Util |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 2238.56 | 11.17 | 1900 | 41.57 | 0/0/0 | 0.0 | 0.0 | 0.0 | 0.00x | 0.00x | 0.0% |
| Pure Quantization | 2220.81 | 10.65 | 1900 | 12.08 | 88/2574/0 | 22.6 | 41.5 | 23.7 | 1.83x | 1.75x | 96.0% |
| Pure Sparsification | 2198.53 | 0.93 | 1900 | 25.05 | 88/0/2574 | 1.3 | 41.5 | 1.4 | 32.73x | 29.99x | 100.0% |
| Paged-TierKV (budget=4) | 2220.46 | 10.55 | 1900 | 12.41 | 132/2530/0 | 23.0 | 41.5 | 23.5 | 1.81x | 1.77x | 98.5% |
| Paged-TierKV (budget=8) | 2222.87 | 10.67 | 1900 | 12.59 | 220/2442/0 | 23.6 | 41.5 | 24.8 | 1.76x | 1.68x | 95.5% |

| Context Tokens | Baseline tok/s | Pure Quant tok/s | Paged-TierKV tok/s | Paged vs Baseline |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 41.41 | 21.44 | 39.14 | 0.95x |
| 1024 | 44.85 | 21.54 | 39.65 | 0.88x |
| 1536 | 44.32 | 21.23 | 38.42 | 0.87x |
| 1900 | 44.00 | 14.49 | 37.98 | 0.86x |

_If this section duplicates the clean sanity output, the current CLI did not emit a distinct standalone quality-only table._

### 4. Stage A Scaling (32,64,96,128,160)
Source log: `final_eval_logs/latest/stage_a.log`

### Staged Throughput Scaling

| Requests | Baseline tok/s | TierKV tok/s | TierKV / Baseline |
| ---: | ---: | ---: | ---: |
| 32 | 47.06 | 41.21 | 0.88x |
| 64 | 46.86 | 41.38 | 0.88x |
| 96 | 47.51 | 41.05 | 0.86x |
| 128 | 47.27 | 37.19 | 0.79x |
| 160 | 47.41 | 38.14 | 0.80x |

### Staged Memory Scaling

| Requests | Baseline Peak MB | TierKV Peak MB | TierKV Logical Resident KV MB | TierKV Physical KV MB | TierKV Dense Eq KV MB | TierKV Logical Compression | TierKV Physical Compression |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 3439.77 | 2910.27 | 745.59 | 792.00 | 1317.94 | 1.77x | 1.66x |
| 64 | 4769.22 | 3729.65 | 1502.19 | 1598.00 | 2646.88 | 1.76x | 1.66x |
| 96 | 6109.68 | 4542.35 | 2269.78 | 2411.00 | 3986.81 | 1.76x | 1.65x |
| 128 | 7461.13 | 5363.82 | 3027.75 | 3224.00 | 5337.75 | 1.76x | 1.66x |
| 160 | 8823.58 | 6180.70 | 3796.72 | 4037.00 | 6699.69 | 1.76x | 1.66x |

### Staged Status

| Requests | Baseline status | TierKV status | Notes |
| ---: | --- | --- | --- |
| 32 | OK | OK | — |
| 64 | OK | OK | — |
| 96 | OK | OK | — |
| 128 | OK | OK | — |
| 160 | OK | OK | — |

### Staged Failure Boundary

| Method | Max Successful Requests | Largest Successful Total Context Tokens | Peak MB At Max Success | Logical Resident MB | Physical KV MB | Dense-Equivalent KV MB | Failure Request Count | Failure Type |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | 160 | 304000 | 8823.58 | 0.00 | 0.00 | 0.00 | n/a | none |
| Paged-TierKV | 160 | 304000 | 6180.70 | 3796.72 | 4037.00 | 6699.69 | n/a | none |

### Scaling Analysis

Common successful range: 32 to 160 requests.
Throughput slope: Baseline 0.0027 tok/s/request (0.7%), TierKV -0.0239 tok/s/request (-7.4%).
Peak-memory slope: Baseline 42.06 MB/request (156.5%), TierKV 25.55 MB/request (112.4%).
TierKV physical KV slope: 25.35 MB/request (409.7%).
Throughput scaling conclusion: TierKV does not yet degrade more slowly than baseline.
Memory scaling conclusion: TierKV peak memory grows more slowly than baseline.
Failure-boundary conclusion: Both methods reached the same maximum tested load.

### 5. Stage B Scaling (192,224,256,320)
Source log: `final_eval_logs/latest/stage_b.log`

### Staged Throughput Scaling

| Requests | Baseline tok/s | TierKV tok/s | TierKV / Baseline |
| ---: | ---: | ---: | ---: |
| 192 | 60.42 | 50.61 | 0.84x |
| 224 | 60.43 | 50.05 | 0.83x |
| 256 | 60.66 | 52.17 | 0.86x |
| 320 | 59.89 | 40.92 | 0.68x |

### Staged Memory Scaling

| Requests | Baseline Peak MB | TierKV Peak MB | TierKV Logical Resident KV MB | TierKV Physical KV MB | TierKV Dense Eq KV MB | TierKV Logical Compression | TierKV Physical Compression |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 192 | 10032.04 | 6928.96 | 4473.56 | 4752.00 | 7907.62 | 1.77x | 1.66x |
| 224 | 11416.49 | 7792.48 | 5285.16 | 5628.00 | 9291.56 | 1.76x | 1.65x |
| 256 | 12811.94 | 8633.45 | 6107.75 | 6476.00 | 10686.50 | 1.75x | 1.65x |
| 320 | 15536.85 | 10267.70 | 7563.19 | 8081.00 | 13410.38 | 1.77x | 1.66x |

### Staged Status

| Requests | Baseline status | TierKV status | Notes |
| ---: | --- | --- | --- |
| 192 | OK | OK | — |
| 224 | OK | OK | — |
| 256 | OK | OK | — |
| 320 | OK | OK | — |

### Staged Failure Boundary

| Method | Max Successful Requests | Largest Successful Total Context Tokens | Peak MB At Max Success | Logical Resident MB | Physical KV MB | Dense-Equivalent KV MB | Failure Request Count | Failure Type |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | 320 | 608000 | 15536.85 | 0.00 | 0.00 | 0.00 | n/a | none |
| Paged-TierKV | 320 | 608000 | 10267.70 | 7563.19 | 8081.00 | 13410.38 | n/a | none |

### Scaling Analysis

Common successful range: 192 to 320 requests.
Throughput slope: Baseline -0.0041 tok/s/request (-0.9%), TierKV -0.0757 tok/s/request (-19.2%).
Peak-memory slope: Baseline 43.01 MB/request (54.9%), TierKV 26.08 MB/request (48.2%).
TierKV physical KV slope: 26.01 MB/request (70.1%).
Throughput scaling conclusion: TierKV does not yet degrade more slowly than baseline.
Memory scaling conclusion: TierKV peak memory grows more slowly than baseline.
Failure-boundary conclusion: Both methods reached the same maximum tested load.

### 6. Boundary Run (384,448,512)
Source log: `final_eval_logs/latest/boundary.log`

### Staged Throughput Scaling

| Requests | Baseline tok/s | TierKV tok/s | TierKV / Baseline |
| ---: | ---: | ---: | ---: |
| 384 | 44.02 | 38.78 | 0.88x |
| 448 | 43.40 | 38.34 | 0.88x |
| 512 | — | 38.24 | — |

### Staged Memory Scaling

| Requests | Baseline Peak MB | TierKV Peak MB | TierKV Logical Resident KV MB | TierKV Physical KV MB | TierKV Dense Eq KV MB | TierKV Logical Compression | TierKV Physical Compression |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 384 | 17942.75 | 11750.07 | 8947.12 | 9504.00 | 15815.25 | 1.77x | 1.66x |
| 448 | 20711.66 | 13471.93 | 10570.31 | 11256.00 | 18583.12 | 1.76x | 1.65x |
| 512 | — | 15159.07 | 12215.50 | 12952.00 | 21373.00 | 1.75x | 1.65x |

### Staged Status

| Requests | Baseline status | TierKV status | Notes |
| ---: | --- | --- | --- |
| 384 | OK | OK | — |
| 448 | OK | OK | — |
| 512 | OOM | OK | Baseline: CUDA out of memory. Tried to allocate 22.00 MiB. GPU 0 has a total capacity of 22.06 GiB of which 23.44 MiB is free. Process 1 has 22.03 GiB memory in use. Of the allocated memory 20.51 GiB is allocated by PyTorch, and 1.22 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) |

### Staged Failure Boundary

| Method | Max Successful Requests | Largest Successful Total Context Tokens | Peak MB At Max Success | Logical Resident MB | Physical KV MB | Dense-Equivalent KV MB | Failure Request Count | Failure Type |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | 448 | 851200 | 20711.66 | 0.00 | 0.00 | 0.00 | 512 | OOM |
| Paged-TierKV | 512 | 972800 | 15159.07 | 12215.50 | 12952.00 | 21373.00 | n/a | none |

### Scaling Analysis

Common successful range: 384 to 448 requests.
Throughput slope: Baseline -0.0098 tok/s/request (-1.4%), TierKV -0.0069 tok/s/request (-1.1%).
Peak-memory slope: Baseline 43.26 MB/request (15.4%), TierKV 26.90 MB/request (14.7%).
TierKV physical KV slope: 27.38 MB/request (18.4%).
Throughput scaling conclusion: TierKV degrades more slowly over the common range.
Memory scaling conclusion: TierKV peak memory grows more slowly than baseline.
Failure-boundary conclusion: TierKV reaches a higher supported load.

### 7. Final Failure-Boundary Summary
| Method | Max Successful Requests | Largest Successful Total Context Tokens | Peak MB At Max Success | Failure Request Count | Failure Type |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline | 448 | 851200 | 20711.66 | 512 | OOM |
| Paged-TierKV | 512 | 972800 | 15159.07 | n/a | none |

## 2. 7B Extension Experiment
The 7B extension is an additional evaluation path and does not replace the TinyLlama main result.

### 2.1 Extension Model
Source logs: `extension_1024.log`, `extension_1536.log`

### 7B Extension Model Config
| Field | Value |
| --- | --- |
| model_id | lmsys/vicuna-7b-v1.5 |
| model_type | llama |
| hidden_size | 4096 |
| num_hidden_layers | 32 |
| num_attention_heads | 32 |
| num_key_value_heads | 32 |
| max_position_embeddings | 4096 |
| attention_layout | MHA |

### 2.2 Passkey Sanity
Source logs: `extension_1024.log`, `extension_1536.log`

### 7B Extension Passkey Sanity
| Method | Status | Contains Passkey | Peak MB | tok/s | Generated Snippet |
| --- | --- | --- | ---: | ---: | --- |
| Baseline | OK | True | 12895.28 | 23.04 | The passkey is 739241. nobodys perfectI'm sorry, |
| Paged-TierKV | OK | True | 12924.89 | 9.97 | The passkey is 739241. nobodys perfectI'm sorry, |

### 2.3 1024-Token Extension Scaling
Source log: `final_eval_logs/latest/extension_1024.log`

### 7B Extension Throughput Scaling
| Requests | Baseline tok/s | TierKV tok/s | TierKV / Baseline |
| ---: | ---: | ---: | ---: |
| 1 | 26.79 | 11.82 | 0.44x |
| 2 | 26.83 | 15.74 | 0.59x |
| 4 | 26.60 | 18.80 | 0.71x |
| 6 | 26.82 | 16.53 | 0.62x |
| 8 | 26.81 | 20.17 | 0.75x |
| 12 | 26.80 | 20.64 | 0.77x |

### 7B Extension Memory Scaling
| Requests | Baseline Peak MB | TierKV Peak MB | TierKV Logical Resident KV MB | TierKV Physical KV MB | TierKV Dense Eq KV MB | TierKV Logical Compression | TierKV Physical Compression |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 13389.32 | 13219.67 | 303.50 | 318.46 | 520.50 | 1.71x | 1.63x |
| 2 | 13917.83 | 13552.23 | 615.00 | 647.42 | 1049.00 | 1.71x | 1.62x |
| 4 | 14975.70 | 14213.68 | 1238.00 | 1301.84 | 2106.00 | 1.70x | 1.62x |
| 6 | 16049.19 | 14937.72 | 1861.50 | 1985.07 | 3179.00 | 1.71x | 1.60x |
| 8 | 17138.82 | 15621.59 | 2501.00 | 2675.30 | 4268.00 | 1.71x | 1.60x |
| 12 | 19285.36 | 16955.56 | 3748.00 | 4020.75 | 6414.00 | 1.71x | 1.60x |

### 7B Extension Status
| Requests | Baseline status | TierKV status | Notes |
| ---: | --- | --- | --- |
| 1 | OK | OK | — |
| 2 | OK | OK | — |
| 4 | OK | OK | — |
| 6 | OK | OK | — |
| 8 | OK | OK | — |
| 12 | OK | OK | — |

### 7B Extension Failure Boundary
| Method | Max Successful Requests | Largest Successful Total Context Tokens | Peak MB At Max Success | Logical Resident MB | Physical KV MB | Dense-Equivalent KV MB | Failure Request Count | Failure Type |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | 12 | 12288 | 19285.36 | 0.00 | 0.00 | 0.00 | n/a | none |
| Paged-TierKV | 12 | 12288 | 16955.56 | 3748.00 | 4020.75 | 6414.00 | n/a | none |

### Scaling Analysis
Common successful range: 1 to 12 requests.
Throughput slope: Baseline 0.0006 tok/s/request (0.0%), TierKV 0.8011 tok/s/request (74.5%).
Peak-memory slope: Baseline 536.00 MB/request (44.0%), TierKV 339.63 MB/request (28.3%).
TierKV physical KV slope: 336.57 MB/request (1162.6%).
Throughput scaling conclusion: TierKV degrades more slowly over the common range.
Memory scaling conclusion: TierKV peak memory grows more slowly than baseline.
Failure-boundary conclusion: Both methods reached the same maximum tested load.

### 7B Extension Crossover Analysis
Model: lmsys/vicuna-7b-v1.5
Context tokens: 1024
Decode steps: 16
No throughput crossover in the tested range; the TierKV throughput gap narrows (ratio 0.44x -> 0.77x).
Memory gap trend: 169.65 MB -> 2329.80 MB (delta 2160.15 MB).
Last physical compression: 1.60x.
Boundary: Baseline max 12 requests, TierKV max 12 requests.

### 2.4 1536-Token Extension Scaling
Source log: `final_eval_logs/latest/extension_1536.log`

### 7B Extension Throughput Scaling
| Requests | Baseline tok/s | TierKV tok/s | TierKV / Baseline |
| ---: | ---: | ---: | ---: |
| 1 | 25.48 | 11.41 | 0.45x |
| 2 | 25.57 | 15.08 | 0.59x |
| 4 | 25.59 | 18.01 | 0.70x |
| 6 | 25.57 | 15.72 | 0.61x |
| 8 | 25.56 | 19.15 | 0.75x |

### 7B Extension Memory Scaling
| Requests | Baseline Peak MB | TierKV Peak MB | TierKV Logical Resident KV MB | TierKV Physical KV MB | TierKV Dense Eq KV MB | TierKV Logical Compression | TierKV Physical Compression |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 13649.32 | 13347.90 | 435.50 | 448.66 | 776.50 | 1.78x | 1.73x |
| 2 | 14434.33 | 13808.46 | 879.00 | 907.81 | 1561.00 | 1.78x | 1.72x |
| 4 | 16003.71 | 14725.36 | 1766.00 | 1822.62 | 3130.00 | 1.77x | 1.72x |
| 6 | 17588.80 | 15762.14 | 2653.50 | 2761.09 | 4715.00 | 1.78x | 1.71x |
| 8 | 19190.84 | 16703.11 | 3557.00 | 3706.55 | 6316.00 | 1.78x | 1.70x |

### 7B Extension Status
| Requests | Baseline status | TierKV status | Notes |
| ---: | --- | --- | --- |
| 1 | OK | OK | — |
| 2 | OK | OK | — |
| 4 | OK | OK | — |
| 6 | OK | OK | — |
| 8 | OK | OK | — |

### 7B Extension Failure Boundary
| Method | Max Successful Requests | Largest Successful Total Context Tokens | Peak MB At Max Success | Logical Resident MB | Physical KV MB | Dense-Equivalent KV MB | Failure Request Count | Failure Type |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | 8 | 12288 | 19190.84 | 0.00 | 0.00 | 0.00 | n/a | none |
| Paged-TierKV | 8 | 12288 | 16703.11 | 3557.00 | 3706.55 | 6316.00 | n/a | none |

### Scaling Analysis
Common successful range: 1 to 8 requests.
Throughput slope: Baseline 0.0116 tok/s/request (0.3%), TierKV 1.1052 tok/s/request (67.8%).
Peak-memory slope: Baseline 791.65 MB/request (40.6%), TierKV 479.32 MB/request (25.1%).
TierKV physical KV slope: 465.41 MB/request (726.1%).
Throughput scaling conclusion: TierKV degrades more slowly over the common range.
Memory scaling conclusion: TierKV peak memory grows more slowly than baseline.
Failure-boundary conclusion: Both methods reached the same maximum tested load.

### 7B Extension Crossover Analysis
Model: lmsys/vicuna-7b-v1.5
Context tokens: 1536
Decode steps: 16
No throughput crossover in the tested range; the TierKV throughput gap narrows (ratio 0.45x -> 0.75x).
Memory gap trend: 301.42 MB -> 2487.73 MB (delta 2186.32 MB).
Last physical compression: 1.70x.
Boundary: Baseline max 8 requests, TierKV max 8 requests.

### 2.5 Extension Crossover Analysis

### 7B Extension Crossover Analysis
Model: lmsys/vicuna-7b-v1.5
Context tokens: 1024
Decode steps: 16
No throughput crossover in the tested range; the TierKV throughput gap narrows (ratio 0.44x -> 0.77x).
Memory gap trend: 169.65 MB -> 2329.80 MB (delta 2160.15 MB).
Last physical compression: 1.60x.
Boundary: Baseline max 12 requests, TierKV max 12 requests.

### 7B Extension Crossover Analysis
Model: lmsys/vicuna-7b-v1.5
Context tokens: 1536
Decode steps: 16
No throughput crossover in the tested range; the TierKV throughput gap narrows (ratio 0.45x -> 0.75x).
Memory gap trend: 301.42 MB -> 2487.73 MB (delta 2186.32 MB).
Last physical compression: 1.70x.
Boundary: Baseline max 8 requests, TierKV max 8 requests.

### 2.6 Extension Conclusion

### Final Extension Conclusion Guidance
Report this section separately from the TinyLlama result. Use the crossover analysis above to state whether TierKV exceeded baseline throughput, or whether the memory-pressure benefit did not translate into an absolute tok/s win.

### Final Extension Conclusion Guidance
Report this section separately from the TinyLlama result. Use the crossover analysis above to state whether TierKV exceeded baseline throughput, or whether the memory-pressure benefit did not translate into an absolute tok/s win.

## 3. Final High-Level Summary
- Main TinyLlama result: TierKV uses compact HOT+WARM storage and reports logical and physical KV compression.
- Boundary memory point: 512 requests, Baseline Peak `—` MB, TierKV Peak `15159.07` MB, TierKV Physical KV `12952.00` MB, Physical Compression `1.65x`.
- Boundary throughput ratios reported: 0.88x, 0.88x, —.
- Strongest systems claim: dense baseline hits the boundary first, while TierKV succeeds at the same request count.
- 7B extension result: this is a supporting experiment, not the primary TinyLlama result path.
- 1024-token extension: no throughput crossover; ratio moves from `0.44x` to `0.77x`.
- 1024-token extension: memory gap grows from `169.65` MB to `2329.80` MB; last physical compression `1.60x`.
- 1536-token extension: no throughput crossover; ratio moves from `0.45x` to `0.75x`.
- 1536-token extension: memory gap grows from `301.42` MB to `2487.73` MB; last physical compression `1.70x`.
- Extension conclusion: TierKV strengthens the memory-scaling story under a larger MHA model, but it does not exceed baseline raw decode throughput in the tested extension ranges.
