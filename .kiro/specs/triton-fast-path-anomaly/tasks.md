# Implementation Plan

- [x] 1. Write a bug-reproduction microbenchmark for the no-score anomaly
  - Goal: Reproduce and document the current anomaly in the single-kernel non-split decode path, where `return_scores=False` is unexpectedly slower than `return_scores=True`
  - Important: This is a performance regression microbenchmark, not a strict correctness/property test
  - Expected behavior on unfixed code: the benchmark should demonstrate that the no-score path is materially slower than the score path for representative block counts
  - Scope: restrict to the single-kernel non-split decode path (`use_split_decode=False`); compare `return_scores=True` vs `return_scores=False`
  - In `tierkv_triton.py`, build a minimal benchmark harness `benchmark_no_score_anomaly()` that:
    - Allocates a `TieredKVTensorPool` and populates `num_blocks` HOT blocks
    - Constructs valid GPU `block_table`, `block_states`, and `block_lengths` tensors
    - Constructs a fixed `query_states` tensor on CUDA
    - Calls `tierkv_decode_attention(...)` on identical inputs for both score and no-score modes
  - Use representative block counts: `num_blocks ∈ {4, 16, 64}`
  - Warm up both paths with at least 5 calls each before timing
  - Measure runtime over at least 50 iterations per path using `torch.cuda.synchronize()` around each call
  - Record: average latency for `return_scores=False`, average latency for `return_scores=True`, ratio `t_no_score / t_score`
  - Also verify the public API contract:
    - `tierkv_decode_attention(..., return_scores=False)` returns `(out, None)`
    - `tierkv_decode_attention(..., return_scores=True)` returns `(out, score_sums_tensor)`
  - Do not assert a brittle hard requirement like `t_no_score <= t_score` as a correctness invariant
  - Instead, record and report the anomaly, e.g. `"num_blocks=16: no-score 0.40 ms, score 0.26 ms, ratio=1.54x"`
  - Run this microbenchmark on unfixed code
  - Expected outcome: the benchmark reproduces that the no-score path is materially slower than the score path; the public API may already return `None` or may require wrapper adjustment
  - Mark task complete when: the benchmark is written, run on unfixed code, and counterexamples are documented
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 2. Write preservation / correctness tests before fixing the anomaly
  - Goal: Lock in the behaviors that must remain correct after the fast-path fix
  - Important: these are correctness-preservation tests; they should pass on unfixed code; they must not depend on performance ratios
  - Scope: `return_scores=True` path, mixed HOT/WARM/COLD behavior, split-decode path, eager reference compatibility
  - Observe the unfixed code first and then write assertions that match actual intended semantics
  - In `tierkv_triton.py`, write a test suite `test_preservation_properties()` covering:

  - [x] 2.1 Preservation test: score-path output shape and normalization
    - For `return_scores=True`, all-HOT blocks: record `score_sums.shape` and whether block probabilities over active blocks sum approximately to 1 per head
    - Write test asserting: `score_sums.shape == (num_heads, max_blocks_per_layer)`
    - Assert `score_sums[:, :num_blocks].sum(dim=-1)` is approximately `ones(num_heads)` with `atol=2e-2`
    - Only use this assertion if `score_sums` semantics are indeed normalized probability mass over active logical blocks

  - [x] 2.2 Preservation test: mixed-state output correctness
    - For random or manually varied HOT/WARM/COLD block assignments with `return_scores=True`, compare Triton output against an eager reference
    - The eager reference must preserve current TierKV semantics: HOT blocks use FP tensors directly, WARM blocks are explicitly dequantized, COLD blocks are reconstructed as zero tensors while preserving valid sequence positions
    - Assert `attn_output` matches eager reference within `atol=2e-2, rtol=2e-2`

  - [x] 2.3 Preservation test: split-decode path remains correct
    - With `TIERKV_SPLIT_DECODE=1` and `num_blocks > 4`, for `return_scores=False`: verify output matches eager reference and wrapper returns `(out, None)`

  - [x] 2.4 Preservation test: public API contract
    - Assert `return_scores=False` returns `score_sums is None`
    - Assert `return_scores=True` returns a tensor of expected shape
    - This test is about the wrapper contract, not about whether the kernel internally uses a placeholder pointer

  - Run all preservation tests on unfixed code
  - Expected outcome: tests pass
  - Mark task complete when: tests are written and run successfully on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

- [x] 3. Fix the no-score fast-path anomaly in `tierkv_decode_attention`

  - [x] 3.1 Fix the public no-score API contract
    - In `tierkv_triton.py`, function `tierkv_decode_attention`, ensure that when `return_scores=False` the function returns `(out, None)` and when `return_scores=True` it returns `(out, score_sums_tensor)`
    - Do not require the internal Triton kernel signature to change yet; if the kernel still needs a score buffer pointer for signature compatibility, that is acceptable as an internal implementation detail
    - Goal: remove any externally visible dummy score tensor on the no-score path; make the wrapper contract clean and explicit
    - Bug scope: single-kernel non-split decode path with `return_scores=False`
    - Expected behavior: public wrapper returns `None` on no-score path
    - Preservation: score-path tensor output is unchanged
    - _Requirements: 1.5, 2.5, 3.1_

  - [x] 3.2 Split the single-kernel launch into explicit score / no-score branches
    - In `tierkv_triton.py`, replace the current shared single-kernel launch path with two explicit branches for the non-split path: one for `return_scores=True`, one for `return_scores=False`
    - Use the same explicit launch hints in both branches initially: `num_warps=4`, `num_stages=2`
    - The purpose is to reduce the chance that divergent Triton specialization or launch configuration is driving the anomaly
    - If the kernel signature still requires a score buffer pointer, use a private placeholder tensor internally for the no-score branch; this placeholder must not leak through the public return value
    - Prefer reusing a cached placeholder tensor (module-level or runtime-level, per device/dtype) instead of allocating a fresh `torch.empty((1,))` on every decode call
    - Update the final return to: `return out, score_sums` when `return_scores=True`; `return out, None` when `return_scores=False`
    - Keep the split-decode path unchanged
    - Bug scope: single-kernel non-split decode path only
    - Expected behavior: the no-score branch should no longer be materially slower due to an avoidable specialization / launch-path problem
    - Preservation: score-path correctness is unchanged; split-decode path remains untouched
    - _Requirements: 1.4, 2.2, 2.4, 3.4_

  - [x] 3.3 Add instrumentation to compare score vs no-score launch paths
    - Add lightweight profiling around the two branches to measure: wrapper overhead, Triton kernel time, any synchronization cost, any extra temporary tensor allocation cost
    - If possible, log or inspect: launch hints, specialized kernel variants used by Triton, whether the no-score path accidentally triggers a different slower specialization
    - This step is diagnostic and supports the fix, but should not permanently clutter the fast path

  - [x] 3.4 Re-run the bug-reproduction microbenchmark from step 1
    - Do not write a new benchmark; reuse the same `benchmark_no_score_anomaly()` harness from task 1
    - Run it on fixed code
    - Expected outcome: no-score path is no longer materially slower than score path; ideally it becomes faster; public API returns `None` on the no-score path
    - Because performance is noisy, do not require exact ratio ≤ 1.0 in all cases; prefer a practical criterion such as ratio no worse than ~1.05x across representative cases, or no-score is faster in most tested cases
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 3.5 Re-run all preservation tests from step 2
    - Do not write new preservation tests; reuse the same `test_preservation_properties()` from task 2
    - Confirm: `return_scores=True` behavior is unchanged, mixed-state correctness is preserved, split-decode path is unaffected, wrapper contract is correct
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

- [x] 4. Checkpoint — Ensure the fix is correct and stable
  - Re-run: `benchmark_no_score_anomaly()` from task 1, all preservation tests from task 2, existing integration tests including `test_tensor_pool_and_triton_decode` in `main.py`
  - Confirm:
    - Public API returns `None` when `return_scores=False`
    - Public API returns correctly shaped tensor when `return_scores=True`
    - Split-decode path remains correct
    - Mixed HOT/WARM/COLD correctness is unchanged
    - No-score path is no longer materially slower in representative microbenchmarks
  - Document final observations for `num_blocks ∈ {4, 16, 64}`
  - Mark task complete when: correctness-preservation tests pass, anomaly benchmark no longer shows pathological no-score slowdown, existing integration tests still pass
