# Triton Fast-Path Anomaly Bugfix Design

## Overview

The `return_scores=False` path in `tierkv_decode_attention` (`tierkv_triton.py`) is
unexpectedly slower than `return_scores=True` despite performing strictly less work (one KV
pass instead of two). The measured throughput ratio is 0.33x–0.50x across context lengths
512–1900 tokens, which directly negates the benefit of the `policy_interval` optimization.

The fix has two parts:

1. **Eliminate the dummy score buffer**: When `return_scores=False`, the current code still
   allocates `score_sums = torch.empty((1,), ...)` and passes it to the kernel as `score_ptr`.
   The fix removes this allocation entirely and avoids passing the pointer to the kernel on
   the no-score path.

2. **Equalize kernel launch configuration**: Because `return_scores` is a `tl.constexpr`,
   Triton generates two separate kernel specializations. The `True` variant has higher memory
   traffic (second KV pass + score writes), which may cause Triton's heuristics to assign
   more warps or pipeline stages, giving it better GPU occupancy. The fix adds explicit
   `num_warps` and `num_stages` hints to the no-score kernel launch so both variants use
   equivalent configurations.

The split-kernel path (`TIERKV_SPLIT_DECODE=1`) is unaffected. All correctness tolerances
(atol=2e-2, rtol=2e-2) and the score-path output contract are preserved.

---

## Glossary

- **Bug_Condition (C)**: The condition that triggers the anomaly — `tierkv_decode_attention`
  is called with `return_scores=False`.
- **Property (P)**: The desired behavior — the no-score path executes faster than or equal to
  the score path, and returns a correct attention output tensor.
- **Preservation**: All behaviors that must remain unchanged: score-path correctness, output
  numerical accuracy, mixed-state block handling, split-decode path, policy scheduling logic,
  and eager fallback.
- **`tierkv_decode_attention`**: The Python dispatch function in `tierkv_triton.py` that
  validates inputs, allocates output buffers, and launches the Triton kernel.
- **`_tierkv_decode_attention_kernel`**: The single-pass Triton JIT kernel. When
  `return_scores=True` it performs a second full KV-block pass to accumulate per-block
  probability sums; when `return_scores=False` it skips that pass.
- **`return_scores`**: A `tl.constexpr` kernel argument that causes Triton to compile two
  distinct kernel specializations — one with the score-accumulation loop and one without.
- **`score_sums`**: The `(num_heads, max_blocks_per_layer)` float32 tensor returned when
  `return_scores=True`. Currently a dummy `(1,)` tensor is allocated even when
  `return_scores=False`.
- **`num_warps` / `num_stages`**: Triton launch-time hints that control warp count and
  software-pipeline depth. Triton's autotuner may select different values for the two
  specializations, causing the no-score variant to receive a weaker configuration.
- **`TIERKV_SPLIT_DECODE`**: Environment variable that activates the split-kernel path
  (`_tierkv_decode_attention_split_kernel` + `_tierkv_decode_attention_reduce_kernel`).
  This path is not affected by the bug or the fix.

---

## Bug Details

### Bug Condition

The anomaly manifests whenever `tierkv_decode_attention` is called with
`return_scores=False`. The function either passes an unnecessary dummy score buffer to the
kernel, or the Triton autotuner selects a weaker launch configuration for the no-score
specialization, or both.

**Formal Specification:**
```
FUNCTION isBugCondition(call)
  INPUT: call — a single invocation of tierkv_decode_attention
  OUTPUT: boolean

  RETURN call.return_scores == False
         AND call.use_split_decode == False
         AND latency(call) > latency(equivalent_call_with_return_scores_True)
END FUNCTION
```

### Examples

- **Context 512 tokens, return_scores=False**: Observed ~0.33x throughput of the
  `return_scores=True` call on the same input. Expected: ≥1.0x (faster or equal).
- **Context 1024 tokens, return_scores=False**: Observed ~0.40x throughput. Expected: ≥1.0x.
- **Context 1900 tokens, return_scores=False**: Observed ~0.50x throughput. Expected: ≥1.0x.
- **Steady-state decode (policy_ticks mode, interval=32)**: 31 out of every 32 decode steps
  use `return_scores=False`. The anomaly suppresses throughput on 97% of decode steps.

---

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- `return_scores=True` MUST continue to return a `score_sums` tensor of shape
  `(num_heads, max_blocks_per_layer)` whose per-head active-block values sum to
  approximately 1.0 (atol=2e-2).
- `return_scores=False` MUST continue to return an attention output tensor that matches
  the reference eager-attention output within atol=2e-2, rtol=2e-2.
- Mixed HOT/WARM/COLD block configurations MUST continue to produce correct attention
  output for both `return_scores` values, including correct WARM dequantization and COLD
  zero-masking.
- The `TIERKV_SPLIT_DECODE=1` path MUST remain entirely unaffected.
- The `score_accumulation_mode="policy_ticks"` scheduling logic
  (`should_collect_decode_scores` / `should_run_decode_policy`) MUST remain unchanged.
- The `score_accumulation_mode="every_step"` mode MUST continue to pass
  `return_scores=True` on every decode step.
- The eager fallback (`TIERKV_ATTENTION_BACKEND=eager`) MUST continue to work without error.

**Scope:**
All inputs that do NOT satisfy `isBugCondition` — i.e., calls with `return_scores=True`,
calls using the split-decode path, and all eager-backend calls — MUST be completely
unaffected by this fix.

---

## Hypothesized Root Cause

Based on code inspection of `tierkv_triton.py`, the most likely causes are:

1. **Triton autotuner configuration asymmetry**: `return_scores` is a `tl.constexpr`,
   so Triton compiles two separate kernel objects. The `True` variant has higher memory
   traffic (second full KV-block loop + score-buffer stores), which may cause Triton's
   internal heuristics to assign more warps or pipeline stages. The `False` variant, having
   less apparent work, may receive fewer warps or shallower pipelining, resulting in lower
   GPU occupancy and worse memory-bandwidth utilization despite doing less arithmetic.

2. **Unnecessary dummy score buffer allocation and pointer passing**: When
   `return_scores=False`, the current code executes:
   ```python
   score_sums = torch.empty((1,), device=query_states.device, dtype=torch.float32)
   ```
   and passes this pointer to the kernel as `score_ptr`. Even though the kernel's
   `if return_scores:` branch is dead code in the `False` specialization, the pointer
   argument is still present in the kernel signature, potentially affecting register
   allocation or preventing certain compiler optimizations.

3. **Kernel argument count / register pressure**: The `False` specialization carries
   `score_ptr` and `max_blocks_per_layer` as arguments that serve no purpose. Removing
   them from the no-score launch path reduces register pressure and may allow the compiler
   to produce a tighter kernel.

4. **Cache-line or L2 thrashing from dummy buffer**: The `torch.empty((1,), ...)` allocation
   is small but introduces a CUDA memory allocation on every decode step when
   `return_scores=False`, adding latency outside the kernel itself.

---

## Correctness Properties

Property 1: Bug Condition — No-Score Path Is Faster Than Score Path

_For any_ valid input to `tierkv_decode_attention` where `return_scores=False` and the
single-kernel path is active (i.e., `TIERKV_SPLIT_DECODE` is not set or `num_blocks <= 4`),
the fixed function SHALL complete in less time than or equal to the time taken by an
equivalent call with `return_scores=True` on the same input, reflecting the savings from
skipping the second KV-block pass and score-buffer writes.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4**

Property 2: Preservation — Attention Output Correctness Is Unchanged

_For any_ input where the bug condition does NOT hold (i.e., `return_scores=True`, or the
split-decode path is active), the fixed `tierkv_decode_attention` SHALL produce attention
output and score tensors that are numerically identical (within atol=2e-2, rtol=2e-2) to
those produced by the original function, preserving all existing correctness guarantees for
the score path, mixed-state blocks, and split-decode path.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4**

---

## Fix Implementation

### Changes Required

Assuming the root cause analysis is correct (autotuner asymmetry + dummy buffer):

**File**: `tierkv_triton.py`

**Function**: `tierkv_decode_attention`

**Specific Changes**:

1. **Remove dummy score buffer allocation**: Replace the unconditional
   `torch.empty((1,), ...)` allocation with a conditional allocation only when
   `return_scores=True`:
   ```python
   # Before
   score_sums = (
       torch.zeros((num_heads, block_table.shape[0]), ...) if return_scores
       else torch.empty((1,), ...)
   )

   # After
   score_sums = (
       torch.zeros((num_heads, block_table.shape[0]), ...) if return_scores
       else None
   )
   ```

2. **Split the kernel launch into two branches**: Rather than passing `score_sums` (or a
   dummy) unconditionally, use separate launch calls for the two paths:
   ```python
   if return_scores:
       _tierkv_decode_attention_kernel[(num_heads,)](
           ..., score_sums, ..., return_scores=True,
           num_warps=NUM_WARPS, num_stages=NUM_STAGES,
       )
   else:
       _tierkv_decode_attention_kernel[(num_heads,)](
           ..., score_sums_placeholder, ..., return_scores=False,
           num_warps=NUM_WARPS, num_stages=NUM_STAGES,
       )
   ```
   where `NUM_WARPS` and `NUM_STAGES` are explicit constants (e.g., `num_warps=4`,
   `num_stages=2`) chosen to match or exceed the configuration Triton selects for the
   `True` variant.

3. **Add explicit `num_warps` / `num_stages` to the no-score launch**: Ensure the
   `return_scores=False` specialization uses at least as many warps and pipeline stages as
   the `return_scores=True` specialization. The exact values should be determined
   empirically (e.g., by inspecting `kernel.best_config` after a warm-up run), but
   `num_warps=4, num_stages=2` is a reasonable starting point for the block sizes in use.

4. **Return `None` for score_sums when `return_scores=False`**: Update the return statement
   to return `(out, None)` on the no-score path, removing the dummy tensor from the return
   value. This is already the documented contract (callers check `if score_sums is not None`).

5. **Verify the split-decode path is untouched**: The `use_split_decode` branch returns
   `(out, None)` already and does not use `score_sums`. No changes needed there.

---

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that
demonstrate the performance anomaly on unfixed code, then verify the fix makes the no-score
path faster and preserves all correctness guarantees.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate the anomaly BEFORE implementing the fix.
Confirm or refute the root cause analysis. If refuted, re-hypothesize.

**Test Plan**: Write a microbenchmark that calls `tierkv_decode_attention` with identical
inputs under both `return_scores=True` and `return_scores=False`, measures wall-clock time
over many iterations (after warm-up), and asserts that the no-score path is faster. Run on
UNFIXED code to observe the failure and confirm the anomaly is real.

**Test Cases**:
1. **Short context (num_blocks=4)**: Benchmark both paths with 4 blocks. Expect no-score
   to be slower on unfixed code (will fail assertion).
2. **Medium context (num_blocks=16)**: Benchmark both paths with 16 blocks. Expect no-score
   to be slower on unfixed code (will fail assertion).
3. **Long context (num_blocks=64)**: Benchmark both paths with 64 blocks. Expect no-score
   to be slower on unfixed code (will fail assertion).
4. **Dummy buffer check**: Assert that when `return_scores=False`, the returned
   `score_sums` is `None` and no `(num_heads, max_blocks)` tensor was allocated. Will fail
   on unfixed code because a dummy `(1,)` tensor is returned.

**Expected Counterexamples**:
- `latency(return_scores=False) > latency(return_scores=True)` for all tested block counts.
- Possible causes: Triton autotuner assigns fewer warps to the no-score specialization;
  dummy buffer allocation adds overhead; pointer argument prevents compiler optimization.

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed function
produces the expected behavior (no-score path is faster).

**Pseudocode:**
```
FOR ALL input WHERE isBugCondition(input) DO
  t_no_score := time(tierkv_decode_attention_fixed(input, return_scores=False))
  t_score    := time(tierkv_decode_attention_fixed(input, return_scores=True))
  ASSERT t_no_score <= t_score
  ASSERT score_sums_no_score IS None
END FOR
```

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold, the fixed
function produces the same result as the original function.

**Pseudocode:**
```
FOR ALL input WHERE NOT isBugCondition(input) DO
  out_orig, scores_orig := tierkv_decode_attention_original(input)
  out_fixed, scores_fixed := tierkv_decode_attention_fixed(input)
  ASSERT allclose(out_orig, out_fixed, atol=2e-2, rtol=2e-2)
  IF input.return_scores:
    ASSERT scores_fixed.shape == (num_heads, max_blocks_per_layer)
    ASSERT allclose(scores_fixed[:, :num_blocks].sum(dim=-1), ones(num_heads), atol=2e-2)
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking
because:
- It generates many random KV configurations (varying block counts, HOT/WARM/COLD mixes,
  head counts, head dims) automatically.
- It catches edge cases (all-COLD blocks, single block, max blocks) that manual tests miss.
- It provides strong guarantees that numerical correctness is preserved across the full
  input domain.

**Test Plan**: Observe correctness on UNFIXED code first for `return_scores=True` and mixed
block states, then write property-based tests capturing that behavior.

**Test Cases**:
1. **Score-path correctness preservation**: Verify `return_scores=True` output and
   `score_sums` are numerically unchanged after the fix.
2. **All-HOT block correctness**: Verify no-score output matches reference for all-HOT
   configurations.
3. **Mixed HOT/WARM/COLD correctness**: Verify no-score output matches reference for
   randomly generated block state assignments.
4. **Split-decode path unaffected**: Verify `TIERKV_SPLIT_DECODE=1` output is unchanged.

### Unit Tests

- Benchmark `return_scores=False` vs `return_scores=True` and assert no-score is faster
  after the fix (for num_blocks in {4, 16, 64}).
- Assert `score_sums is None` when `return_scores=False` after the fix.
- Assert `score_sums.shape == (num_heads, max_blocks_per_layer)` when `return_scores=True`.
- Assert attention output matches reference eager attention within atol=2e-2, rtol=2e-2
  for both `return_scores` values.

### Property-Based Tests

- Generate random `(num_blocks, block_state_assignment)` pairs and verify that the fixed
  no-score kernel output matches the reference eager output within tolerance for all
  generated configurations.
- Generate random `(num_heads, kv_heads, head_dim, num_blocks)` combinations and verify
  that the score-path output (`score_sums`) sums to approximately 1.0 per head over active
  blocks, confirming the score path is unaffected by the fix.
- Generate random block counts spanning 1 to `max_blocks_per_layer` and verify that
  `latency(return_scores=False) <= latency(return_scores=True)` holds across the full range.

### Integration Tests

- Run the full decode benchmark (`run_decode_throughput_benchmark`) with
  `score_accumulation_mode="policy_ticks"` and `policy_interval=32` before and after the
  fix; assert that tokens/sec increases after the fix.
- Run `test_tensor_pool_and_triton_decode` (existing integration test) and verify all
  assertions still pass after the fix.
- Run a decode loop with `score_accumulation_mode="every_step"` and verify that
  `score_sums` is non-None on every step, confirming the every-step mode is unaffected.
