# Bugfix Requirements Document

## Introduction

The nominal no-score fast path (`return_scores=False`) in the TierKV Triton decode kernel is
unexpectedly slower than the forced-score path (`return_scores=True`). Measured decode throughput
shows the no-score path running at roughly 0.33x–0.50x the speed of the score path across all
tested context lengths (512–1900 tokens). Because `return_scores=False` is the common case during
steady-state decoding (scores are only collected every `policy_interval` steps), this anomaly
directly suppresses the throughput gains that the policy-overhead optimization pass was intended
to deliver. The fix must make `return_scores=False` the true fast path without removing Triton,
without redesigning the memory layout, and without reintroducing Python-side KV reconstruction.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN `tierkv_decode_attention` is called with `return_scores=False` THEN the system executes
    the decode step slower than when called with `return_scores=True` (observed ratio: 0.33x–0.50x
    across context lengths 512–1900 tokens).

1.2 WHEN `return_scores=False` is passed as a `tl.constexpr` to
    `_tierkv_decode_attention_kernel` THEN the system produces a Triton kernel specialization
    that is slower than the `return_scores=True` specialization, despite the `True` variant
    performing a second full pass over all KV blocks to accumulate per-block probability sums.

1.3 WHEN `return_scores=False` is active during steady-state decoding (i.e., on all decode steps
    that are not policy-tick steps) THEN the system fails to realize the throughput benefit of
    skipping score accumulation, negating the purpose of the `policy_interval` optimization.

1.4 WHEN the `return_scores=False` kernel variant is launched THEN the system may use a
    suboptimal launch configuration (e.g., fewer warps, fewer pipeline stages, or a different
    register allocation) compared to the `return_scores=True` variant, causing lower GPU
    occupancy or memory-bandwidth utilization on the no-score path.

1.5 WHEN `return_scores=False` is selected THEN the system allocates `score_sums` as
    `torch.empty((1,), ...)` and passes it to the kernel as a write target, potentially
    introducing unnecessary pointer indirection or preventing compiler optimizations that
    would otherwise be available when the score output buffer is absent.

### Expected Behavior (Correct)

2.1 WHEN `tierkv_decode_attention` is called with `return_scores=False` THEN the system SHALL
    execute the decode step faster than or equal to the `return_scores=True` path, because the
    no-score path performs strictly less work (one KV pass instead of two).

2.2 WHEN `return_scores=False` is passed to `_tierkv_decode_attention_kernel` THEN the system
    SHALL produce a Triton kernel specialization whose launch configuration (num_warps,
    num_stages) is at least as performant as the `return_scores=True` specialization.

2.3 WHEN `return_scores=False` is active during steady-state decoding THEN the system SHALL
    deliver measurably higher decode throughput than the `return_scores=True` path, reflecting
    the savings from skipping the second KV-block pass and score-buffer writes.

2.4 WHEN the `return_scores=False` kernel variant is launched THEN the system SHALL use a
    launch configuration that achieves occupancy and memory-bandwidth utilization at least
    equal to the `return_scores=True` variant, either by explicit `num_warps`/`num_stages`
    hints or by restructuring the kernel so Triton's autotuner selects equivalent or better
    configurations for both variants.

2.5 WHEN `return_scores=False` is selected THEN the system SHALL NOT allocate or pass a
    score output buffer to the kernel; the score accumulation pointer and associated stores
    SHALL be fully absent from the no-score kernel path.

### Unchanged Behavior (Regression Prevention)

3.1 WHEN `return_scores=True` is requested THEN the system SHALL CONTINUE TO return a valid
    `score_sums` tensor of shape `(num_heads, max_blocks_per_layer)` whose per-head values
    sum to approximately 1.0 over the active blocks, within the existing tolerance (atol=2e-2).

3.2 WHEN `return_scores=False` is requested THEN the system SHALL CONTINUE TO return an
    attention output tensor that matches the reference eager-attention output within the
    existing tolerance (atol=2e-2, rtol=2e-2).

3.3 WHEN blocks in mixed HOT/WARM/COLD states are present THEN the system SHALL CONTINUE TO
    produce correct attention output for both `return_scores=True` and `return_scores=False`,
    including correct dequantization of WARM blocks and zero-masking of COLD blocks.

3.4 WHEN `TIERKV_SPLIT_DECODE=1` is set THEN the system SHALL CONTINUE TO use the split-kernel
    path for `return_scores=False` with `num_blocks > 4`, and that path SHALL remain unaffected
    by any changes made to the single-kernel path.

3.5 WHEN `score_accumulation_mode="policy_ticks"` is configured THEN the system SHALL CONTINUE
    TO collect scores only on policy-tick decode steps and skip score collection on all other
    steps, preserving the existing `should_collect_decode_scores` / `should_run_decode_policy`
    scheduling logic.

3.6 WHEN `score_accumulation_mode="every_step"` is configured THEN the system SHALL CONTINUE
    TO collect scores on every decode step, passing `return_scores=True` to the kernel on
    every step.

3.7 WHEN the Triton backend is unavailable or `TIERKV_ATTENTION_BACKEND=eager` is set THEN
    the system SHALL CONTINUE TO fall back to the eager reconstruction path without error.
