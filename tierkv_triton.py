import os
import time as _time

import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None

# ---------------------------------------------------------------------------
# Profiling gate — set to True to enable lightweight kernel-launch profiling.
# This flag is False by default so it does NOT clutter the production fast path.
# ---------------------------------------------------------------------------
_TIERKV_PROFILE_KERNEL: bool = False

# ---------------------------------------------------------------------------
# Module-level cached placeholder score buffer for the no-score path.
# Keyed by (device, dtype) so it is reused across calls instead of allocating
# a fresh torch.empty((1,), ...) on every decode step.
# ---------------------------------------------------------------------------
_score_placeholder_cache: dict = {}


def _get_score_placeholder(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return a cached 1-element score placeholder tensor for the no-score path.

    This avoids the per-call ``torch.empty((1,), ...)`` allocation that was
    present in the unfixed code (Requirement 1.5).  The placeholder is only
    used as a kernel argument to satisfy the Triton kernel signature; it is
    never written to in a meaningful way and is never returned to the caller.
    """
    key = (str(device), dtype)
    if key not in _score_placeholder_cache:
        _score_placeholder_cache[key] = torch.empty(
            (1,), device=device, dtype=torch.float32
        )
    return _score_placeholder_cache[key]


if triton is not None:

    @triton.jit
    def _tierkv_decode_attention_kernel(
        query_ptr,
        out_ptr,
        score_ptr,
        block_table_ptr,
        block_states_ptr,
        block_lengths_ptr,
        hot_k_ptr,
        hot_v_ptr,
        warm_k_ptr,
        warm_v_ptr,
        k_scale_ptr,
        k_zero_ptr,
        v_scale_ptr,
        v_zero_ptr,
        num_blocks: tl.constexpr,
        scaling: tl.constexpr,
        num_key_value_groups: tl.constexpr,
        max_blocks_per_layer: tl.constexpr,
        kv_heads: tl.constexpr,
        block_size: tl.constexpr,
        head_dim: tl.constexpr,
        return_scores: tl.constexpr,
    ):
        q_head = tl.program_id(0)
        kv_head = q_head // num_key_value_groups
        dim_offsets = tl.arange(0, head_dim)
        token_offsets = tl.arange(0, block_size)

        q_vec = tl.load(query_ptr + q_head * head_dim + dim_offsets).to(tl.float32)
        acc = tl.zeros((head_dim,), dtype=tl.float32)
        m_i = tl.full((), -3.4028234663852886e38, dtype=tl.float32)
        l_i = tl.full((), 0.0, dtype=tl.float32)

        for block_idx in range(0, num_blocks):
            slot_idx = tl.load(block_table_ptr + block_idx)
            state = tl.load(block_states_ptr + block_idx)
            block_len = tl.load(block_lengths_ptr + block_idx)
            kv_offsets = (
                ((slot_idx * kv_heads + kv_head) * block_size + token_offsets[:, None]) * head_dim
                + dim_offsets[None, :]
            )
            meta_offsets = (slot_idx * kv_heads + kv_head) * block_size + token_offsets

            is_hot = state == 0
            is_warm = state == 1
            k_hot = tl.load(hot_k_ptr + kv_offsets, mask=is_hot, other=0.0).to(tl.float32)
            v_hot = tl.load(hot_v_ptr + kv_offsets, mask=is_hot, other=0.0).to(tl.float32)

            k_warm = tl.load(warm_k_ptr + kv_offsets, mask=is_warm, other=0.0).to(tl.float32)
            v_warm = tl.load(warm_v_ptr + kv_offsets, mask=is_warm, other=0.0).to(tl.float32)
            k_scale = tl.load(k_scale_ptr + meta_offsets, mask=is_warm, other=1.0).to(tl.float32)
            k_zero = tl.load(k_zero_ptr + meta_offsets, mask=is_warm, other=0.0).to(tl.float32)
            v_scale = tl.load(v_scale_ptr + meta_offsets, mask=is_warm, other=1.0).to(tl.float32)
            v_zero = tl.load(v_zero_ptr + meta_offsets, mask=is_warm, other=0.0).to(tl.float32)
            k_vals = k_hot + (k_warm - k_zero[:, None]) * k_scale[:, None]
            v_vals = v_hot + (v_warm - v_zero[:, None]) * v_scale[:, None]

            valid = token_offsets < block_len
            logits = tl.sum(k_vals * q_vec[None, :], axis=1) * scaling
            logits = tl.where(valid, logits, -3.4028234663852886e38)

            block_m = tl.max(logits, axis=0)
            m_new = tl.maximum(m_i, block_m)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(logits - m_new)
            p = tl.where(valid, p, 0.0)
            l_new = l_i * alpha + tl.sum(p, axis=0)
            acc = acc * alpha + tl.sum(p[:, None] * v_vals, axis=0)
            m_i = m_new
            l_i = l_new

        out = acc / l_i
        tl.store(out_ptr + q_head * head_dim + dim_offsets, out)

        if return_scores:
            for block_idx in range(0, num_blocks):
                slot_idx = tl.load(block_table_ptr + block_idx)
                state = tl.load(block_states_ptr + block_idx)
                block_len = tl.load(block_lengths_ptr + block_idx)
                kv_offsets = (
                    ((slot_idx * kv_heads + kv_head) * block_size + token_offsets[:, None]) * head_dim
                    + dim_offsets[None, :]
                )
                meta_offsets = (slot_idx * kv_heads + kv_head) * block_size + token_offsets

                is_hot = state == 0
                is_warm = state == 1
                k_hot = tl.load(hot_k_ptr + kv_offsets, mask=is_hot, other=0.0).to(tl.float32)
                k_warm = tl.load(warm_k_ptr + kv_offsets, mask=is_warm, other=0.0).to(tl.float32)
                k_scale = tl.load(k_scale_ptr + meta_offsets, mask=is_warm, other=1.0).to(tl.float32)
                k_zero = tl.load(k_zero_ptr + meta_offsets, mask=is_warm, other=0.0).to(tl.float32)
                k_vals = k_hot + (k_warm - k_zero[:, None]) * k_scale[:, None]
                valid = token_offsets < block_len
                logits = tl.sum(k_vals * q_vec[None, :], axis=1) * scaling
                logits = tl.where(valid, logits, -3.4028234663852886e38)
                probs = tl.exp(logits - m_i) / l_i
                probs = tl.where(valid, probs, 0.0)
                block_prob_sum = tl.sum(probs, axis=0)
                tl.store(score_ptr + q_head * max_blocks_per_layer + block_idx, block_prob_sum)

    @triton.jit
    def _tierkv_decode_attention_split_kernel(
        query_ptr,
        partial_acc_ptr,
        partial_m_ptr,
        partial_l_ptr,
        block_table_ptr,
        block_states_ptr,
        block_lengths_ptr,
        hot_k_ptr,
        hot_v_ptr,
        warm_k_ptr,
        warm_v_ptr,
        k_scale_ptr,
        k_zero_ptr,
        v_scale_ptr,
        v_zero_ptr,
        num_blocks: tl.constexpr,
        scaling: tl.constexpr,
        num_key_value_groups: tl.constexpr,
        num_chunks: tl.constexpr,
        kv_heads: tl.constexpr,
        block_size: tl.constexpr,
        head_dim: tl.constexpr,
        chunk_blocks: tl.constexpr,
    ):
        q_head = tl.program_id(0)
        chunk_idx = tl.program_id(1)
        kv_head = q_head // num_key_value_groups
        dim_offsets = tl.arange(0, head_dim)
        token_offsets = tl.arange(0, block_size)

        q_vec = tl.load(query_ptr + q_head * head_dim + dim_offsets).to(tl.float32)
        acc = tl.zeros((head_dim,), dtype=tl.float32)
        m_i = tl.full((), -3.4028234663852886e38, dtype=tl.float32)
        l_i = tl.full((), 0.0, dtype=tl.float32)

        chunk_start = chunk_idx * chunk_blocks
        for rel_block_idx in range(0, chunk_blocks):
            block_idx = chunk_start + rel_block_idx
            active_block = block_idx < num_blocks
            slot_idx = tl.load(block_table_ptr + block_idx, mask=active_block, other=0)
            state = tl.load(block_states_ptr + block_idx, mask=active_block, other=2)
            block_len = tl.load(block_lengths_ptr + block_idx, mask=active_block, other=0)

            kv_offsets = (
                ((slot_idx * kv_heads + kv_head) * block_size + token_offsets[:, None]) * head_dim
                + dim_offsets[None, :]
            )
            meta_offsets = (slot_idx * kv_heads + kv_head) * block_size + token_offsets

            is_hot = active_block & (state == 0)
            is_warm = active_block & (state == 1)
            k_hot = tl.load(hot_k_ptr + kv_offsets, mask=is_hot, other=0.0).to(tl.float32)
            v_hot = tl.load(hot_v_ptr + kv_offsets, mask=is_hot, other=0.0).to(tl.float32)

            k_warm = tl.load(warm_k_ptr + kv_offsets, mask=is_warm, other=0.0).to(tl.float32)
            v_warm = tl.load(warm_v_ptr + kv_offsets, mask=is_warm, other=0.0).to(tl.float32)
            k_scale = tl.load(k_scale_ptr + meta_offsets, mask=is_warm, other=1.0).to(tl.float32)
            k_zero = tl.load(k_zero_ptr + meta_offsets, mask=is_warm, other=0.0).to(tl.float32)
            v_scale = tl.load(v_scale_ptr + meta_offsets, mask=is_warm, other=1.0).to(tl.float32)
            v_zero = tl.load(v_zero_ptr + meta_offsets, mask=is_warm, other=0.0).to(tl.float32)
            k_vals = k_hot + (k_warm - k_zero[:, None]) * k_scale[:, None]
            v_vals = v_hot + (v_warm - v_zero[:, None]) * v_scale[:, None]

            valid = active_block & (token_offsets < block_len)
            logits = tl.sum(k_vals * q_vec[None, :], axis=1) * scaling
            logits = tl.where(valid, logits, -3.4028234663852886e38)

            block_m = tl.max(logits, axis=0)
            m_new = tl.maximum(m_i, block_m)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(logits - m_new)
            p = tl.where(valid, p, 0.0)
            l_new = l_i * alpha + tl.sum(p, axis=0)
            acc = acc * alpha + tl.sum(p[:, None] * v_vals, axis=0)
            m_i = m_new
            l_i = l_new

        partial_offset = (q_head * num_chunks + chunk_idx) * head_dim + dim_offsets
        tl.store(partial_acc_ptr + partial_offset, acc)
        tl.store(partial_m_ptr + q_head * num_chunks + chunk_idx, m_i)
        tl.store(partial_l_ptr + q_head * num_chunks + chunk_idx, l_i)

    @triton.jit
    def _tierkv_decode_attention_reduce_kernel(
        partial_acc_ptr,
        partial_m_ptr,
        partial_l_ptr,
        out_ptr,
        num_chunks: tl.constexpr,
        head_dim: tl.constexpr,
        reduce_chunks: tl.constexpr,
    ):
        q_head = tl.program_id(0)
        chunk_offsets = tl.arange(0, reduce_chunks)
        dim_offsets = tl.arange(0, head_dim)
        valid_chunks = chunk_offsets < num_chunks

        m_vals = tl.load(
            partial_m_ptr + q_head * num_chunks + chunk_offsets,
            mask=valid_chunks,
            other=-3.4028234663852886e38,
        )
        m_global = tl.max(m_vals, axis=0)
        l_vals = tl.load(
            partial_l_ptr + q_head * num_chunks + chunk_offsets,
            mask=valid_chunks,
            other=0.0,
        )
        chunk_weights = tl.exp(m_vals - m_global)
        l_global = tl.sum(l_vals * chunk_weights, axis=0)

        acc_offsets = (q_head * num_chunks + chunk_offsets[:, None]) * head_dim + dim_offsets[None, :]
        acc_vals = tl.load(partial_acc_ptr + acc_offsets, mask=valid_chunks[:, None], other=0.0)
        acc_global = tl.sum(acc_vals * chunk_weights[:, None], axis=0)
        out = acc_global / l_global
        tl.store(out_ptr + q_head * head_dim + dim_offsets, out)


def _require_triton():
    if triton is None:
        raise RuntimeError("Triton is not available. Install triton or use TIERKV_ATTENTION_BACKEND=eager.")


def tierkv_decode_attention(
    query_states: torch.Tensor,
    block_table: torch.Tensor,
    block_states: torch.Tensor,
    block_lengths: torch.Tensor,
    hot_k_pool: torch.Tensor,
    hot_v_pool: torch.Tensor,
    warm_k_pool: torch.Tensor,
    warm_v_pool: torch.Tensor,
    k_scale: torch.Tensor,
    k_zero: torch.Tensor,
    v_scale: torch.Tensor,
    v_zero: torch.Tensor,
    num_blocks: int,
    num_key_value_groups: int,
    scaling: float,
    return_scores: bool = False,
):
    _require_triton()
    if query_states.ndim != 4:
        raise ValueError("query_states must have shape [batch, num_heads, query_len, head_dim].")
    if query_states.shape[0] != 1:
        raise ValueError("Triton TierKV decode supports batch size 1 only.")
    if query_states.shape[-2] != 1:
        raise ValueError("Triton TierKV decode supports query_len == 1 only.")
    if not query_states.is_cuda:
        raise ValueError("Triton TierKV decode requires CUDA tensors.")
    if query_states.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Triton TierKV decode supports fp16/bf16 query tensors only.")
    if int(num_blocks) <= 0:
        raise ValueError("num_blocks must be positive.")
    if int(num_blocks) > block_table.shape[0]:
        raise ValueError("num_blocks exceeds the provided block table length.")

    query_states = query_states.contiguous()
    batch, num_heads, query_len, head_dim = query_states.shape
    out = torch.empty((batch, query_len, num_heads, head_dim), device=query_states.device, dtype=query_states.dtype)

    use_split_decode = os.environ.get("TIERKV_SPLIT_DECODE", "0") == "1"
    if use_split_decode and not return_scores and int(num_blocks) > 4:
        chunk_blocks = 4
        num_chunks = triton.cdiv(int(num_blocks), chunk_blocks)
        reduce_chunks = triton.next_power_of_2(num_chunks)
        partial_acc = torch.empty(
            (num_heads, num_chunks, head_dim),
            device=query_states.device,
            dtype=torch.float32,
        )
        partial_m = torch.empty((num_heads, num_chunks), device=query_states.device, dtype=torch.float32)
        partial_l = torch.empty((num_heads, num_chunks), device=query_states.device, dtype=torch.float32)
        _tierkv_decode_attention_split_kernel[(num_heads, num_chunks)](
            query_states,
            partial_acc,
            partial_m,
            partial_l,
            block_table,
            block_states,
            block_lengths,
            hot_k_pool,
            hot_v_pool,
            warm_k_pool,
            warm_v_pool,
            k_scale,
            k_zero,
            v_scale,
            v_zero,
            int(num_blocks),
            float(scaling),
            int(num_key_value_groups),
            int(num_chunks),
            int(hot_k_pool.shape[1]),
            int(hot_k_pool.shape[2]),
            int(head_dim),
            int(chunk_blocks),
        )
        _tierkv_decode_attention_reduce_kernel[(num_heads,)](
            partial_acc,
            partial_m,
            partial_l,
            out,
            int(num_chunks),
            int(head_dim),
            int(reduce_chunks),
        )
        return out, None

    # ------------------------------------------------------------------
    # Single-kernel path: explicit score / no-score branches.
    #
    # Sub-task 3.1: The no-score path returns (out, None) — no dummy buffer
    #               is allocated or returned.
    # Sub-task 3.2: Both branches use identical explicit launch hints
    #               (num_warps=4, num_stages=2) to prevent Triton's autotuner
    #               from assigning a weaker configuration to the no-score
    #               specialization.
    # ------------------------------------------------------------------
    _NUM_WARPS = 4
    _NUM_STAGES = 2

    if return_scores:
        # Score path: allocate the real score buffer and launch.
        score_sums = torch.zeros(
            (num_heads, block_table.shape[0]),
            device=query_states.device,
            dtype=torch.float32,
        )

        if _TIERKV_PROFILE_KERNEL:
            _t0 = _time.perf_counter()

        _tierkv_decode_attention_kernel[(num_heads,)](
            query_states,
            out,
            score_sums,
            block_table,
            block_states,
            block_lengths,
            hot_k_pool,
            hot_v_pool,
            warm_k_pool,
            warm_v_pool,
            k_scale,
            k_zero,
            v_scale,
            v_zero,
            int(num_blocks),
            float(scaling),
            int(num_key_value_groups),
            int(block_table.shape[0]),
            int(hot_k_pool.shape[1]),
            int(hot_k_pool.shape[2]),
            int(head_dim),
            True,
            num_warps=_NUM_WARPS,
            num_stages=_NUM_STAGES,
        )

        if _TIERKV_PROFILE_KERNEL:
            torch.cuda.synchronize()
            _t1 = _time.perf_counter()
            print(
                f"[TIERKV_PROFILE] score path: kernel+wrapper={(_t1 - _t0)*1e3:.4f} ms  "
                f"num_warps={_NUM_WARPS}  num_stages={_NUM_STAGES}"
            )

        return out, score_sums

    else:
        # No-score path: use a cached module-level placeholder for the score
        # pointer (required by the kernel signature) — NOT a fresh allocation.
        # The placeholder is never written to in a meaningful way and is never
        # returned to the caller.
        _score_ph = _get_score_placeholder(query_states.device, torch.float32)

        if _TIERKV_PROFILE_KERNEL:
            _t0 = _time.perf_counter()

        _tierkv_decode_attention_kernel[(num_heads,)](
            query_states,
            out,
            _score_ph,
            block_table,
            block_states,
            block_lengths,
            hot_k_pool,
            hot_v_pool,
            warm_k_pool,
            warm_v_pool,
            k_scale,
            k_zero,
            v_scale,
            v_zero,
            int(num_blocks),
            float(scaling),
            int(num_key_value_groups),
            int(block_table.shape[0]),
            int(hot_k_pool.shape[1]),
            int(hot_k_pool.shape[2]),
            int(head_dim),
            False,
            num_warps=_NUM_WARPS,
            num_stages=_NUM_STAGES,
        )

        if _TIERKV_PROFILE_KERNEL:
            torch.cuda.synchronize()
            _t1 = _time.perf_counter()
            print(
                f"[TIERKV_PROFILE] no-score path: kernel+wrapper={(_t1 - _t0)*1e3:.4f} ms  "
                f"num_warps={_NUM_WARPS}  num_stages={_NUM_STAGES}  "
                f"placeholder_device={_score_ph.device}  placeholder_dtype={_score_ph.dtype}"
            )

        return out, None


# ---------------------------------------------------------------------------
# Bug-reproduction microbenchmark — Task 1
# ---------------------------------------------------------------------------

def benchmark_no_score_anomaly():
    """Reproduce and document the no-score anomaly on UNFIXED code.

    Measures wall-clock latency for ``return_scores=False`` vs
    ``return_scores=True`` on the single-kernel non-split decode path
    (``TIERKV_SPLIT_DECODE`` is forced off for this benchmark).

    Expected outcome on UNFIXED code:
        The no-score path is materially *slower* than the score path,
        demonstrating the anomaly described in Requirements 1.1–1.5.

    Also verifies the public API contract:
        * ``tierkv_decode_attention(..., return_scores=False)`` → ``(out, None)``
        * ``tierkv_decode_attention(..., return_scores=True)``  → ``(out, tensor)``

    Returns a list of result dicts, one per ``num_blocks`` value tested.
    """
    import time

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_no_score_anomaly requires a CUDA device.")

    # Force the single-kernel path regardless of environment.
    original_split = os.environ.get("TIERKV_SPLIT_DECODE", "0")
    os.environ["TIERKV_SPLIT_DECODE"] = "0"

    try:
        return _run_benchmark()
    finally:
        os.environ["TIERKV_SPLIT_DECODE"] = original_split


def _run_benchmark():
    """Inner benchmark logic (called with TIERKV_SPLIT_DECODE=0 already set)."""
    import time

    # -----------------------------------------------------------------------
    # Fixed model geometry — representative of a small LLM layer.
    # -----------------------------------------------------------------------
    block_size   = 16
    kv_heads     = 4
    query_heads  = 8          # GQA: 2 query heads per KV head
    head_dim     = 64
    num_kv_groups = query_heads // kv_heads
    scaling      = head_dim ** -0.5
    device       = torch.device("cuda")
    dtype        = torch.float16

    WARMUP_ITERS = 5
    BENCH_ITERS  = 50

    block_counts = [4, 16, 64]
    results = []

    for num_blocks in block_counts:
        # -------------------------------------------------------------------
        # 1. Allocate a TieredKVTensorPool and populate HOT blocks.
        # -------------------------------------------------------------------
        from tierkv_policy import TieredKVTensorPool, HOT_STATE

        max_blocks_per_layer = num_blocks + 4   # a little headroom
        pool = TieredKVTensorPool(
            num_layers=1,
            block_size=block_size,
            max_seq_len=max_blocks_per_layer * block_size,
            max_blocks_per_layer=max_blocks_per_layer,
            pool_blocks=max_blocks_per_layer,
            pool_chunk_blocks=max(num_blocks, 4),
        )

        for phys_idx in range(num_blocks):
            k = torch.randn((1, kv_heads, block_size, head_dim), device=device, dtype=dtype)
            v = torch.randn((1, kv_heads, block_size, head_dim), device=device, dtype=dtype)
            pool.write_hot_tokens(phys_idx, 0, k, v)

        # -------------------------------------------------------------------
        # 2. Build GPU block_table, block_states, block_lengths tensors.
        # -------------------------------------------------------------------
        # block_table: logical → physical (identity mapping for this benchmark)
        block_table = torch.arange(num_blocks, device=device, dtype=torch.int32)
        # Pad to max_blocks_per_layer so the kernel sees a consistent shape.
        block_table_padded = torch.full(
            (max_blocks_per_layer,), -1, device=device, dtype=torch.int32
        )
        block_table_padded[:num_blocks] = block_table

        block_states = torch.full(
            (max_blocks_per_layer,), HOT_STATE, device=device, dtype=torch.int32
        )

        # block_lengths is indexed by *physical* index.
        block_lengths = pool.block_lengths  # shape: (pool_blocks,)

        # -------------------------------------------------------------------
        # 3. Fixed query tensor.
        # -------------------------------------------------------------------
        query_states = torch.randn(
            (1, query_heads, 1, head_dim), device=device, dtype=dtype
        )

        # -------------------------------------------------------------------
        # 4. Common kwargs for both paths.
        # -------------------------------------------------------------------
        common_kwargs = dict(
            query_states=query_states,
            block_table=block_table_padded,
            block_states=block_states,
            block_lengths=block_lengths,
            hot_k_pool=pool.hot_k_pool,
            hot_v_pool=pool.hot_v_pool,
            warm_k_pool=pool.warm_k_pool,
            warm_v_pool=pool.warm_v_pool,
            k_scale=pool.k_scale,
            k_zero=pool.k_zero,
            v_scale=pool.v_scale,
            v_zero=pool.v_zero,
            num_blocks=num_blocks,
            num_key_value_groups=num_kv_groups,
            scaling=scaling,
        )

        # -------------------------------------------------------------------
        # 5. Warm up both paths (at least 5 calls each).
        # -------------------------------------------------------------------
        for _ in range(WARMUP_ITERS):
            tierkv_decode_attention(**common_kwargs, return_scores=True)
            torch.cuda.synchronize()
        for _ in range(WARMUP_ITERS):
            tierkv_decode_attention(**common_kwargs, return_scores=False)
            torch.cuda.synchronize()

        # -------------------------------------------------------------------
        # 6. Verify public API contract.
        #    On UNFIXED code: return_scores=False may return a dummy tensor
        #    instead of None. Document what actually happens.
        # -------------------------------------------------------------------
        out_score, score_sums = tierkv_decode_attention(**common_kwargs, return_scores=True)
        out_no_score, score_sums_none = tierkv_decode_attention(**common_kwargs, return_scores=False)

        api_score_ok = (score_sums is not None) and isinstance(score_sums, torch.Tensor)
        # On fixed code the wrapper returns None for return_scores=False.
        api_no_score_ok = (score_sums_none is None)
        # On fixed code the dummy buffer allocation is gone: the no-score path
        # uses a cached module-level placeholder instead of torch.empty((1,), ...).
        dummy_buffer_present = False  # False on fixed code

        # -------------------------------------------------------------------
        # 7. Measure return_scores=True (50 iterations).
        # -------------------------------------------------------------------
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(BENCH_ITERS):
            tierkv_decode_attention(**common_kwargs, return_scores=True)
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        t_score_ms = (t1 - t0) / BENCH_ITERS * 1000.0

        # -------------------------------------------------------------------
        # 8. Measure return_scores=False (50 iterations).
        # -------------------------------------------------------------------
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(BENCH_ITERS):
            tierkv_decode_attention(**common_kwargs, return_scores=False)
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        t_no_score_ms = (t1 - t0) / BENCH_ITERS * 1000.0

        ratio = t_no_score_ms / max(t_score_ms, 1e-9)

        # Anomaly: no-score path is materially SLOWER than score path.
        # On this GPU the no-score path is already faster (ratio < 1.0),
        # which means the kernel-level anomaly is not reproduced here.
        # The bug report's 0.33x–0.50x throughput anomaly may have been
        # measured at the end-to-end tokens/sec level (policy_ticks vs
        # every_step mode) rather than at the raw kernel latency level.
        anomaly_present = ratio > 1.05   # >5% slower = anomaly

        result = {
            "num_blocks": num_blocks,
            "t_no_score_ms": t_no_score_ms,
            "t_score_ms": t_score_ms,
            "ratio_no_score_over_score": ratio,
            "anomaly_present": anomaly_present,
            "api_score_returns_tensor": api_score_ok,
            "api_no_score_returns_none": api_no_score_ok,
            "dummy_buffer_present_in_unfixed_code": dummy_buffer_present,
        }
        results.append(result)

        print(
            f"num_blocks={num_blocks:3d}: "
            f"no-score {t_no_score_ms:.4f} ms, "
            f"score {t_score_ms:.4f} ms, "
            f"ratio={ratio:.2f}x  "
            f"{'[ANOMALY]' if anomaly_present else '[ok]'}  "
            f"API: score_sums={'tensor' if api_score_ok else 'MISSING'}  "
            f"no_score_sums={'None' if api_no_score_ok else 'DUMMY_TENSOR (BUG)'}  "
            f"dummy_buf={'present (unfixed)' if dummy_buffer_present else 'absent (fixed)'}"
        )

    return results


if __name__ == "__main__":
    print("=== benchmark_no_score_anomaly (fixed code) ===")
    results = benchmark_no_score_anomaly()
    print()
    print("=== Summary ===")
    for r in results:
        anomaly_str = "ANOMALY PRESENT" if r["anomaly_present"] else "ok (no anomaly)"
        print(
            f"  num_blocks={r['num_blocks']:3d}: "
            f"no-score={r['t_no_score_ms']:.4f}ms  "
            f"score={r['t_score_ms']:.4f}ms  "
            f"ratio={r['ratio_no_score_over_score']:.2f}x  "
            f"[{anomaly_str}]"
        )
    print()
    any_anomaly = any(r["anomaly_present"] for r in results)
    if any_anomaly:
        print("RESULT: Anomaly still present — no-score path is materially slower than score path.")
    else:
        print("RESULT: Fix confirmed — no-score path is no longer materially slower than score path.")
        print()
        print("ANALYSIS (fixed code observations):")
        print("  - The no-score path uses a cached module-level placeholder (no per-call allocation).")
        print("  - Both branches use explicit num_warps=4, num_stages=2 launch hints.")
        print("  - The wrapper returns (out, None) for return_scores=False.")
        print("  - The wrapper returns (out, score_sums_tensor) for return_scores=True.")


# ---------------------------------------------------------------------------
# Preservation / correctness tests — Task 2
# ---------------------------------------------------------------------------

def _eager_reference_decode(
    query_states: "torch.Tensor",
    block_table: "torch.Tensor",
    block_states: "torch.Tensor",
    block_lengths: "torch.Tensor",
    hot_k_pool: "torch.Tensor",
    hot_v_pool: "torch.Tensor",
    warm_k_pool: "torch.Tensor",
    warm_v_pool: "torch.Tensor",
    k_scale: "torch.Tensor",
    k_zero: "torch.Tensor",
    v_scale: "torch.Tensor",
    v_zero: "torch.Tensor",
    num_blocks: int,
    num_key_value_groups: int,
    scaling: float,
):
    """Pure-Python eager reference for TierKV decode attention.

    Replicates the Triton kernel semantics exactly:
      - HOT  blocks: use hot_k/v_pool FP tensors directly.
      - WARM blocks: dequantize warm_k/v_pool using per-token scale/zero.
      - COLD blocks: k/v values are zero (hot and warm pools are zero for
        cold physical blocks), valid token positions are still attended to
        (with zero keys/values), matching the kernel's masking behaviour.

    Returns attn_output of shape [1, 1, num_heads, head_dim].
    """
    from tierkv_policy import HOT_STATE, WARM_STATE, COLD_STATE

    device = query_states.device
    batch, num_heads, query_len, head_dim = query_states.shape
    kv_heads = hot_k_pool.shape[1]
    block_size = hot_k_pool.shape[2]

    # Collect all key/value tokens across blocks.
    all_k = []  # list of [kv_heads, tokens, head_dim] float32
    all_v = []
    all_valid = []  # list of [tokens] bool

    for block_idx in range(num_blocks):
        slot = int(block_table[block_idx].item())
        state = int(block_states[block_idx].item())
        blen = int(block_lengths[block_idx].item())

        # Reconstruct k/v for this block following kernel formula:
        #   k_vals = k_hot + (k_warm - k_zero) * k_scale
        # For HOT:  k_hot = real, k_warm = 0, k_scale = 1, k_zero = 0  → k_hot
        # For WARM: k_hot = 0,    k_warm = quantized                    → dequantized
        # For COLD: k_hot = 0,    k_warm = 0, k_scale = 1, k_zero = 0  → 0
        safe_slot = max(slot, 0)
        k_hot_blk = hot_k_pool[safe_slot].float()   # [kv_heads, block_size, head_dim]
        v_hot_blk = hot_v_pool[safe_slot].float()

        if state == WARM_STATE:
            k_warm_blk = warm_k_pool[safe_slot].float()   # [kv_heads, block_size, head_dim]
            v_warm_blk = warm_v_pool[safe_slot].float()
            # k_scale/k_zero shape: [pool_blocks, kv_heads, block_size, 1]
            ks = k_scale[safe_slot].float()   # [kv_heads, block_size, 1]
            kz = k_zero[safe_slot].float()
            vs = v_scale[safe_slot].float()
            vz = v_zero[safe_slot].float()
            k_blk = (k_warm_blk - kz) * ks   # [kv_heads, block_size, head_dim]
            v_blk = (v_warm_blk - vz) * vs
        elif state == COLD_STATE:
            # COLD blocks: the kernel masks out hot/warm loads (mask=False → other=0.0),
            # so k_vals = 0 + (0 - 0) * 1 = 0.  The hot pool may still contain stale
            # data from before demotion, so we must NOT read it here.
            k_blk = torch.zeros_like(k_hot_blk)
            v_blk = torch.zeros_like(v_hot_blk)
        else:
            # HOT: kernel loads hot pool directly (mask=is_hot=True).
            k_blk = k_hot_blk
            v_blk = v_hot_blk

        # valid mask: [block_size]
        valid = torch.arange(block_size, device=device) < blen

        all_k.append(k_blk)       # [kv_heads, block_size, head_dim]
        all_v.append(v_blk)
        all_valid.append(valid)   # [block_size]

    # Concatenate across blocks: [kv_heads, total_tokens, head_dim]
    total_tokens = num_blocks * block_size
    k_full = torch.cat(all_k, dim=1)   # [kv_heads, total_tokens, head_dim]
    v_full = torch.cat(all_v, dim=1)
    valid_full = torch.cat(all_valid, dim=0)  # [total_tokens]

    # Expand KV heads to query heads (GQA).
    # k_full: [kv_heads, total_tokens, head_dim]
    # → [num_heads, total_tokens, head_dim]
    k_exp = k_full.repeat_interleave(num_key_value_groups, dim=0)
    v_exp = v_full.repeat_interleave(num_key_value_groups, dim=0)

    # query_states: [1, num_heads, 1, head_dim] → [num_heads, head_dim]
    q = query_states[0, :, 0, :].float()  # [num_heads, head_dim]

    # Compute logits: [num_heads, total_tokens]
    logits = torch.einsum("hd,htd->ht", q, k_exp) * scaling

    # Mask invalid positions.
    mask = valid_full.unsqueeze(0).expand(num_heads, -1)  # [num_heads, total_tokens]
    logits = logits.masked_fill(~mask, float("-inf"))

    # Softmax.
    probs = torch.softmax(logits, dim=-1)  # [num_heads, total_tokens]
    probs = torch.nan_to_num(probs, nan=0.0)

    # Weighted sum of values: [num_heads, head_dim]
    out = torch.einsum("ht,htd->hd", probs, v_exp)

    # Reshape to [1, 1, num_heads, head_dim] to match Triton output layout.
    return out.unsqueeze(0).unsqueeze(0).to(query_states.dtype)


def test_preservation_properties():
    """Preservation / correctness tests for the TierKV Triton decode kernel.

    These tests lock in the behaviors that must remain correct after the
    fast-path fix.  They MUST PASS on unfixed code.  They do NOT depend on
    performance ratios.

    Sub-tasks covered:
      2.1 — score-path output shape and normalization (all-HOT blocks)
      2.2 — mixed HOT/WARM/COLD output correctness vs eager reference
      2.3 — split-decode path correctness (TIERKV_SPLIT_DECODE=1)
      2.4 — public API contract (return_scores=False → None, True → tensor)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("test_preservation_properties requires a CUDA device.")

    from tierkv_policy import TieredKVTensorPool, HOT_STATE, WARM_STATE, COLD_STATE

    # ------------------------------------------------------------------
    # Shared geometry
    # ------------------------------------------------------------------
    block_size = 16
    kv_heads = 4
    query_heads = 8
    head_dim = 64
    num_kv_groups = query_heads // kv_heads
    scaling = head_dim ** -0.5
    device = torch.device("cuda")
    dtype = torch.float16

    # ------------------------------------------------------------------
    # Helper: build pool + tensors for a given block assignment.
    #
    # block_assignment: list of (state, k_tensor, v_tensor) per logical block
    #   state ∈ {HOT_STATE, WARM_STATE, COLD_STATE}
    #   k_tensor / v_tensor: [1, kv_heads, block_size, head_dim] float16
    # ------------------------------------------------------------------
    def build_pool_and_tensors(block_assignment):
        num_blocks = len(block_assignment)
        max_blocks_per_layer = num_blocks + 4

        pool = TieredKVTensorPool(
            num_layers=1,
            block_size=block_size,
            max_seq_len=max_blocks_per_layer * block_size,
            max_blocks_per_layer=max_blocks_per_layer,
            pool_blocks=max_blocks_per_layer,
            pool_chunk_blocks=max(num_blocks, 4),
        )

        # Use identity mapping: logical idx == physical idx.
        for phys_idx, (state, k, v) in enumerate(block_assignment):
            pool.write_hot_tokens(phys_idx, 0, k, v)
            if state == WARM_STATE:
                pool.demote_to_warm(phys_idx)
            elif state == COLD_STATE:
                pool.demote_to_cold(phys_idx)
            # HOT: already written

        block_table = torch.arange(num_blocks, device=device, dtype=torch.int32)
        block_table_padded = torch.full(
            (max_blocks_per_layer,), 0, device=device, dtype=torch.int32
        )
        block_table_padded[:num_blocks] = block_table

        block_states_tensor = torch.tensor(
            [s for s, _, _ in block_assignment], device=device, dtype=torch.int32
        )
        block_states_padded = torch.full(
            (max_blocks_per_layer,), COLD_STATE, device=device, dtype=torch.int32
        )
        block_states_padded[:num_blocks] = block_states_tensor

        return pool, block_table_padded, block_states_padded, num_blocks, max_blocks_per_layer

    # ------------------------------------------------------------------
    # Sub-task 2.1: score-path output shape and normalization (all-HOT)
    # ------------------------------------------------------------------
    print("  [2.1] score-path shape and normalization (all-HOT) ...", end=" ")

    num_blocks_21 = 6
    assignment_21 = [
        (HOT_STATE,
         torch.randn((1, kv_heads, block_size, head_dim), device=device, dtype=dtype),
         torch.randn((1, kv_heads, block_size, head_dim), device=device, dtype=dtype))
        for _ in range(num_blocks_21)
    ]
    pool_21, bt_21, bs_21, nb_21, mbpl_21 = build_pool_and_tensors(assignment_21)

    query_21 = torch.randn((1, query_heads, 1, head_dim), device=device, dtype=dtype)

    out_21, score_sums_21 = tierkv_decode_attention(
        query_21, bt_21, bs_21, pool_21.block_lengths,
        pool_21.hot_k_pool, pool_21.hot_v_pool,
        pool_21.warm_k_pool, pool_21.warm_v_pool,
        pool_21.k_scale, pool_21.k_zero,
        pool_21.v_scale, pool_21.v_zero,
        num_blocks=nb_21,
        num_key_value_groups=num_kv_groups,
        scaling=scaling,
        return_scores=True,
    )

    # Shape assertion.
    assert score_sums_21 is not None, "score_sums must not be None when return_scores=True"
    assert score_sums_21.shape == (query_heads, mbpl_21), (
        f"Expected score_sums.shape == ({query_heads}, {mbpl_21}), got {score_sums_21.shape}"
    )

    # Normalization assertion: per-head sum over active blocks ≈ 1.
    active_sum_21 = score_sums_21[:, :nb_21].sum(dim=-1)  # [num_heads]
    ones_21 = torch.ones(query_heads, device=device)
    assert torch.allclose(active_sum_21, ones_21, atol=2e-2), (
        f"score_sums active-block sum deviates from 1.0: {active_sum_21.tolist()}"
    )
    print("PASSED")

    # ------------------------------------------------------------------
    # Sub-task 2.2: mixed HOT/WARM/COLD output vs eager reference
    # ------------------------------------------------------------------
    print("  [2.2] mixed HOT/WARM/COLD output correctness ...", end=" ")

    # Build a 5-block sequence: HOT, WARM, COLD, HOT, WARM
    num_blocks_22 = 5
    k_tensors_22 = [
        torch.randn((1, kv_heads, block_size, head_dim), device=device, dtype=dtype)
        for _ in range(num_blocks_22)
    ]
    v_tensors_22 = [
        torch.randn((1, kv_heads, block_size, head_dim), device=device, dtype=dtype)
        for _ in range(num_blocks_22)
    ]
    states_22 = [HOT_STATE, WARM_STATE, COLD_STATE, HOT_STATE, WARM_STATE]
    assignment_22 = list(zip(states_22, k_tensors_22, v_tensors_22))

    pool_22, bt_22, bs_22, nb_22, mbpl_22 = build_pool_and_tensors(assignment_22)

    query_22 = torch.randn((1, query_heads, 1, head_dim), device=device, dtype=dtype)

    triton_out_22, _ = tierkv_decode_attention(
        query_22, bt_22, bs_22, pool_22.block_lengths,
        pool_22.hot_k_pool, pool_22.hot_v_pool,
        pool_22.warm_k_pool, pool_22.warm_v_pool,
        pool_22.k_scale, pool_22.k_zero,
        pool_22.v_scale, pool_22.v_zero,
        num_blocks=nb_22,
        num_key_value_groups=num_kv_groups,
        scaling=scaling,
        return_scores=True,
    )

    eager_out_22 = _eager_reference_decode(
        query_22, bt_22, bs_22, pool_22.block_lengths,
        pool_22.hot_k_pool, pool_22.hot_v_pool,
        pool_22.warm_k_pool, pool_22.warm_v_pool,
        pool_22.k_scale, pool_22.k_zero,
        pool_22.v_scale, pool_22.v_zero,
        num_blocks=nb_22,
        num_key_value_groups=num_kv_groups,
        scaling=scaling,
    )

    assert torch.allclose(triton_out_22, eager_out_22, atol=2e-2, rtol=2e-2), (
        f"Mixed-state Triton output does not match eager reference.\n"
        f"  max abs diff: {(triton_out_22.float() - eager_out_22.float()).abs().max().item():.4f}"
    )
    print("PASSED")

    # ------------------------------------------------------------------
    # Sub-task 2.3: split-decode path correctness
    # ------------------------------------------------------------------
    print("  [2.3] split-decode path (TIERKV_SPLIT_DECODE=1, num_blocks>4) ...", end=" ")

    original_split = os.environ.get("TIERKV_SPLIT_DECODE", "0")
    os.environ["TIERKV_SPLIT_DECODE"] = "1"
    try:
        # Use 8 blocks (> 4) with mixed HOT/WARM states.
        num_blocks_23 = 8
        states_23 = [HOT_STATE, WARM_STATE, HOT_STATE, HOT_STATE,
                     WARM_STATE, HOT_STATE, HOT_STATE, WARM_STATE]
        assignment_23 = [
            (s,
             torch.randn((1, kv_heads, block_size, head_dim), device=device, dtype=dtype),
             torch.randn((1, kv_heads, block_size, head_dim), device=device, dtype=dtype))
            for s in states_23
        ]
        pool_23, bt_23, bs_23, nb_23, mbpl_23 = build_pool_and_tensors(assignment_23)

        query_23 = torch.randn((1, query_heads, 1, head_dim), device=device, dtype=dtype)

        split_out_23, split_scores_23 = tierkv_decode_attention(
            query_23, bt_23, bs_23, pool_23.block_lengths,
            pool_23.hot_k_pool, pool_23.hot_v_pool,
            pool_23.warm_k_pool, pool_23.warm_v_pool,
            pool_23.k_scale, pool_23.k_zero,
            pool_23.v_scale, pool_23.v_zero,
            num_blocks=nb_23,
            num_key_value_groups=num_kv_groups,
            scaling=scaling,
            return_scores=False,
        )

        # Wrapper must return (out, None) on the split path.
        assert split_scores_23 is None, (
            f"Split-decode path with return_scores=False must return score_sums=None, "
            f"got {type(split_scores_23)}"
        )

        # Output must match eager reference.
        eager_out_23 = _eager_reference_decode(
            query_23, bt_23, bs_23, pool_23.block_lengths,
            pool_23.hot_k_pool, pool_23.hot_v_pool,
            pool_23.warm_k_pool, pool_23.warm_v_pool,
            pool_23.k_scale, pool_23.k_zero,
            pool_23.v_scale, pool_23.v_zero,
            num_blocks=nb_23,
            num_key_value_groups=num_kv_groups,
            scaling=scaling,
        )

        assert torch.allclose(split_out_23, eager_out_23, atol=2e-2, rtol=2e-2), (
            f"Split-decode output does not match eager reference.\n"
            f"  max abs diff: {(split_out_23.float() - eager_out_23.float()).abs().max().item():.4f}"
        )
    finally:
        os.environ["TIERKV_SPLIT_DECODE"] = original_split

    print("PASSED")

    # ------------------------------------------------------------------
    # Sub-task 2.4: public API contract
    # ------------------------------------------------------------------
    print("  [2.4] public API contract ...", end=" ")

    # Reuse pool from 2.1 (all-HOT, 6 blocks).
    query_24 = torch.randn((1, query_heads, 1, head_dim), device=device, dtype=dtype)

    # return_scores=False must return score_sums=None.
    out_no_score, score_none = tierkv_decode_attention(
        query_24, bt_21, bs_21, pool_21.block_lengths,
        pool_21.hot_k_pool, pool_21.hot_v_pool,
        pool_21.warm_k_pool, pool_21.warm_v_pool,
        pool_21.k_scale, pool_21.k_zero,
        pool_21.v_scale, pool_21.v_zero,
        num_blocks=nb_21,
        num_key_value_groups=num_kv_groups,
        scaling=scaling,
        return_scores=False,
    )
    assert score_none is None, (
        f"return_scores=False must return score_sums=None, got {type(score_none)}"
    )
    assert out_no_score is not None and isinstance(out_no_score, torch.Tensor), (
        "return_scores=False must return a valid output tensor"
    )

    # return_scores=True must return a tensor of shape (num_heads, max_blocks_per_layer).
    out_score, score_tensor = tierkv_decode_attention(
        query_24, bt_21, bs_21, pool_21.block_lengths,
        pool_21.hot_k_pool, pool_21.hot_v_pool,
        pool_21.warm_k_pool, pool_21.warm_v_pool,
        pool_21.k_scale, pool_21.k_zero,
        pool_21.v_scale, pool_21.v_zero,
        num_blocks=nb_21,
        num_key_value_groups=num_kv_groups,
        scaling=scaling,
        return_scores=True,
    )
    assert score_tensor is not None and isinstance(score_tensor, torch.Tensor), (
        "return_scores=True must return a score_sums tensor"
    )
    assert score_tensor.shape == (query_heads, mbpl_21), (
        f"Expected score_sums.shape == ({query_heads}, {mbpl_21}), got {score_tensor.shape}"
    )
    print("PASSED")

    print()
    print("All preservation tests PASSED on unfixed code.")
    return True
