from __future__ import annotations

import os
from dataclasses import dataclass

import torch

try:
    from transformers.cache_utils import Cache as HFCache
except Exception:
    HFCache = object


HOT_STATE = 0
WARM_STATE = 1
COLD_STATE = 2

_GLOBAL_TIERKV_RUNTIME = None


def quantize_fp16_to_int8(tensor: torch.Tensor):
    if tensor.dtype != torch.float16:
        tensor = tensor.to(torch.float16)

    min_val = tensor.amin(dim=-1, keepdim=True)
    max_val = tensor.amax(dim=-1, keepdim=True)
    scale = (max_val - min_val) / 255.0
    scale = torch.where(
        scale == 0,
        torch.ones_like(scale, device=tensor.device, dtype=torch.float16),
        scale.to(torch.float16),
    ).to(torch.float16)
    zero_point = (-torch.round(min_val / scale)).to(torch.float16)
    quantized = torch.clamp(torch.round(tensor / scale) + zero_point, 0, 255).to(torch.uint8)

    return quantized, scale, zero_point


def dequantize_int8_to_fp16(
    quantized_tensor: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
):
    return ((quantized_tensor.to(torch.float16) - zero_point) * scale).to(torch.float16)


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def estimate_pool_blocks(num_layers: int, block_size: int, max_seq_len: int) -> int:
    return int(num_layers) * _ceil_div(max_seq_len, block_size)


class TieredKVTensorPool:
    def __init__(
        self,
        num_layers: int = 1,
        block_size: int = 16,
        max_seq_len: int = 2048,
        max_blocks_per_layer: int | None = None,
        pool_blocks: int | None = None,
        pool_chunk_blocks: int = 128,
    ):
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if block_size <= 0:
            raise ValueError("block_size must be positive.")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")

        self.num_layers = int(num_layers)
        self.block_size = int(block_size)
        self.max_seq_len = int(max_seq_len)
        self.max_blocks_per_layer = int(max_blocks_per_layer or _ceil_div(max_seq_len, block_size))
        self.pool_blocks = int(pool_blocks or self.num_layers * self.max_blocks_per_layer)
        self.pool_chunk_blocks = int(pool_chunk_blocks)
        if self.pool_chunk_blocks <= 0:
            raise ValueError("pool_chunk_blocks must be positive.")

        self.device = None
        self.dtype = torch.float16
        self.kv_heads = None
        self.head_dim = None
        self.allocated_pool_blocks = 0

        self.hot_k_pool = None
        self.hot_v_pool = None
        self.warm_k_pool = None
        self.warm_v_pool = None
        self.k_scale = None
        self.k_zero = None
        self.v_scale = None
        self.v_zero = None
        self.block_lengths = None
        self.block_tables = None
        self.block_states = None
        self.full_table_sync_count = 0

        self.physical_states: dict[int, int] = {}
        self.hot_storage = set()
        self.warm_storage = set()
        self.cold_storage = set()

    def is_allocated(self) -> bool:
        return self.hot_k_pool is not None

    def _target_capacity(self, physical_idx: int) -> int:
        required = int(physical_idx) + 1
        return min(self.pool_blocks, _ceil_div(required, self.pool_chunk_blocks) * self.pool_chunk_blocks)

    def _check_physical_idx(self, physical_idx: int) -> int:
        physical_idx = int(physical_idx)
        if physical_idx < 0 or physical_idx >= self.pool_blocks:
            raise RuntimeError(
                f"Physical block {physical_idx} is outside tensor-pool capacity {self.pool_blocks}."
            )
        return physical_idx

    def _new_zero_pool(self, capacity: int, kv_heads: int, head_dim: int, dtype):
        return torch.zeros(
            (capacity, kv_heads, self.block_size, head_dim),
            device=self.device,
            dtype=dtype,
        )

    def _new_meta_pool(self, capacity: int, kv_heads: int, fill_value: int | float):
        return torch.full(
            (capacity, kv_heads, self.block_size, 1),
            fill_value,
            device=self.device,
            dtype=torch.float16,
        )

    def _grow_tensor(self, old_tensor: torch.Tensor | None, new_tensor: torch.Tensor):
        if old_tensor is not None and self.allocated_pool_blocks > 0:
            new_tensor[: self.allocated_pool_blocks].copy_(old_tensor[: self.allocated_pool_blocks])
        return new_tensor

    def _grow_storage(self, required_physical_idx: int) -> None:
        if required_physical_idx < self.allocated_pool_blocks:
            return
        new_capacity = self._target_capacity(required_physical_idx)
        if new_capacity <= self.allocated_pool_blocks:
            return

        old_capacity = self.allocated_pool_blocks
        self.hot_k_pool = self._grow_tensor(
            self.hot_k_pool,
            self._new_zero_pool(new_capacity, self.kv_heads, self.head_dim, torch.float16),
        )
        self.hot_v_pool = self._grow_tensor(
            self.hot_v_pool,
            self._new_zero_pool(new_capacity, self.kv_heads, self.head_dim, torch.float16),
        )
        self.warm_k_pool = self._grow_tensor(
            self.warm_k_pool,
            self._new_zero_pool(new_capacity, self.kv_heads, self.head_dim, torch.uint8),
        )
        self.warm_v_pool = self._grow_tensor(
            self.warm_v_pool,
            self._new_zero_pool(new_capacity, self.kv_heads, self.head_dim, torch.uint8),
        )
        self.k_scale = self._grow_tensor(
            self.k_scale,
            self._new_meta_pool(new_capacity, self.kv_heads, 1),
        )
        self.k_zero = self._grow_tensor(
            self.k_zero,
            self._new_meta_pool(new_capacity, self.kv_heads, 0),
        )
        self.v_scale = self._grow_tensor(
            self.v_scale,
            self._new_meta_pool(new_capacity, self.kv_heads, 1),
        )
        self.v_zero = self._grow_tensor(
            self.v_zero,
            self._new_meta_pool(new_capacity, self.kv_heads, 0),
        )

        new_block_lengths = torch.zeros((new_capacity,), device=self.device, dtype=torch.int32)
        if self.block_lengths is not None and old_capacity > 0:
            new_block_lengths[:old_capacity].copy_(self.block_lengths[:old_capacity])
        self.block_lengths = new_block_lengths
        self.allocated_pool_blocks = new_capacity

    def _ensure_allocated(self, key_states: torch.Tensor, required_physical_idx: int) -> None:
        if key_states.ndim != 4:
            raise ValueError("KV tensors must have shape [1, kv_heads, tokens, head_dim].")
        if key_states.shape[0] != 1:
            raise ValueError("TieredKVTensorPool currently supports batch size 1 only.")

        device = key_states.device
        kv_heads = int(key_states.shape[1])
        head_dim = int(key_states.shape[-1])

        if self.is_allocated():
            if device != self.device or kv_heads != self.kv_heads or head_dim != self.head_dim:
                raise ValueError("KV tensor geometry changed after tensor-pool allocation.")
            self._grow_storage(required_physical_idx)
            return

        self.device = device
        self.kv_heads = kv_heads
        self.head_dim = head_dim

        table_shape = (self.num_layers, self.max_blocks_per_layer)

        self.block_tables = torch.full(table_shape, -1, device=device, dtype=torch.int32)
        self.block_states = torch.full(table_shape, COLD_STATE, device=device, dtype=torch.int32)
        self._grow_storage(required_physical_idx)

    def _coerce_kv(self, key_states: torch.Tensor, value_states: torch.Tensor):
        if key_states.shape != value_states.shape:
            raise ValueError("Key and value tensors must have identical shape.")
        if key_states.ndim != 4 or key_states.shape[0] != 1:
            raise ValueError("KV tensors must have shape [1, kv_heads, tokens, head_dim].")
        if key_states.shape[-2] > self.block_size:
            raise ValueError("A single stored block cannot exceed block_size tokens.")
        if key_states.dtype != torch.float16:
            key_states = key_states.to(torch.float16)
        if value_states.dtype != torch.float16:
            value_states = value_states.to(torch.float16)
        return key_states.contiguous(), value_states.contiguous()

    def get_physical_state(self, physical_idx: int) -> int | None:
        return self.physical_states.get(int(physical_idx))

    def get_block_length(self, physical_idx: int) -> int:
        self._check_physical_idx(physical_idx)
        if self.block_lengths is None or int(physical_idx) >= self.allocated_pool_blocks:
            return 0
        return int(self.block_lengths[int(physical_idx)].item())

    def write_hot_tokens(
        self,
        physical_idx: int,
        start: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        physical_idx = self._check_physical_idx(physical_idx)
        key_states, value_states = self._coerce_kv(key_states, value_states)
        self._ensure_allocated(key_states, physical_idx)

        start = int(start)
        tokens = int(key_states.shape[-2])
        if start < 0 or start + tokens > self.block_size:
            raise ValueError("KV write exceeds the physical block capacity.")

        self.hot_k_pool[physical_idx, :, start : start + tokens, :].copy_(key_states[0])
        self.hot_v_pool[physical_idx, :, start : start + tokens, :].copy_(value_states[0])
        if start + tokens < self.block_size:
            self.hot_k_pool[physical_idx, :, start + tokens :, :].zero_()
            self.hot_v_pool[physical_idx, :, start + tokens :, :].zero_()

        self.block_lengths[physical_idx] = start + tokens
        self.physical_states[physical_idx] = HOT_STATE
        self.hot_storage.add(physical_idx)
        self.warm_storage.discard(physical_idx)
        self.cold_storage.discard(physical_idx)

    def set_block_entry(self, layer_idx: int, logical_idx: int, physical_idx: int, state: int) -> None:
        if not self.is_allocated():
            raise RuntimeError("TieredKVTensorPool has not been allocated yet.")
        layer_idx = int(layer_idx)
        logical_idx = int(logical_idx)
        physical_idx = self._check_physical_idx(physical_idx)
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError("layer_idx is out of range.")
        if logical_idx < 0 or logical_idx >= self.max_blocks_per_layer:
            raise RuntimeError(
                f"Logical block {logical_idx} exceeds max_blocks_per_layer={self.max_blocks_per_layer}."
            )
        self.block_tables[layer_idx, logical_idx] = physical_idx
        self.block_states[layer_idx, logical_idx] = int(state)

    def set_block_state(self, layer_idx: int, logical_idx: int, state: int) -> None:
        if not self.is_allocated():
            raise RuntimeError("TieredKVTensorPool has not been allocated yet.")
        self.block_states[int(layer_idx), int(logical_idx)] = int(state)

    def store_hot_block(self, physical_idx, k_tensor, v_tensor) -> None:
        self.write_hot_tokens(int(physical_idx), 0, k_tensor, v_tensor)

    def demote_to_warm(self, physical_idx) -> None:
        physical_idx = self._check_physical_idx(physical_idx)
        if self.get_physical_state(physical_idx) == WARM_STATE:
            return
        if self.get_physical_state(physical_idx) == COLD_STATE:
            raise KeyError(f"Physical block {physical_idx} is not eligible for WARM demotion from COLD storage.")
        if not self.is_allocated() or self.get_block_length(physical_idx) == 0:
            raise KeyError(f"Physical block {physical_idx} is not present in HOT storage.")

        length = self.get_block_length(physical_idx)
        k_tensor = self.hot_k_pool[physical_idx : physical_idx + 1, :, :length, :].contiguous()
        v_tensor = self.hot_v_pool[physical_idx : physical_idx + 1, :, :length, :].contiguous()
        k_quantized, k_scale, k_zero_point = quantize_fp16_to_int8(k_tensor)
        v_quantized, v_scale, v_zero_point = quantize_fp16_to_int8(v_tensor)

        self.warm_k_pool[physical_idx, :, :length, :].copy_(k_quantized[0])
        self.warm_v_pool[physical_idx, :, :length, :].copy_(v_quantized[0])
        self.k_scale[physical_idx, :, :length, :].copy_(k_scale[0])
        self.k_zero[physical_idx, :, :length, :].copy_(k_zero_point[0])
        self.v_scale[physical_idx, :, :length, :].copy_(v_scale[0])
        self.v_zero[physical_idx, :, :length, :].copy_(v_zero_point[0])
        if length < self.block_size:
            self.warm_k_pool[physical_idx, :, length:, :].zero_()
            self.warm_v_pool[physical_idx, :, length:, :].zero_()
            self.k_scale[physical_idx, :, length:, :].fill_(1)
            self.v_scale[physical_idx, :, length:, :].fill_(1)
            self.k_zero[physical_idx, :, length:, :].zero_()
            self.v_zero[physical_idx, :, length:, :].zero_()

        self.physical_states[physical_idx] = WARM_STATE
        self.hot_storage.discard(physical_idx)
        self.warm_storage.add(physical_idx)
        self.cold_storage.discard(physical_idx)

    def demote_to_cold(self, physical_idx) -> None:
        physical_idx = self._check_physical_idx(physical_idx)
        if self.get_physical_state(physical_idx) == COLD_STATE:
            return
        if not self.is_allocated() or self.get_block_length(physical_idx) == 0:
            raise KeyError(f"Physical block {physical_idx} is not present in HOT or WARM storage.")

        self.physical_states[physical_idx] = COLD_STATE
        self.hot_storage.discard(physical_idx)
        self.warm_storage.discard(physical_idx)
        self.cold_storage.add(physical_idx)

    def promote_to_hot(self, physical_idx) -> None:
        physical_idx = self._check_physical_idx(physical_idx)
        state = self.get_physical_state(physical_idx)
        if state == HOT_STATE:
            return
        if state == COLD_STATE:
            raise KeyError(f"Physical block {physical_idx} is not eligible for HOT promotion from COLD storage.")
        if state != WARM_STATE:
            raise KeyError(f"Physical block {physical_idx} is not present in WARM storage.")

        length = self.get_block_length(physical_idx)
        k_tensor = dequantize_int8_to_fp16(
            self.warm_k_pool[physical_idx : physical_idx + 1, :, :length, :],
            self.k_scale[physical_idx : physical_idx + 1, :, :length, :],
            self.k_zero[physical_idx : physical_idx + 1, :, :length, :],
        )
        v_tensor = dequantize_int8_to_fp16(
            self.warm_v_pool[physical_idx : physical_idx + 1, :, :length, :],
            self.v_scale[physical_idx : physical_idx + 1, :, :length, :],
            self.v_zero[physical_idx : physical_idx + 1, :, :length, :],
        )
        self.hot_k_pool[physical_idx, :, :length, :].copy_(k_tensor[0])
        self.hot_v_pool[physical_idx, :, :length, :].copy_(v_tensor[0])

        self.physical_states[physical_idx] = HOT_STATE
        self.hot_storage.add(physical_idx)
        self.warm_storage.discard(physical_idx)
        self.cold_storage.discard(physical_idx)

    def get_dequantized_block(self, physical_idx, state):
        physical_idx = self._check_physical_idx(physical_idx)
        if not self.is_allocated():
            raise KeyError(f"Physical block {physical_idx} is not present in tensor-pool storage.")

        length = self.get_block_length(physical_idx)
        if state == HOT_STATE:
            if self.get_physical_state(physical_idx) == WARM_STATE:
                self.promote_to_hot(physical_idx)
            if self.get_physical_state(physical_idx) == COLD_STATE:
                raise KeyError(f"Physical block {physical_idx} is marked HOT but only exists in COLD storage.")
            return (
                self.hot_k_pool[physical_idx : physical_idx + 1, :, :length, :].contiguous(),
                self.hot_v_pool[physical_idx : physical_idx + 1, :, :length, :].contiguous(),
            )

        if state == WARM_STATE:
            if self.get_physical_state(physical_idx) == COLD_STATE:
                raise KeyError(f"Physical block {physical_idx} is marked WARM but only exists in COLD storage.")
            if self.get_physical_state(physical_idx) == HOT_STATE:
                return (
                    self.hot_k_pool[physical_idx : physical_idx + 1, :, :length, :].contiguous(),
                    self.hot_v_pool[physical_idx : physical_idx + 1, :, :length, :].contiguous(),
                )
            return (
                dequantize_int8_to_fp16(
                    self.warm_k_pool[physical_idx : physical_idx + 1, :, :length, :],
                    self.k_scale[physical_idx : physical_idx + 1, :, :length, :],
                    self.k_zero[physical_idx : physical_idx + 1, :, :length, :],
                ).contiguous(),
                dequantize_int8_to_fp16(
                    self.warm_v_pool[physical_idx : physical_idx + 1, :, :length, :],
                    self.v_scale[physical_idx : physical_idx + 1, :, :length, :],
                    self.v_zero[physical_idx : physical_idx + 1, :, :length, :],
                ).contiguous(),
            )

        if state == COLD_STATE:
            shape = (1, self.kv_heads, length, self.head_dim)
            return (
                torch.zeros(shape, device=self.device, dtype=torch.float16).contiguous(),
                torch.zeros(shape, device=self.device, dtype=torch.float16).contiguous(),
            )

        raise ValueError(f"Unsupported block state: {state}")

    def sync_gpu_block_tables(self, layer_idx: int, physical_indices, states) -> None:
        if not self.is_allocated():
            return
        self.full_table_sync_count += 1
        layer_idx = int(layer_idx)
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError("layer_idx is out of range.")
        if len(physical_indices) > self.max_blocks_per_layer:
            raise RuntimeError(
                f"Layer {layer_idx} has {len(physical_indices)} blocks, exceeding max_blocks_per_layer="
                f"{self.max_blocks_per_layer}."
            )
        if len(physical_indices) != len(states):
            raise ValueError("Block table and state table lengths must match.")

        self.block_tables[layer_idx].fill_(-1)
        self.block_states[layer_idx].fill_(COLD_STATE)
        if physical_indices:
            physical_tensor = torch.tensor(physical_indices, device=self.device, dtype=torch.int32)
            state_tensor = torch.tensor(states, device=self.device, dtype=torch.int32)
            self.block_tables[layer_idx, : len(physical_indices)].copy_(physical_tensor)
            self.block_states[layer_idx, : len(states)].copy_(state_tensor)

    def get_layer_table(self, layer_idx: int):
        if not self.is_allocated():
            raise RuntimeError("TieredKVTensorPool has not been allocated yet.")
        return self.block_tables[int(layer_idx)], self.block_states[int(layer_idx)]

    def clear(self) -> None:
        if self.is_allocated():
            self.block_lengths.zero_()
            self.block_tables.fill_(-1)
            self.block_states.fill_(COLD_STATE)
        self.physical_states.clear()
        self.hot_storage.clear()
        self.warm_storage.clear()
        self.cold_storage.clear()
        self.full_table_sync_count = 0

    def release(self) -> None:
        self.hot_k_pool = None
        self.hot_v_pool = None
        self.warm_k_pool = None
        self.warm_v_pool = None
        self.k_scale = None
        self.k_zero = None
        self.v_scale = None
        self.v_zero = None
        self.block_lengths = None
        self.block_tables = None
        self.block_states = None
        self.device = None
        self.kv_heads = None
        self.head_dim = None
        self.allocated_pool_blocks = 0
        self.physical_states.clear()
        self.hot_storage.clear()
        self.warm_storage.clear()
        self.cold_storage.clear()
        self.full_table_sync_count = 0


# Keep the old public name for tests and handoff code, but route it to tensor storage.
PhysicalKVPool = TieredKVTensorPool


class TierKVPolicyEngine:
    def __init__(self, block_size: int, block_manager):
        if block_size <= 0:
            raise ValueError("block_size must be positive.")

        self.block_size = block_size
        self.block_manager = block_manager

    def compute_block_scores(self, attn_weights: torch.Tensor) -> torch.Tensor:
        if attn_weights.ndim != 4:
            raise ValueError("attn_weights must have shape [batch, num_heads, query_seq_len, past_seq_len].")
        if attn_weights.shape[-1] == 0:
            raise ValueError("attn_weights past_seq_len must be positive.")

        reduced = attn_weights.mean(dim=(0, 1, 2))
        block_chunks = torch.split(reduced, self.block_size)

        return torch.stack([chunk.mean() for chunk in block_chunks])

    def enforce_budget(
        self,
        seq_id: int,
        block_scores: torch.Tensor,
        hot_budget: int,
        demoted_state: int = WARM_STATE,
    ):
        physical_indices = self.block_manager.get_physical_indices(seq_id)
        current_states = self.block_manager.get_states(seq_id)
        num_blocks = len(physical_indices)

        if num_blocks == 0:
            return
        if demoted_state not in (WARM_STATE, COLD_STATE):
            raise ValueError("demoted_state must be WARM_STATE or COLD_STATE.")
        if block_scores.ndim != 1:
            raise ValueError("block_scores must be a 1D tensor.")
        if len(block_scores) != num_blocks:
            raise ValueError("block_scores length must match the allocated block count.")
        if len(current_states) != num_blocks:
            raise ValueError("Block state table is inconsistent with physical index table.")

        if num_blocks <= 2:
            for block_idx in range(num_blocks):
                if current_states[block_idx] != HOT_STATE:
                    self.block_manager.update_block_state(seq_id, block_idx, HOT_STATE)
            return

        reserved_hot = {0, num_blocks - 1}
        reserved_count = len(reserved_hot)
        extra_hot_slots = max(hot_budget - reserved_count, 0)

        candidate_indices = [idx for idx in range(num_blocks) if idx not in reserved_hot]
        sorted_candidates = sorted(candidate_indices, key=lambda idx: float(block_scores[idx]), reverse=True)
        selected_hot = set(sorted_candidates[:extra_hot_slots])

        for block_idx in range(num_blocks):
            desired_state = HOT_STATE if block_idx in reserved_hot or block_idx in selected_hot else demoted_state
            if current_states[block_idx] == COLD_STATE and desired_state != COLD_STATE:
                desired_state = COLD_STATE
            if current_states[block_idx] != desired_state:
                self.block_manager.update_block_state(seq_id, block_idx, desired_state)

    def enforce_tiered_budget(
        self,
        seq_id: int,
        block_scores: torch.Tensor,
        hot_budget: int,
        warm_budget: int,
    ):
        physical_indices = self.block_manager.get_physical_indices(seq_id)
        current_states = self.block_manager.get_states(seq_id)
        num_blocks = len(physical_indices)

        if num_blocks == 0:
            return
        if block_scores.ndim != 1:
            raise ValueError("block_scores must be a 1D tensor.")
        if len(block_scores) != num_blocks:
            raise ValueError("block_scores length must match the allocated block count.")
        if len(current_states) != num_blocks:
            raise ValueError("Block state table is inconsistent with physical index table.")

        if num_blocks <= 2:
            for block_idx in range(num_blocks):
                if current_states[block_idx] != HOT_STATE:
                    self.block_manager.update_block_state(seq_id, block_idx, HOT_STATE)
            return

        reserved_hot = {0, num_blocks - 1}
        candidate_indices = [idx for idx in range(num_blocks) if idx not in reserved_hot]
        sorted_candidates = sorted(candidate_indices, key=lambda idx: float(block_scores[idx]), reverse=True)
        extra_hot_slots = max(int(hot_budget) - len(reserved_hot), 0)
        selected_hot = set(sorted_candidates[:extra_hot_slots])
        selected_warm = set(sorted_candidates[extra_hot_slots : extra_hot_slots + max(int(warm_budget), 0)])

        for block_idx in range(num_blocks):
            if block_idx in reserved_hot or block_idx in selected_hot:
                desired_state = HOT_STATE
            elif block_idx in selected_warm:
                desired_state = WARM_STATE
            else:
                desired_state = COLD_STATE
            if current_states[block_idx] == COLD_STATE and desired_state != COLD_STATE:
                desired_state = COLD_STATE
            if current_states[block_idx] != desired_state:
                self.block_manager.update_block_state(seq_id, block_idx, desired_state)


@dataclass
class LayerRuntimeState:
    open_physical_idx: int | None = None
    open_token_count: int = 0
    num_blocks: int = 0


class TieredKVCache(HFCache):
    is_compileable = False

    def __init__(self, runtime):
        self.runtime = runtime
        self.is_tierkv_cache = True
        self.pending_query_length = 0
        self.committed_seq_len = 0

    def begin_forward(self, query_length: int) -> None:
        self.pending_query_length = int(query_length)
        self.runtime.begin_forward(query_length)

    def finish_forward(self) -> None:
        self.committed_seq_len += self.pending_query_length
        self.pending_query_length = 0
        self.runtime.finish_forward()

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.committed_seq_len

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> tuple[int, int]:
        return self.committed_seq_len + query_length, 0

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return -1

    def reset(self) -> None:
        self.pending_query_length = 0
        self.committed_seq_len = 0

    def reorder_cache(self, beam_idx: torch.LongTensor):
        return None

    def crop(self, max_length: int):
        return None

    def batch_repeat_interleave(self, repeats: int):
        return None

    def batch_select_indices(self, indices: torch.Tensor):
        return None

    def __len__(self):
        return self.runtime.num_layers


class TieredKVRuntime:
    def __init__(
        self,
        block_manager,
        num_layers: int,
        block_size: int,
        hot_budget: int,
        max_seq_len: int = 2048,
        attention_backend: str | None = None,
        policy_interval: int | None = None,
        pool_chunk_blocks: int = 128,
        policy_on_new_block: bool = False,
        score_accumulation_mode: str = "policy_ticks",
        policy_mode: str = "hot_warm",
        warm_budget: int = 32,
    ):
        attention_backend = attention_backend or os.environ.get("TIERKV_ATTENTION_BACKEND", "triton")
        if attention_backend not in ("eager", "triton"):
            raise ValueError('attention_backend must be "eager" or "triton".')

        if policy_interval is None:
            policy_interval = int(os.environ.get("TIERKV_POLICY_INTERVAL", "16"))
        if policy_interval <= 0:
            raise ValueError("policy_interval must be positive.")
        pool_chunk_blocks = int(os.environ.get("TIERKV_POOL_CHUNK_BLOCKS", pool_chunk_blocks))
        if pool_chunk_blocks <= 0:
            raise ValueError("pool_chunk_blocks must be positive.")
        policy_on_new_block = os.environ.get("TIERKV_POLICY_ON_NEW_BLOCK", str(policy_on_new_block)).lower()
        policy_on_new_block = policy_on_new_block in ("1", "true", "yes", "on")
        score_accumulation_mode = os.environ.get("TIERKV_SCORE_ACCUMULATION_MODE", score_accumulation_mode)
        if score_accumulation_mode not in ("policy_ticks", "every_step"):
            raise ValueError('score_accumulation_mode must be "policy_ticks" or "every_step".')
        policy_mode = os.environ.get("TIERKV_POLICY_MODE", policy_mode)
        if policy_mode not in ("hot_only", "hot_warm", "hot_warm_cold"):
            raise ValueError('policy_mode must be "hot_only", "hot_warm", or "hot_warm_cold".')
        warm_budget = int(os.environ.get("TIERKV_WARM_BUDGET", warm_budget))
        if warm_budget < 0:
            raise ValueError("warm_budget must be non-negative.")

        self.block_manager = block_manager
        self.num_layers = int(num_layers)
        self.block_size = int(block_size)
        self.hot_budget = int(hot_budget)
        self.max_seq_len = int(max_seq_len)
        self.max_blocks_per_layer = _ceil_div(self.max_seq_len, self.block_size)
        self.pool_blocks = self.num_layers * self.max_blocks_per_layer
        self.attention_backend = attention_backend
        self.policy_interval = int(policy_interval)
        self.pool_chunk_blocks = int(pool_chunk_blocks)
        self.policy_on_new_block = bool(policy_on_new_block)
        self.score_accumulation_mode = score_accumulation_mode
        self.policy_mode = policy_mode
        self.warm_budget = int(warm_budget)
        self.demoted_state = WARM_STATE
        self.cold_block_policy = "zero"
        self.tiered_layer_indices = set(range(num_layers))
        self.kv_pool = TieredKVTensorPool(
            num_layers=self.num_layers,
            block_size=self.block_size,
            max_seq_len=self.max_seq_len,
            max_blocks_per_layer=self.max_blocks_per_layer,
            pool_blocks=self.pool_blocks,
            pool_chunk_blocks=self.pool_chunk_blocks,
        )
        self.policy_engine = TierKVPolicyEngine(block_size=block_size, block_manager=block_manager)
        self.layer_states = {layer_idx: LayerRuntimeState() for layer_idx in range(num_layers)}
        self.cache = TieredKVCache(self)
        self.current_query_length = 0
        self.decode_step = 0
        self.policy_tick_this_forward = False
        self.profile_enabled = os.environ.get("TIERKV_PROFILE", "0") == "1"
        self.profile_totals_ms: dict[str, float] = {}
        self.profile_counts: dict[str, int] = {}

    def begin_forward(self, query_length: int) -> None:
        self.current_query_length = int(query_length)
        self.policy_tick_this_forward = False
        if self.current_query_length == 1:
            self.decode_step += 1
            self.policy_tick_this_forward = (self.decode_step % self.policy_interval) == 0

    def finish_forward(self) -> None:
        self.current_query_length = 0
        self.policy_tick_this_forward = False

    def reset(self, release_storage: bool = False) -> None:
        for seq_id in range(self.num_layers):
            self.block_manager.free_sequence(seq_id)
        if release_storage:
            self.kv_pool.release()
        else:
            self.kv_pool.clear()
        self.layer_states = {layer_idx: LayerRuntimeState() for layer_idx in range(self.num_layers)}
        self.decode_step = 0
        self.current_query_length = 0
        self.policy_tick_this_forward = False
        self.profile_totals_ms.clear()
        self.profile_counts.clear()
        self.cache.reset()

    def profile_start(self):
        if not self.profile_enabled or not torch.cuda.is_available():
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def profile_end(self, name: str, start_event) -> None:
        if start_event is None:
            return
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        end_event.synchronize()
        elapsed = float(start_event.elapsed_time(end_event))
        self.profile_totals_ms[name] = self.profile_totals_ms.get(name, 0.0) + elapsed
        self.profile_counts[name] = self.profile_counts.get(name, 0) + 1

    def profile_summary(self) -> dict[str, tuple[float, int, float]]:
        summary = {}
        for name, total_ms in self.profile_totals_ms.items():
            count = self.profile_counts.get(name, 0)
            summary[name] = (total_ms, count, total_ms / max(count, 1))
        return summary

    def append_tokens(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor) -> bool:
        return self.append_to_layer(layer_idx, key_states, value_states)

    def append_to_layer(self, seq_id: int, key_states: torch.Tensor, value_states: torch.Tensor) -> bool:
        if key_states.shape[-2] != value_states.shape[-2]:
            raise ValueError("Key and value states must have the same sequence length.")

        layer_state = self.layer_states[seq_id]
        offset = 0
        total_tokens = int(key_states.shape[-2])
        allocated_new_block = False

        while offset < total_tokens:
            new_logical_idx = None
            if layer_state.open_physical_idx is None:
                layer_state.open_physical_idx = self.block_manager.allocate_block(seq_id)
                layer_state.open_token_count = 0
                allocated_new_block = True

                new_logical_idx = layer_state.num_blocks
                if new_logical_idx >= self.max_blocks_per_layer:
                    raise RuntimeError(
                        f"Layer {seq_id} exceeded max_blocks_per_layer={self.max_blocks_per_layer}."
                    )
                self.kv_pool._check_physical_idx(layer_state.open_physical_idx)

            physical_idx = layer_state.open_physical_idx
            if self.kv_pool.get_physical_state(physical_idx) == WARM_STATE:
                self.kv_pool.promote_to_hot(physical_idx)

            remaining_capacity = self.block_size - layer_state.open_token_count
            take = min(remaining_capacity, total_tokens - offset)
            k_chunk = key_states[:, :, offset : offset + take, :].contiguous()
            v_chunk = value_states[:, :, offset : offset + take, :].contiguous()

            self.kv_pool.write_hot_tokens(
                physical_idx,
                layer_state.open_token_count,
                k_chunk,
                v_chunk,
            )
            if new_logical_idx is not None:
                self.kv_pool.set_block_entry(seq_id, new_logical_idx, physical_idx, HOT_STATE)
                layer_state.num_blocks += 1
            layer_state.open_token_count += take
            offset += take

            if layer_state.open_token_count == self.block_size:
                layer_state.open_physical_idx = None
                layer_state.open_token_count = 0

        return allocated_new_block

    def reconstruct_layer(self, seq_id: int, allow_decode_reconstruct: bool = False):
        if (
            self.attention_backend == "triton"
            and self.current_query_length == 1
            and not allow_decode_reconstruct
        ):
            raise RuntimeError("Decode reconstruction is disabled for the Triton TierKV hot path.")

        physical_indices = self.block_manager.get_physical_indices(seq_id)
        states = self.block_manager.get_states(seq_id)

        if not physical_indices:
            return None, None, physical_indices, states

        key_blocks = []
        value_blocks = []
        for physical_idx, state in zip(physical_indices, states):
            key_block, value_block = self.kv_pool.get_dequantized_block(physical_idx, state)
            key_blocks.append(key_block)
            value_blocks.append(value_block)

        key_states = torch.cat(key_blocks, dim=-2).contiguous()
        value_states = torch.cat(value_blocks, dim=-2).contiguous()
        return key_states, value_states, physical_indices, states

    def sync_gpu_block_tables(self, seq_id: int) -> None:
        physical_indices = self.block_manager.get_physical_indices(seq_id)
        states = self.block_manager.get_states(seq_id)
        self.kv_pool.sync_gpu_block_tables(seq_id, physical_indices, states)
        self.layer_states[seq_id].num_blocks = len(physical_indices)

    def sync_storage_states(self, seq_id: int) -> None:
        physical_indices = self.block_manager.get_physical_indices(seq_id)
        states = self.block_manager.get_states(seq_id)

        for logical_idx, (physical_idx, state) in enumerate(zip(physical_indices, states)):
            current_state = self.kv_pool.get_physical_state(physical_idx)
            if state == HOT_STATE and current_state == WARM_STATE:
                self.kv_pool.promote_to_hot(physical_idx)
            elif state == COLD_STATE and current_state != COLD_STATE:
                self.kv_pool.demote_to_cold(physical_idx)
            elif state == WARM_STATE and current_state == HOT_STATE:
                self.kv_pool.demote_to_warm(physical_idx)
            self.kv_pool.set_block_state(seq_id, logical_idx, state)

    def get_layer_table(self, seq_id: int):
        block_table, block_states = self.kv_pool.get_layer_table(seq_id)
        num_blocks = self.layer_states[seq_id].num_blocks
        return block_table, block_states, self.kv_pool.block_lengths, num_blocks

    def should_run_decode_policy(self, seq_id: int, allocated_new_block: bool) -> bool:
        return (
            seq_id in self.tiered_layer_indices
            and self.policy_mode != "hot_only"
            and self.current_query_length == 1
            and (self.policy_tick_this_forward or (self.policy_on_new_block and allocated_new_block))
        )

    def should_collect_decode_scores(self, seq_id: int, run_policy: bool) -> bool:
        if seq_id not in self.tiered_layer_indices or self.policy_mode == "hot_only":
            return False
        return bool(run_policy or self.score_accumulation_mode == "every_step")

    def enforce_policy(self, seq_id: int, block_scores: torch.Tensor) -> None:
        if self.policy_mode == "hot_only":
            return
        if self.policy_mode == "hot_warm_cold":
            self.policy_engine.enforce_tiered_budget(
                seq_id,
                block_scores,
                self.hot_budget,
                self.warm_budget,
            )
            return
        self.policy_engine.enforce_budget(
            seq_id,
            block_scores,
            self.hot_budget,
            demoted_state=self.demoted_state,
        )

    def block_scores_from_probability_sums(self, seq_id: int, score_sums: torch.Tensor) -> torch.Tensor:
        num_blocks = self.layer_states[seq_id].num_blocks
        if num_blocks == 0:
            return torch.empty(0, device=score_sums.device, dtype=torch.float32)

        block_table, _ = self.kv_pool.get_layer_table(seq_id)
        physical_indices = block_table[:num_blocks].to(torch.long)
        lengths = self.kv_pool.block_lengths.index_select(0, physical_indices).clamp_min(1).to(torch.float32)
        per_block_probability = score_sums[:, :num_blocks].mean(dim=0)
        return per_block_probability / lengths


def initialize_global_tierkv(
    block_manager,
    num_layers,
    block_size=16,
    hot_budget=3,
    max_seq_len=2048,
    attention_backend: str | None = None,
    policy_interval: int | None = None,
    pool_chunk_blocks: int = 128,
    policy_on_new_block: bool = False,
    score_accumulation_mode: str = "policy_ticks",
    policy_mode: str = "hot_warm",
    warm_budget: int = 32,
) -> TieredKVCache:
    global _GLOBAL_TIERKV_RUNTIME

    if _GLOBAL_TIERKV_RUNTIME is not None:
        _GLOBAL_TIERKV_RUNTIME.reset(release_storage=True)

    _GLOBAL_TIERKV_RUNTIME = TieredKVRuntime(
        block_manager=block_manager,
        num_layers=num_layers,
        block_size=block_size,
        hot_budget=hot_budget,
        max_seq_len=max_seq_len,
        attention_backend=attention_backend,
        policy_interval=policy_interval,
        pool_chunk_blocks=pool_chunk_blocks,
        policy_on_new_block=policy_on_new_block,
        score_accumulation_mode=score_accumulation_mode,
        policy_mode=policy_mode,
        warm_budget=warm_budget,
    )
    return _GLOBAL_TIERKV_RUNTIME.cache


def reset_global_tierkv() -> None:
    global _GLOBAL_TIERKV_RUNTIME

    if _GLOBAL_TIERKV_RUNTIME is not None:
        _GLOBAL_TIERKV_RUNTIME.reset(release_storage=True)
    _GLOBAL_TIERKV_RUNTIME = None


def get_global_tierkv_runtime():
    return _GLOBAL_TIERKV_RUNTIME
