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


class PhysicalKVPool:
    def __init__(self):
        self.hot_storage = {}
        self.warm_storage = {}
        self.cold_storage = {}

    def store_hot_block(self, physical_idx, k_tensor, v_tensor) -> None:
        self.hot_storage[physical_idx] = (k_tensor.contiguous(), v_tensor.contiguous())
        self.warm_storage.pop(physical_idx, None)
        self.cold_storage.pop(physical_idx, None)

    def demote_to_warm(self, physical_idx) -> None:
        if physical_idx in self.warm_storage:
            return
        if physical_idx not in self.hot_storage:
            raise KeyError(f"Physical block {physical_idx} is not present in HOT storage.")

        k_tensor, v_tensor = self.hot_storage.pop(physical_idx)
        k_quantized, k_scale, k_zero_point = quantize_fp16_to_int8(k_tensor)
        v_quantized, v_scale, v_zero_point = quantize_fp16_to_int8(v_tensor)
        self.warm_storage[physical_idx] = (
            k_quantized,
            k_scale,
            k_zero_point,
            v_quantized,
            v_scale,
            v_zero_point,
        )

    def demote_to_cold(self, physical_idx) -> None:
        if physical_idx in self.cold_storage:
            return

        if physical_idx in self.hot_storage:
            k_tensor, v_tensor = self.hot_storage.pop(physical_idx)
            metadata = (
                tuple(k_tensor.shape),
                k_tensor.device,
                k_tensor.dtype,
                tuple(v_tensor.shape),
                v_tensor.device,
                v_tensor.dtype,
            )
        elif physical_idx in self.warm_storage:
            (
                k_quantized,
                _k_scale,
                _k_zero_point,
                v_quantized,
                _v_scale,
                _v_zero_point,
            ) = self.warm_storage.pop(physical_idx)
            metadata = (
                tuple(k_quantized.shape),
                k_quantized.device,
                torch.float16,
                tuple(v_quantized.shape),
                v_quantized.device,
                torch.float16,
            )
        else:
            raise KeyError(f"Physical block {physical_idx} is not present in HOT or WARM storage.")

        self.cold_storage[physical_idx] = metadata

    def promote_to_hot(self, physical_idx) -> None:
        if physical_idx in self.hot_storage:
            return
        if physical_idx in self.cold_storage:
            raise KeyError(f"Physical block {physical_idx} is not eligible for HOT promotion from COLD storage.")
        if physical_idx not in self.warm_storage:
            raise KeyError(f"Physical block {physical_idx} is not present in WARM storage.")

        (
            k_quantized,
            k_scale,
            k_zero_point,
            v_quantized,
            v_scale,
            v_zero_point,
        ) = self.warm_storage.pop(physical_idx)
        k_tensor = dequantize_int8_to_fp16(k_quantized, k_scale, k_zero_point)
        v_tensor = dequantize_int8_to_fp16(v_quantized, v_scale, v_zero_point)
        self.hot_storage[physical_idx] = (k_tensor.contiguous(), v_tensor.contiguous())

    def get_dequantized_block(self, physical_idx, state):
        if state == HOT_STATE:
            if physical_idx not in self.hot_storage and physical_idx in self.warm_storage:
                self.promote_to_hot(physical_idx)
            if physical_idx in self.cold_storage:
                raise KeyError(f"Physical block {physical_idx} is marked HOT but only exists in COLD storage.")
            return self.hot_storage[physical_idx]

        if state == WARM_STATE:
            if physical_idx in self.hot_storage:
                return self.hot_storage[physical_idx]
            if physical_idx in self.cold_storage:
                raise KeyError(f"Physical block {physical_idx} is marked WARM but only exists in COLD storage.")

            (
                k_quantized,
                k_scale,
                k_zero_point,
                v_quantized,
                v_scale,
                v_zero_point,
            ) = self.warm_storage[physical_idx]
            k_tensor = dequantize_int8_to_fp16(k_quantized, k_scale, k_zero_point)
            v_tensor = dequantize_int8_to_fp16(v_quantized, v_scale, v_zero_point)
            return k_tensor.contiguous(), v_tensor.contiguous()

        if state == COLD_STATE:
            if physical_idx not in self.cold_storage:
                raise KeyError(f"Physical block {physical_idx} is not present in COLD storage.")

            (
                k_shape,
                k_device,
                k_dtype,
                v_shape,
                v_device,
                v_dtype,
            ) = self.cold_storage[physical_idx]
            key_block = torch.zeros(k_shape, device=k_device, dtype=k_dtype).contiguous()
            value_block = torch.zeros(v_shape, device=v_device, dtype=v_dtype).contiguous()
            return key_block, value_block

        raise ValueError(f"Unsupported block state: {state}")

    def clear(self) -> None:
        self.hot_storage.clear()
        self.warm_storage.clear()
        self.cold_storage.clear()


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


@dataclass
class LayerRuntimeState:
    open_physical_idx: int | None = None
    open_token_count: int = 0


class TieredKVCache(HFCache):
    is_compileable = False

    def __init__(self, runtime):
        self.runtime = runtime
        self.is_tierkv_cache = True
        self.pending_query_length = 0
        self.committed_seq_len = 0

    def begin_forward(self, query_length: int) -> None:
        self.pending_query_length = int(query_length)

    def finish_forward(self) -> None:
        self.committed_seq_len += self.pending_query_length
        self.pending_query_length = 0

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
    def __init__(self, block_manager, num_layers: int, block_size: int, hot_budget: int):
        self.block_manager = block_manager
        self.num_layers = num_layers
        self.block_size = block_size
        self.hot_budget = hot_budget
        self.demoted_state = WARM_STATE
        self.cold_block_policy = "zero"
        self.tiered_layer_indices = set(range(num_layers))
        self.kv_pool = PhysicalKVPool()
        self.policy_engine = TierKVPolicyEngine(block_size=block_size, block_manager=block_manager)
        self.layer_states = {layer_idx: LayerRuntimeState() for layer_idx in range(num_layers)}
        self.cache = TieredKVCache(self)

    def reset(self) -> None:
        for seq_id in range(self.num_layers):
            self.block_manager.free_sequence(seq_id)
        self.kv_pool.clear()
        self.layer_states = {layer_idx: LayerRuntimeState() for layer_idx in range(self.num_layers)}
        self.cache.reset()

    def append_to_layer(self, seq_id: int, key_states: torch.Tensor, value_states: torch.Tensor) -> bool:
        if key_states.shape[-2] != value_states.shape[-2]:
            raise ValueError("Key and value states must have the same sequence length.")

        layer_state = self.layer_states[seq_id]
        offset = 0
        total_tokens = key_states.shape[-2]
        allocated_new_block = False

        while offset < total_tokens:
            if layer_state.open_physical_idx is None:
                layer_state.open_physical_idx = self.block_manager.allocate_block(seq_id)
                layer_state.open_token_count = 0
                allocated_new_block = True

            physical_idx = layer_state.open_physical_idx
            if physical_idx in self.kv_pool.warm_storage:
                self.kv_pool.promote_to_hot(physical_idx)

            remaining_capacity = self.block_size - layer_state.open_token_count
            take = min(remaining_capacity, total_tokens - offset)
            k_chunk = key_states[:, :, offset : offset + take, :].contiguous()
            v_chunk = value_states[:, :, offset : offset + take, :].contiguous()

            if physical_idx in self.kv_pool.hot_storage:
                existing_k, existing_v = self.kv_pool.hot_storage[physical_idx]
                k_chunk = torch.cat([existing_k, k_chunk], dim=-2).contiguous()
                v_chunk = torch.cat([existing_v, v_chunk], dim=-2).contiguous()

            self.kv_pool.store_hot_block(physical_idx, k_chunk, v_chunk)
            layer_state.open_token_count += take
            offset += take

            if layer_state.open_token_count == self.block_size:
                layer_state.open_physical_idx = None
                layer_state.open_token_count = 0

        return allocated_new_block

    def reconstruct_layer(self, seq_id: int):
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

    def sync_storage_states(self, seq_id: int) -> None:
        physical_indices = self.block_manager.get_physical_indices(seq_id)
        states = self.block_manager.get_states(seq_id)

        for physical_idx, state in zip(physical_indices, states):
            if state == HOT_STATE and physical_idx in self.kv_pool.warm_storage:
                self.kv_pool.promote_to_hot(physical_idx)
            elif state == COLD_STATE:
                if physical_idx in self.kv_pool.hot_storage or physical_idx in self.kv_pool.warm_storage:
                    self.kv_pool.demote_to_cold(physical_idx)
            elif state == WARM_STATE and physical_idx in self.kv_pool.hot_storage:
                self.kv_pool.demote_to_warm(physical_idx)


def initialize_global_tierkv(block_manager, num_layers, block_size=16, hot_budget=3) -> TieredKVCache:
    global _GLOBAL_TIERKV_RUNTIME

    if _GLOBAL_TIERKV_RUNTIME is not None:
        _GLOBAL_TIERKV_RUNTIME.reset()

    _GLOBAL_TIERKV_RUNTIME = TieredKVRuntime(
        block_manager=block_manager,
        num_layers=num_layers,
        block_size=block_size,
        hot_budget=hot_budget,
    )
    return _GLOBAL_TIERKV_RUNTIME.cache


def reset_global_tierkv() -> None:
    global _GLOBAL_TIERKV_RUNTIME

    if _GLOBAL_TIERKV_RUNTIME is not None:
        _GLOBAL_TIERKV_RUNTIME.reset()
    _GLOBAL_TIERKV_RUNTIME = None


def get_global_tierkv_runtime():
    return _GLOBAL_TIERKV_RUNTIME
