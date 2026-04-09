import torch


def quantize_fp16_to_int8(tensor: torch.Tensor):
    if tensor.dtype != torch.float16:
        tensor = tensor.to(torch.float16)

    min_val = tensor.amin()
    max_val = tensor.amax()
    scale = (max_val - min_val) / 255.0
    scale = torch.where(
        scale == 0,
        torch.ones((), device=tensor.device, dtype=torch.float16),
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


class TierKVPolicyEngine:
    def __init__(self, block_size: int, block_manager):
        if block_size <= 0:
            raise ValueError("block_size must be positive.")

        self.block_size = block_size
        self.block_manager = block_manager

    def compute_block_scores(self, attn_weights: torch.Tensor) -> torch.Tensor:
        if attn_weights.ndim != 4:
            raise ValueError("attn_weights must have shape [batch, num_heads, 1, past_seq_len].")
        if attn_weights.shape[2] != 1:
            raise ValueError("attn_weights query dimension must be 1 for the standalone policy test.")

        reduced = attn_weights.mean(dim=(0, 1)).squeeze(0)
        block_chunks = torch.split(reduced, self.block_size)

        return torch.stack([chunk.mean() for chunk in block_chunks])

    def enforce_budget(self, seq_id: int, block_scores: torch.Tensor, hot_budget: int):
        physical_indices = self.block_manager.get_physical_indices(seq_id)
        num_blocks = len(physical_indices)

        if num_blocks == 0:
            return
        if block_scores.ndim != 1:
            raise ValueError("block_scores must be a 1D tensor.")
        if len(block_scores) != num_blocks:
            raise ValueError("block_scores length must match the allocated block count.")

        if num_blocks <= 2:
            for block_idx in range(num_blocks):
                self.block_manager.update_block_state(seq_id, block_idx, 0)
            return

        reserved_hot = {0, num_blocks - 1}
        reserved_count = len(reserved_hot)
        extra_hot_slots = max(hot_budget - reserved_count, 0)

        candidate_indices = [idx for idx in range(num_blocks) if idx not in reserved_hot]
        sorted_candidates = sorted(candidate_indices, key=lambda idx: float(block_scores[idx]), reverse=True)
        selected_hot = set(sorted_candidates[:extra_hot_slots])

        for block_idx in range(num_blocks):
            if block_idx in reserved_hot or block_idx in selected_hot:
                self.block_manager.update_block_state(seq_id, block_idx, 0)
            else:
                self.block_manager.update_block_state(seq_id, block_idx, 1)
