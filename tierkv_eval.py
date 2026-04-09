import math
import re

import torch


NEEDLE_FILLER_PARAGRAPH = (
    "Tiered KV cache experiments stress prompt prefill, decode reuse, and long-context retrieval. "
    "A serving stack accumulates keys and values for every decoder layer, so older activations compete "
    "with recent tokens for scarce high-bandwidth memory. Quantization and sparsification reduce memory "
    "pressure, but they can also destroy context if the routing policy demotes semantically important blocks. "
    "Reliable evaluation therefore measures targeted retrieval and memory usage under the same "
    "long-context prompt construction."
)


def _snapshot_tierkv_runtime():
    try:
        from tierkv_policy import get_global_tierkv_runtime
    except Exception:
        return None

    runtime = get_global_tierkv_runtime()
    if runtime is None:
        return None

    return {
        "block_manager_cls": runtime.block_manager.__class__,
        "num_layers": runtime.num_layers,
        "block_size": runtime.block_size,
        "hot_budget": runtime.hot_budget,
        "demoted_state": runtime.demoted_state,
        "cold_block_policy": runtime.cold_block_policy,
        "tiered_layer_indices": set(runtime.tiered_layer_indices),
    }


def _prepare_tierkv_cache(runtime_snapshot):
    if runtime_snapshot is None:
        return None

    from tierkv_policy import initialize_global_tierkv

    block_manager = runtime_snapshot["block_manager_cls"]()
    cache = initialize_global_tierkv(
        block_manager=block_manager,
        num_layers=runtime_snapshot["num_layers"],
        block_size=runtime_snapshot["block_size"],
        hot_budget=runtime_snapshot["hot_budget"],
    )
    runtime = cache.runtime
    runtime.demoted_state = runtime_snapshot["demoted_state"]
    runtime.cold_block_policy = runtime_snapshot["cold_block_policy"]
    runtime.tiered_layer_indices = set(runtime_snapshot["tiered_layer_indices"])
    return cache


def _reset_tierkv_runtime(runtime_snapshot):
    if runtime_snapshot is None:
        return

    from tierkv_policy import reset_global_tierkv

    reset_global_tierkv()


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _build_needle_prompt(tokenizer, answer: str, context_tokens: int, sample_idx: int) -> tuple[str, int]:
    intro = (
        f"Document {sample_idx}: the audit retrieval code is {answer}. "
        "Preserve that exact code because a later verification step will ask for it. "
    )
    question = "\nQuestion: What is the audit retrieval code? Answer with only the code.\nAnswer:"

    filler_sections = []
    prompt_text = intro + question
    prompt_token_length = tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1]

    while prompt_token_length < context_tokens:
        filler_sections.append(f"Section {len(filler_sections)}: {NEEDLE_FILLER_PARAGRAPH}")
        prompt_text = intro + " ".join(filler_sections) + question
        prompt_token_length = tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1]

    return prompt_text, prompt_token_length


def _greedy_generate(model, tokenizer, prompt_text: str, max_new_tokens: int, runtime_snapshot):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")
    past_key_values = _prepare_tierkv_cache(runtime_snapshot)

    generated_tokens = []
    current_input_ids = input_ids
    current_attention_mask = attention_mask

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated_tokens.append(next_token)
            current_input_ids = next_token
            current_attention_mask = torch.cat(
                [
                    current_attention_mask,
                    torch.ones(
                        (current_attention_mask.size(0), 1),
                        dtype=current_attention_mask.dtype,
                        device=current_attention_mask.device,
                    ),
                ],
                dim=1,
            )

    if generated_tokens:
        completion_ids = torch.cat(generated_tokens, dim=1)
        completion_text = tokenizer.decode(completion_ids[0], skip_special_tokens=True).strip()
    else:
        completion_text = ""

    _reset_tierkv_runtime(runtime_snapshot)
    return completion_text


def evaluate_long_context(model, tokenizer, num_samples=5, context_tokens=1024, max_new_tokens=32) -> dict:
    runtime_snapshot = _snapshot_tierkv_runtime()
    answer_keys = [f"KEY-{1000 + idx}" for idx in range(num_samples)]
    hits = 0
    examples = []
    model.eval()

    for sample_idx, answer in enumerate(answer_keys):
        prompt_text, prompt_token_length = _build_needle_prompt(tokenizer, answer, context_tokens, sample_idx)
        completion_text = _greedy_generate(
            model,
            tokenizer,
            prompt_text,
            max_new_tokens=max_new_tokens,
            runtime_snapshot=runtime_snapshot,
        )
        hit = _normalize_text(answer) in _normalize_text(completion_text)
        hits += int(hit)
        examples.append(
            {
                "answer": answer,
                "completion": completion_text,
                "hit": hit,
                "prompt_token_length": prompt_token_length,
            }
        )

    return {
        "hits": hits,
        "total": num_samples,
        "success_rate": hits / max(num_samples, 1),
        "examples": examples,
    }


def format_markdown_table(results) -> str:
    def format_number(value):
        if value is None:
            return "n/a"
        if isinstance(value, float) and not math.isfinite(value):
            return "inf"
        return f"{value:.2f}"

    lines = [
        "| Configuration | Peak Memory (MB) | Needle Success | Tokens/s |",
        "| --- | ---: | --- | ---: |",
    ]

    for result in results:
        needle = result["needle"]
        lines.append(
            "| "
            f"{result['configuration']} | "
            f"{format_number(result['peak_memory_mb'])} | "
            f"{needle['hits']}/{needle['total']} ({needle['success_rate'] * 100:.0f}%) | "
            f"{format_number(result['tokens_per_sec'])} |"
        )

    return "\n".join(lines)
