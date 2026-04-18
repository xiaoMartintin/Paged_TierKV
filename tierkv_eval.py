import collections
import gc
import math
import re
import string
import time

import torch


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
        "max_seq_len": runtime.max_seq_len,
        "attention_backend": runtime.attention_backend,
        "policy_interval": runtime.policy_interval,
        "policy_on_new_block": runtime.policy_on_new_block,
        "score_accumulation_mode": runtime.score_accumulation_mode,
        "policy_mode": runtime.policy_mode,
        "warm_budget": runtime.warm_budget,
        "pool_layout": runtime.pool_layout,
        "pool_blocks": runtime.pool_blocks,
        "pool_chunk_blocks": runtime.pool_chunk_blocks,
        "demoted_state": runtime.demoted_state,
        "cold_block_policy": runtime.cold_block_policy,
        "tiered_layer_indices": set(runtime.tiered_layer_indices),
    }


def _prepare_tierkv_cache(runtime_snapshot):
    if runtime_snapshot is None:
        return None

    from tierkv_policy import create_tierkv_runtime

    block_manager = runtime_snapshot["block_manager_cls"](total_blocks=runtime_snapshot["pool_blocks"])
    runtime = create_tierkv_runtime(
        block_manager=block_manager,
        num_layers=runtime_snapshot["num_layers"],
        block_size=runtime_snapshot["block_size"],
        hot_budget=runtime_snapshot["hot_budget"],
        max_seq_len=runtime_snapshot["max_seq_len"],
        attention_backend=runtime_snapshot["attention_backend"],
        policy_interval=runtime_snapshot["policy_interval"],
        policy_on_new_block=runtime_snapshot["policy_on_new_block"],
        score_accumulation_mode=runtime_snapshot["score_accumulation_mode"],
        policy_mode=runtime_snapshot["policy_mode"],
        warm_budget=runtime_snapshot["warm_budget"],
        pool_layout=runtime_snapshot["pool_layout"],
        pool_chunk_blocks=runtime_snapshot["pool_chunk_blocks"],
    )
    runtime.demoted_state = runtime_snapshot["demoted_state"]
    runtime.cold_block_policy = runtime_snapshot["cold_block_policy"]
    runtime.tiered_layer_indices = set(runtime_snapshot["tiered_layer_indices"])
    return runtime.cache


def _reset_tierkv_cache(cache):
    if not getattr(cache, "is_tierkv_cache", False):
        return

    cache.runtime.reset(release_storage=True)


def _normalize_answer(text: str) -> str:
    def remove_articles(value):
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value):
        return " ".join(value.split())

    def remove_punc(value):
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def token_f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = _normalize_answer(prediction).split()
    ground_truth_tokens = _normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return float(prediction_tokens == ground_truth_tokens)
    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def max_token_f1(prediction: str, answers) -> float:
    if not answers:
        return 0.0
    return max(token_f1_score(prediction, answer) for answer in answers)


def _empty_memory_metrics() -> dict:
    return {
        "hot_blocks": 0,
        "warm_blocks": 0,
        "cold_blocks": 0,
        "hot_tokens": 0,
        "warm_tokens": 0,
        "cold_tokens": 0,
        "active_tokens": 0,
        "logical_hot_mb": 0.0,
        "logical_warm_mb": 0.0,
        "logical_warm_meta_mb": 0.0,
        "logical_cold_evicted_mb": 0.0,
        "logical_dense_equivalent_mb": 0.0,
        "logical_resident_mb": 0.0,
        "logical_compression_ratio": 0.0,
        "logical_compression_x": 0.0,
        "physical_hot_pool_mb": 0.0,
        "physical_warm_pool_mb": 0.0,
        "physical_warm_meta_mb": 0.0,
        "physical_total_kv_mb": 0.0,
        "physical_compression_x": 0.0,
        "active_pool_blocks": 0,
        "allocated_pool_blocks": 0,
        "pool_blocks": 0,
        "hot_slots_used": 0,
        "hot_slots_capacity": 0,
        "warm_slots_used": 0,
        "warm_slots_capacity": 0,
        "pool_utilization": 0.0,
        "capacity_utilization": 0.0,
    }


def _aggregate_sample_memory_metrics(metrics_list) -> dict:
    metrics_list = [metrics for metrics in metrics_list if metrics]
    if not metrics_list:
        return _empty_memory_metrics()

    base = {}
    numeric_keys = set().union(*(metrics.keys() for metrics in metrics_list))
    for key in numeric_keys:
        values = [metrics.get(key, 0) for metrics in metrics_list]
        if all(isinstance(value, (int, float)) for value in values):
            base[key] = max(values)
            base[f"mean_{key}"] = sum(values) / max(len(values), 1)

    resident = float(base.get("logical_resident_mb", 0.0))
    dense = float(base.get("logical_dense_equivalent_mb", 0.0))
    base["logical_compression_ratio"] = resident / dense if dense > 0 else 0.0
    base["logical_compression_x"] = dense / resident if resident > 0 else 0.0
    base["pool_utilization"] = (
        float(base.get("active_pool_blocks", 0)) / max(float(base.get("allocated_pool_blocks", 0)), 1.0)
    )
    base["capacity_utilization"] = (
        float(base.get("allocated_pool_blocks", 0)) / max(float(base.get("pool_blocks", 0)), 1.0)
    )
    physical_total = float(base.get("physical_total_kv_mb", 0.0))
    base["physical_compression_x"] = dense / physical_total if physical_total > 0 else 0.0
    return {**_empty_memory_metrics(), **base}


def build_qasper_prompt(row) -> str:
    return (
        "Read the paper excerpt below and answer the question using only the provided context.\n\n"
        f"Context:\n{row['context']}\n\n"
        f"Question: {row['input']}\n\n"
        "Answer concisely.\nAnswer:"
    )


def _token_length(tokenizer, prompt_text: str) -> int:
    return len(
        tokenizer(
            prompt_text,
            add_special_tokens=True,
            truncation=False,
            verbose=False,
        )["input_ids"]
    )


def _tokenize_left_truncated(tokenizer, prompt_text: str, max_input_tokens: int):
    old_truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    try:
        return tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=int(max_input_tokens),
            verbose=False,
        )
    finally:
        tokenizer.truncation_side = old_truncation_side


def load_qasper_samples(
    tokenizer,
    dataset_name: str = "qasper_e",
    fallback_dataset_name: str = "qasper",
    num_samples: int = 50,
    max_input_tokens: int = 1900,
):
    from datasets import load_dataset

    selected = []
    seen_ids = set()

    for config_name in (dataset_name, fallback_dataset_name):
        try:
            dataset = load_dataset("THUDM/LongBench", config_name, split="test", trust_remote_code=True)
        except Exception as exc:
            if config_name == fallback_dataset_name:
                raise
            print(f"Skipping LongBench config {config_name!r}: {exc}")
            continue
        for row in dataset:
            row_id = row.get("_id", f"{config_name}-{len(selected)}")
            if row_id in seen_ids:
                continue
            prompt_text = build_qasper_prompt(row)
            raw_prompt_tokens = _token_length(tokenizer, prompt_text)
            truncated_inputs = _tokenize_left_truncated(tokenizer, prompt_text, max_input_tokens)
            input_tokens = int(truncated_inputs["input_ids"].shape[1])
            selected.append(
                {
                    "_id": row_id,
                    "dataset": config_name,
                    "prompt": prompt_text,
                    "raw_prompt_tokens": raw_prompt_tokens,
                    "input_tokens": input_tokens,
                    "answers": list(row["answers"]),
                }
            )
            seen_ids.add(row_id)
            if len(selected) == num_samples:
                return selected

    raise RuntimeError(
        f"Only found {len(selected)} Qasper samples, expected {num_samples}."
    )


def _greedy_generate(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int,
    runtime_snapshot,
    max_input_tokens: int = 1900,
):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = _tokenize_left_truncated(tokenizer, prompt_text, max_input_tokens)
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")
    input_tokens = int(input_ids.shape[1])
    if input_tokens > max_input_tokens:
        raise RuntimeError(f"Qasper input has {input_tokens} tokens, exceeding {max_input_tokens}.")

    past_key_values = _prepare_tierkv_cache(runtime_snapshot)

    generated_tokens = []
    current_input_ids = input_ids
    full_attention_mask = torch.ones(
        (attention_mask.size(0), input_tokens + max_new_tokens),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    full_attention_mask[:, :input_tokens].copy_(attention_mask)
    current_attention_mask = full_attention_mask[:, :input_tokens]

    with torch.no_grad():
        for step_idx in range(max_new_tokens):
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
            current_attention_mask = full_attention_mask[:, : input_tokens + step_idx + 1]

    if generated_tokens:
        completion_ids = torch.cat(generated_tokens, dim=1)
        completion_text = tokenizer.decode(completion_ids[0], skip_special_tokens=True).strip()
    else:
        completion_text = ""

    mem_metrics = {}
    if getattr(past_key_values, "is_tierkv_cache", False):
        mem_metrics = past_key_values.runtime.collect_memory_metrics()
        _reset_tierkv_cache(past_key_values)
    return completion_text, input_tokens, mem_metrics


def evaluate_qasper(
    model,
    tokenizer,
    num_samples: int = 50,
    max_input_tokens: int = 1900,
    max_new_tokens: int = 32,
    dataset_name: str = "qasper_e",
    fallback_dataset_name: str = "qasper",
) -> dict:
    runtime_snapshot = _snapshot_tierkv_runtime()
    samples = load_qasper_samples(
        tokenizer,
        dataset_name=dataset_name,
        fallback_dataset_name=fallback_dataset_name,
        num_samples=num_samples,
        max_input_tokens=max_input_tokens,
    )
    model.eval()

    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    examples = []
    f1_sum = 0.0
    generated_token_count = 0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()

    sample_memory_metrics = []
    for sample in samples:
        completion_text, input_tokens, mem_metrics = _greedy_generate(
            model,
            tokenizer,
            sample["prompt"],
            max_new_tokens=max_new_tokens,
            runtime_snapshot=runtime_snapshot,
            max_input_tokens=max_input_tokens,
        )
        if mem_metrics:
            sample_memory_metrics.append(mem_metrics)
        f1 = max_token_f1(completion_text, sample["answers"])
        f1_sum += f1
        generated_token_count += max_new_tokens
        examples.append(
            {
                "_id": sample["_id"],
                "dataset": sample["dataset"],
                "input_tokens": input_tokens,
                "raw_prompt_tokens": sample["raw_prompt_tokens"],
                "prediction": completion_text,
                "answers": sample["answers"],
                "f1": f1,
                "mem_metrics": mem_metrics,
            }
        )

    input_lengths = [example["input_tokens"] for example in examples]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_memory_mb = None
    elapsed_seconds = time.time() - start_time

    return {
        "dataset": dataset_name,
        "fallback_dataset": fallback_dataset_name,
        "num_samples": len(examples),
        "mean_f1": f1_sum / max(len(examples), 1),
        "peak_memory_mb": peak_memory_mb,
        "tokens_per_sec": generated_token_count / max(elapsed_seconds, 1e-6),
        "avg_input_tokens": sum(input_lengths) / max(len(input_lengths), 1),
        "min_input_tokens": min(input_lengths) if input_lengths else 0,
        "max_input_tokens": max(input_lengths) if input_lengths else 0,
        "mem_metrics": _aggregate_sample_memory_metrics(sample_memory_metrics),
        "examples": examples,
    }


def _format_number(value):
    if value is None:
        return "n/a"
    if isinstance(value, float) and not math.isfinite(value):
        return "inf"
    return f"{value:.2f}"


def format_markdown_table(results) -> str:
    lines = [
        "| Configuration | Peak Memory (MB) | Qasper F1 | Avg Input Tokens | Tokens/s |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for result in results:
        qasper = result["qasper"]
        lines.append(
            "| "
            f"{result['configuration']} | "
            f"{_format_number(result['peak_memory_mb'])} | "
            f"{qasper['mean_f1'] * 100:.2f} | "
            f"{qasper['avg_input_tokens']:.0f} | "
            f"{_format_number(result['tokens_per_sec'])} |"
        )

    return "\n".join(lines)


def format_markdown_table_with_memory(results) -> str:
    """Extended sanity table including memory-realism metrics for TierKV rows."""
    lines = [
        "| Configuration | Peak Mem (MB) | Qasper F1 | Avg Tokens | tok/s | HOT/WARM/COLD | Logical Resident (MB) | Dense Eq KV (MB) | Physical KV (MB) | Logical Compression | Physical Compression | Pool Util |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        qasper = result["qasper"]
        m = result.get("mem_metrics") or qasper.get("mem_metrics", {})
        hot = m.get("hot_blocks", "—")
        warm = m.get("warm_blocks", "—")
        cold = m.get("cold_blocks", "—")
        resident = f"{m['logical_resident_mb']:.1f}" if "logical_resident_mb" in m else "—"
        dense = f"{m['logical_dense_equivalent_mb']:.1f}" if "logical_dense_equivalent_mb" in m else "—"
        physical = f"{m['physical_total_kv_mb']:.1f}" if "physical_total_kv_mb" in m else "—"
        compression = f"{m['logical_compression_x']:.2f}x" if "logical_compression_x" in m else "—"
        physical_compression = f"{m['physical_compression_x']:.2f}x" if "physical_compression_x" in m else "—"
        pool_util = f"{m['pool_utilization'] * 100:.1f}%" if "pool_utilization" in m else "—"
        lines.append(
            "| "
            f"{result['configuration']} | "
            f"{_format_number(result['peak_memory_mb'])} | "
            f"{qasper['mean_f1'] * 100:.2f} | "
            f"{qasper['avg_input_tokens']:.0f} | "
            f"{_format_number(result['tokens_per_sec'])} | "
            f"{hot}/{warm}/{cold} | "
            f"{resident} | "
            f"{dense} | "
            f"{physical} | "
            f"{compression} | "
            f"{physical_compression} | "
            f"{pool_util} |"
        )
    return "\n".join(lines)


def format_throughput_scaling_table(results) -> str:
    lines = [
        "| Context Tokens | Baseline tok/s | Pure Quant tok/s | Paged-TierKV tok/s | Paged vs Baseline |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        baseline = result["Baseline"]
        paged = result["Paged-TierKV"]
        ratio = paged / max(baseline, 1e-6)
        lines.append(
            "| "
            f"{result['context_tokens']} | "
            f"{baseline:.2f} | "
            f"{result['Pure Quantization']:.2f} | "
            f"{paged:.2f} | "
            f"{ratio:.2f}x |"
        )
    return "\n".join(lines)
