import modal

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MIN_PROMPT_TOKENS = 512
MAX_NEW_TOKENS = 50
TIERKV_HOT_BUDGET = 8
NEEDLE_NUM_SAMPLES = 2
NEEDLE_CONTEXT_TOKENS = 1024
NEEDLE_MAX_NEW_TOKENS = 16
BENCHMARK_PARAGRAPH = (
    "Transformer inference over long contexts is dominated by key value cache growth, memory bandwidth limits, "
    "and the cost of repeatedly loading old activations during autoregressive decoding. In a production serving "
    "stack, each decoder layer stores key and value tensors for every previous token, so prompt prefill quickly "
    "creates a large resident working set that scales with sequence length, number of layers, number of heads, "
    "and head dimension. A tiered cache can protect the attention sink region and the newest blocks in full "
    "precision while quantizing older blocks, but the policy must react to real attention patterns instead of "
    "uniform heuristics or it will evict semantically critical context. Systems engineers therefore track "
    "logical to physical block mappings, preserve causal ordering, and balance memory savings against the cost "
    "of dequantization, temporary reconstruction buffers, and any quality loss introduced by reduced precision. "
    "When the context reaches hundreds of tokens, this tradeoff becomes visible in GPU peak memory, decode "
    "throughput, and answer quality, especially for prompts that interleave definitions, constraints, numerical "
    "details, and references that the model must revisit later in the response."
)
LOCAL_PROJECT_ROOT = "/root/paged_tierkv"
LOCAL_MODELING_LLAMA_PATH = "/root/paged_tierkv/modeling_llama.py"
LOCAL_MODELING_LLAMA_MODULE = "transformers.models.llama.modeling_llama"
LOCAL_TIERKV_POLICY_PATH = "/root/paged_tierkv/tierkv_policy.py"
LOCAL_TIERKV_POLICY_MODULE = "tierkv_policy"
LOCAL_TIERKV_EVAL_PATH = "/root/paged_tierkv/tierkv_eval.py"
LOCAL_TIERKV_EVAL_MODULE = "tierkv_eval"
TIERKV_CPP = None

tierkv_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install(
        "numpy<2",
        "torch==2.4.1",
        "git+https://github.com/huggingface/transformers.git@main",
        "accelerate==1.2.1",
        "ninja",
        "pybind11>=2.12"
    )
    .add_local_dir(".", remote_path="/root/paged_tierkv")
)

app = modal.App("paged-tierkv-baseline")


def load_tierkv_policy_module():
    import importlib.util
    import sys

    existing_module = sys.modules.get(LOCAL_TIERKV_POLICY_MODULE)
    if existing_module is not None and getattr(existing_module, "__file__", None) == LOCAL_TIERKV_POLICY_PATH:
        return existing_module

    spec = importlib.util.spec_from_file_location(
        LOCAL_TIERKV_POLICY_MODULE,
        LOCAL_TIERKV_POLICY_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load tierkv_policy from {LOCAL_TIERKV_POLICY_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[LOCAL_TIERKV_POLICY_MODULE] = module
    spec.loader.exec_module(module)
    return module


def load_local_llama_classes():
    import importlib.util
    import sys

    existing_module = sys.modules.get(LOCAL_MODELING_LLAMA_MODULE)
    if existing_module is not None and getattr(existing_module, "__file__", None) == LOCAL_MODELING_LLAMA_PATH:
        module = existing_module
    else:
        spec = importlib.util.spec_from_file_location(
            LOCAL_MODELING_LLAMA_MODULE,
            LOCAL_MODELING_LLAMA_PATH,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load local modeling_llama from {LOCAL_MODELING_LLAMA_PATH}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[LOCAL_MODELING_LLAMA_MODULE] = module
        spec.loader.exec_module(module)

    LlamaForCausalLM = module.LlamaForCausalLM
    LlamaConfig = module.LlamaConfig

    if getattr(LlamaForCausalLM, "config_class", None) is None:
        LlamaForCausalLM.config_class = LlamaConfig

    return LlamaForCausalLM, LlamaConfig


def load_tierkv_policy_symbols():
    module = load_tierkv_policy_module()
    return (
        module.TierKVPolicyEngine,
        module.dequantize_int8_to_fp16,
        module.quantize_fp16_to_int8,
    )


def load_tierkv_eval_module():
    import importlib.util
    import sys

    existing_module = sys.modules.get(LOCAL_TIERKV_EVAL_MODULE)
    if existing_module is not None and getattr(existing_module, "__file__", None) == LOCAL_TIERKV_EVAL_PATH:
        return existing_module

    spec = importlib.util.spec_from_file_location(
        LOCAL_TIERKV_EVAL_MODULE,
        LOCAL_TIERKV_EVAL_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load tierkv_eval from {LOCAL_TIERKV_EVAL_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[LOCAL_TIERKV_EVAL_MODULE] = module
    spec.loader.exec_module(module)
    return module


def build_long_benchmark_prompt(tokenizer, min_prompt_tokens=MIN_PROMPT_TOKENS):
    prompt_sections = []
    prompt_token_length = 0

    while prompt_token_length < min_prompt_tokens:
        prompt_sections.append(BENCHMARK_PARAGRAPH)
        prompt_text = "User: " + " ".join(prompt_sections) + "\nAssistant:"
        prompt_token_length = tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1]

    if prompt_token_length < min_prompt_tokens:
        raise RuntimeError("Failed to construct a sufficiently long benchmark prompt.")

    return prompt_text, prompt_token_length


def summarize_tieredkv_states(block_manager, num_layers):
    layer_summaries = {}
    warm_layer_count = 0
    cold_layer_count = 0
    total_warm_blocks = 0
    total_cold_blocks = 0

    for layer_idx in range(num_layers):
        states = block_manager.get_states(layer_idx)
        hot_count = sum(state == 0 for state in states)
        warm_count = sum(state == 1 for state in states)
        cold_count = sum(state == 2 for state in states)
        layer_summaries[layer_idx] = {
            "total_blocks": len(states),
            "hot": hot_count,
            "warm": warm_count,
            "cold": cold_count,
        }
        warm_layer_count += int(warm_count > 0)
        cold_layer_count += int(cold_count > 0)
        total_warm_blocks += warm_count
        total_cold_blocks += cold_count

    return layer_summaries, warm_layer_count, total_warm_blocks, cold_layer_count, total_cold_blocks


def test_block_manager():
    global TIERKV_CPP

    if TIERKV_CPP is None:
        raise RuntimeError("C++ extension must be loaded before running block manager tests.")

    seq_id = 7
    manager = TIERKV_CPP.BlockManager(total_blocks=8)

    allocated = [manager.allocate_block(seq_id) for _ in range(3)]
    assert allocated == [0, 1, 2], f"Unexpected allocated physical indices: {allocated}"
    assert manager.get_physical_indices(seq_id) == [0, 1, 2]

    manager.update_block_state(seq_id, 1, 1)
    assert manager.get_states(seq_id) == [0, 1, 0], f"Unexpected block states: {manager.get_states(seq_id)}"

    overflow_manager = TIERKV_CPP.BlockManager(total_blocks=1)
    overflow_manager.allocate_block(99)
    try:
        overflow_manager.allocate_block(99)
        raise AssertionError("Expected allocation beyond capacity to raise RuntimeError.")
    except RuntimeError:
        pass

    try:
        manager.update_block_state(seq_id, 1, 9)
        raise AssertionError("Expected invalid block state to raise RuntimeError.")
    except RuntimeError:
        pass

    try:
        manager.update_block_state(seq_id, 99, 1)
        raise AssertionError("Expected invalid logical_id to raise RuntimeError.")
    except RuntimeError:
        pass

    manager.free_sequence(-1)
    manager.free_sequence(seq_id)
    assert manager.get_physical_indices(seq_id) == []
    assert manager.get_states(seq_id) == []

    print("BlockManager integration test passed.")


def test_policy_and_quantization():
    import torch

    global TIERKV_CPP

    if TIERKV_CPP is None:
        raise RuntimeError("C++ extension must be loaded before running policy tests.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside the Modal container.")

    policy_module = load_tierkv_policy_module()
    (
        TierKVPolicyEngine,
        dequantize_int8_to_fp16,
        quantize_fp16_to_int8,
    ) = load_tierkv_policy_symbols()

    seq_id = 21
    manager = TIERKV_CPP.BlockManager(total_blocks=16)
    engine = TierKVPolicyEngine(block_size=16, block_manager=manager)

    for _ in range(4):
        manager.allocate_block(seq_id)

    def quantize_scalar_reference(tensor):
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

    def dequantize_scalar_reference(quantized_tensor, scale, zero_point):
        return ((quantized_tensor.to(torch.float16) - zero_point) * scale).to(torch.float16)

    kv_tensor = torch.randn((1, 32, 16, 128), device="cuda", dtype=torch.float16)
    scalar_quantized, scalar_scale, scalar_zero_point = quantize_scalar_reference(kv_tensor)
    scalar_dequantized = dequantize_scalar_reference(scalar_quantized, scalar_scale, scalar_zero_point)

    quantized_tensor, scale, zero_point = quantize_fp16_to_int8(kv_tensor)
    dequantized_tensor = dequantize_int8_to_fp16(quantized_tensor, scale, zero_point)

    assert quantized_tensor.shape == kv_tensor.shape
    assert dequantized_tensor.shape == kv_tensor.shape
    assert quantized_tensor.dtype == torch.uint8
    assert scale.shape == (1, 32, 16, 1)
    assert zero_point.shape == (1, 32, 16, 1)
    assert scale.dtype == torch.float16
    assert zero_point.dtype == torch.float16

    scalar_mse = torch.mean((scalar_dequantized.float() - kv_tensor.float()) ** 2).item()
    channel_wise_mse = torch.mean((dequantized_tensor.float() - kv_tensor.float()) ** 2).item()
    assert channel_wise_mse < scalar_mse, "Expected channel-wise quantization MSE to improve over scalar quantization."

    block_pattern = torch.cat(
        [
            torch.full((16,), 0.4, device="cuda", dtype=torch.float16),
            torch.full((16,), 0.9, device="cuda", dtype=torch.float16),
            torch.full((16,), 0.1, device="cuda", dtype=torch.float16),
            torch.full((16,), 0.2, device="cuda", dtype=torch.float16),
        ]
    )
    attn_weights = block_pattern.view(1, 1, 1, 64).repeat(1, 32, 1, 1)

    block_scores = engine.compute_block_scores(attn_weights)
    engine.enforce_budget(seq_id, block_scores, hot_budget=3)

    states = manager.get_states(seq_id)
    assert states == [0, 0, 1, 0], f"Unexpected policy states: {states}"

    cold_pool = policy_module.PhysicalKVPool()
    cold_physical_idx = 55
    cold_key = kv_tensor[:, :4, :8, :16].contiguous()
    cold_value = kv_tensor[:, 4:8, :8, :16].contiguous()
    cold_pool.store_hot_block(cold_physical_idx, cold_key, cold_value)
    cold_pool.demote_to_cold(cold_physical_idx)
    reconstructed_key, reconstructed_value = cold_pool.get_dequantized_block(
        cold_physical_idx,
        policy_module.COLD_STATE,
    )
    assert reconstructed_key.shape == cold_key.shape
    assert reconstructed_value.shape == cold_value.shape
    assert torch.count_nonzero(reconstructed_key) == 0
    assert torch.count_nonzero(reconstructed_value) == 0
    print("Policy and quantization integration test passed.")


def run_generation_benchmark(model, tokenizer, label, max_new_tokens=MAX_NEW_TOKENS, initial_past_key_values=None):
    import time
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside the Modal container.")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_text, prompt_token_length = build_long_benchmark_prompt(tokenizer)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")
    if input_ids.shape[1] < MIN_PROMPT_TOKENS:
        raise RuntimeError(f"{label} prompt token length regressed below {MIN_PROMPT_TOKENS}.")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    generated_tokens = []
    current_input_ids = input_ids
    current_attention_mask = attention_mask
    past_key_values = initial_past_key_values

    start = time.time()
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

    torch.cuda.synchronize()
    end = time.time()

    generated_ids = torch.cat([input_ids] + generated_tokens, dim=1)
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    tokens_per_sec = max_new_tokens / max(end - start, 1e-6)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return peak_mem, tokens_per_sec, generated_text, prompt_token_length


def load_official_baseline_assets():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return model, tokenizer


def load_local_tieredkv_assets():
    import torch
    from transformers import AutoTokenizer

    LlamaForCausalLM, LlamaConfig = load_local_llama_classes()
    config = LlamaConfig.from_pretrained(MODEL_ID)

    model = LlamaForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return model, tokenizer, config


def initialize_tierkv_runtime_for_row(policy_module, num_layers, hot_budget, demoted_state):
    global TIERKV_CPP

    if TIERKV_CPP is None:
        raise RuntimeError("C++ extension must be loaded before running TieredKV evaluation rows.")

    policy_module.reset_global_tierkv()
    tiered_cache = policy_module.initialize_global_tierkv(
        block_manager=TIERKV_CPP.BlockManager(),
        num_layers=num_layers,
        block_size=16,
        hot_budget=hot_budget,
    )
    runtime = tiered_cache.runtime
    runtime.demoted_state = demoted_state
    runtime.cold_block_policy = "zero"
    runtime.tiered_layer_indices = set(range(num_layers))
    return tiered_cache, runtime


def run_baseline_row(model, tokenizer, eval_module):
    peak_mem, tokens_per_sec, _, _ = run_generation_benchmark(
        model,
        tokenizer,
        label="Baseline",
    )
    needle = eval_module.evaluate_long_context(
        model,
        tokenizer,
        num_samples=NEEDLE_NUM_SAMPLES,
        context_tokens=NEEDLE_CONTEXT_TOKENS,
        max_new_tokens=NEEDLE_MAX_NEW_TOKENS,
    )

    return {
        "configuration": "Baseline",
        "peak_memory_mb": peak_mem,
        "needle": needle,
        "tokens_per_sec": tokens_per_sec,
    }


def run_tieredkv_row(model, tokenizer, config, eval_module, policy_module, label, hot_budget, demoted_state):
    tiered_cache, runtime = initialize_tierkv_runtime_for_row(
        policy_module,
        config.num_hidden_layers,
        hot_budget=hot_budget,
        demoted_state=demoted_state,
    )
    peak_mem, tokens_per_sec, _, _ = run_generation_benchmark(
        model,
        tokenizer,
        label=label,
        initial_past_key_values=tiered_cache,
    )

    _, _, total_warm_blocks, _, total_cold_blocks = summarize_tieredkv_states(
        runtime.block_manager,
        runtime.num_layers,
    )

    if demoted_state == policy_module.WARM_STATE and total_warm_blocks == 0:
        raise RuntimeError(f"{label} did not produce any WARM blocks.")
    if demoted_state == policy_module.COLD_STATE and total_cold_blocks == 0:
        raise RuntimeError(f"{label} did not produce any COLD blocks.")

    _, _ = initialize_tierkv_runtime_for_row(
        policy_module,
        config.num_hidden_layers,
        hot_budget=hot_budget,
        demoted_state=demoted_state,
    )
    needle = eval_module.evaluate_long_context(
        model,
        tokenizer,
        num_samples=NEEDLE_NUM_SAMPLES,
        context_tokens=NEEDLE_CONTEXT_TOKENS,
        max_new_tokens=NEEDLE_MAX_NEW_TOKENS,
    )

    policy_module.reset_global_tierkv()
    return {
        "configuration": label,
        "peak_memory_mb": peak_mem,
        "needle": needle,
        "tokens_per_sec": tokens_per_sec,
    }


def ablation_runner():
    import torch

    policy_module = load_tierkv_policy_module()
    eval_module = load_tierkv_eval_module()
    results = []

    policy_module.reset_global_tierkv()
    baseline_model, baseline_tokenizer = load_official_baseline_assets()
    results.append(run_baseline_row(baseline_model, baseline_tokenizer, eval_module))
    del baseline_model
    del baseline_tokenizer
    torch.cuda.empty_cache()

    tiered_model, tiered_tokenizer, tiered_config = load_local_tieredkv_assets()
    row_specs = [
        {
            "configuration": "Pure Quantization",
            "hot_budget": 0,
            "demoted_state": policy_module.WARM_STATE,
        },
        {
            "configuration": "Pure Sparsification",
            "hot_budget": 0,
            "demoted_state": policy_module.COLD_STATE,
        },
        {
            "configuration": "Paged-TierKV (budget=4)",
            "hot_budget": 4,
            "demoted_state": policy_module.WARM_STATE,
        },
        {
            "configuration": "Paged-TierKV (budget=8)",
            "hot_budget": 8,
            "demoted_state": policy_module.WARM_STATE,
        },
    ]

    for row_spec in row_specs:
        results.append(
            run_tieredkv_row(
                tiered_model,
                tiered_tokenizer,
                tiered_config,
                eval_module,
                policy_module,
                label=row_spec["configuration"],
                hot_budget=row_spec["hot_budget"],
                demoted_state=row_spec["demoted_state"],
            )
        )
        torch.cuda.empty_cache()

    del tiered_model
    del tiered_tokenizer
    policy_module.reset_global_tierkv()
    torch.cuda.empty_cache()

    markdown_table = eval_module.format_markdown_table(results)
    print(markdown_table)
    return results


@app.function(
    image=tierkv_image,
    gpu="A10G",
    timeout=1800
)
def test_cpp_and_baseline():
    from torch.utils.cpp_extension import load

    tierkv_cpp = load(
        name="tierkv_cpp",
        sources=["/root/paged_tierkv/csrc/block_manager.cpp"],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False
    )

    global TIERKV_CPP
    TIERKV_CPP = tierkv_cpp

    test_block_manager()
    test_policy_and_quantization()
    ablation_runner()


@app.local_entrypoint()
def main():
    test_cpp_and_baseline.remote()
