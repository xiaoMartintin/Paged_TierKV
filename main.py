import modal

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LOCAL_PROJECT_ROOT = "/root/paged_tierkv"
LOCAL_MODELING_LLAMA_PATH = "/root/paged_tierkv/modeling_llama.py"
LOCAL_MODELING_LLAMA_MODULE = "transformers.models.llama.modeling_llama"
LOCAL_TIERKV_POLICY_PATH = "/root/paged_tierkv/tierkv_policy.py"
LOCAL_TIERKV_POLICY_MODULE = "tierkv_policy"
TIERKV_CPP = None

tierkv_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install(
        "numpy<2",
        "torch==2.4.1",
        "git+https://github.com/huggingface/transformers.git@main",
        "datasets==3.2.0",
        "accelerate==1.2.1",
        "ninja",
        "pybind11>=2.12"
    )
    .add_local_dir(".", remote_path="/root/paged_tierkv")
)

app = modal.App("paged-tierkv-baseline")


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
    import importlib.util
    import sys

    existing_module = sys.modules.get(LOCAL_TIERKV_POLICY_MODULE)
    if existing_module is not None and getattr(existing_module, "__file__", None) == LOCAL_TIERKV_POLICY_PATH:
        module = existing_module
    else:
        spec = importlib.util.spec_from_file_location(
            LOCAL_TIERKV_POLICY_MODULE,
            LOCAL_TIERKV_POLICY_PATH,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load tierkv_policy from {LOCAL_TIERKV_POLICY_PATH}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[LOCAL_TIERKV_POLICY_MODULE] = module
        spec.loader.exec_module(module)

    return (
        module.TierKVPolicyEngine,
        module.dequantize_int8_to_fp16,
        module.quantize_fp16_to_int8,
    )


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

    kv_tensor = torch.randn((1, 32, 16, 128), device="cuda", dtype=torch.float16)
    quantized_tensor, scale, zero_point = quantize_fp16_to_int8(kv_tensor)
    dequantized_tensor = dequantize_int8_to_fp16(quantized_tensor, scale, zero_point)

    assert quantized_tensor.shape == kv_tensor.shape
    assert dequantized_tensor.shape == kv_tensor.shape
    assert quantized_tensor.dtype == torch.uint8
    assert scale.dtype == torch.float16
    assert zero_point.dtype == torch.float16

    mse = torch.mean((dequantized_tensor.float() - kv_tensor.float()) ** 2).item()
    print(f"Quantization MSE: {mse:.8f}")

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
    print(f"Block Scores: {block_scores.tolist()}")
    print(f"Block States: {states}")
    assert states == [0, 0, 1, 0], f"Unexpected policy states: {states}"

    print("Policy and quantization integration test passed.")


def run_baseline_generation_benchmark():
    import time
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside the Modal container.")

    LlamaForCausalLM, LlamaConfig = load_local_llama_classes()

    print("Loading TinyLlama weights into local LlamaForCausalLM...")
    config = LlamaConfig.from_pretrained(MODEL_ID)
    model = LlamaForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = "User: Explain how KV cache growth affects LLM inference efficiency.\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")
    max_new_tokens = 50

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    generated_tokens = []
    current_input_ids = input_ids
    current_attention_mask = attention_mask
    past_key_values = None

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

    print("=== Baseline Results ===")
    print(f"Peak Memory: {peak_mem:.2f} MB")
    print(f"Tokens/sec: {tokens_per_sec:.2f}")
    print("Generated Text Preview:")
    print(generated_text)


@app.function(
    image=tierkv_image,
    gpu="A10G",
    timeout=1800
)
def test_cpp_and_baseline():
    from torch.utils.cpp_extension import load

    print("Compiling C++ extension on cloud GPU...")
    tierkv_cpp = load(
        name="tierkv_cpp",
        sources=["/root/paged_tierkv/csrc/block_manager.cpp"],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=True
    )

    global TIERKV_CPP
    TIERKV_CPP = tierkv_cpp

    test_block_manager()
    test_policy_and_quantization()


@app.local_entrypoint()
def main():
    test_cpp_and_baseline.remote()
