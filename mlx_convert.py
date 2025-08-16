from mlx_lm.convert import convert


def mixed_quantization(layer_path, layer, model_config):
    if "lm_head" in layer_path or "embed_tokens" in layer_path:
        return {"bits": 6, "group_size": 64}
    elif hasattr(layer, "to_quantized"):
        return {"bits": 4, "group_size": 64}
    else:
        return False


convert(
    # hf_path="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    hf_path="dphn/Dolphin3.0-Qwen2.5-3b",
    # mlx_path="./DeepSeek-R1-0528-Qwen3-8B",
    mlx_path="./Dolphin3.0-Qwen2.5-3b",
    quantize=True,
    quant_predicate=mixed_quantization
)
