"""Preserve original non-expert FP32 checkpoint values for val-only DSV4 exports."""
import json
import struct
from functools import lru_cache
from pathlib import Path

from safetensors import safe_open


@lru_cache(maxsize=4)
def _source_fp32_parameters(model_path):
    root = Path(model_path)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    result = {}
    for shard in sorted(set(index.values())):
        with (root / shard).open("rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            metadata = json.loads(handle.read(size))
        for name, info in metadata.items():
            if name != "__metadata__" and info["dtype"] == "F32" and ".experts." not in name:
                result[name] = shard
    return result


def _source_name(name):
    if name.startswith("model."):
        name = name[len("model."):]
    if name.startswith("lm_head."):
        name = "head." + name[len("lm_head."):]
    return (name.replace(".self_attn.", ".attn.").replace(".mlp.", ".ffn.")
            .replace(".input_layernorm.", ".attn_norm.")
            .replace(".post_attention_layernorm.", ".ffn_norm.")
            .replace(".e_score_correction_bias", ".bias"))


def iter_dsv4_fp32_source_weights(weights, model_path):
    source = _source_fp32_parameters(str(model_path))
    for name, tensor in weights:
        key = _source_name(name)
        shard = source.get(key)
        if shard is None:
            yield name, tensor
            continue
        with safe_open(str(Path(model_path) / shard), framework="pt", device="cpu") as handle:
            original = handle.get_tensor(key)
        if tuple(original.shape) != tuple(tensor.shape):
            raise ValueError(f"FP32 source shape mismatch for {name}: {original.shape} != {tensor.shape}")
        yield name, original.to(device=tensor.device)
