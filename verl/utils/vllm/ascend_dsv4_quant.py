#!/usr/bin/env python3
# ascend_dsv4_quant.py — DSV4 W4A8_MXFP 在线量化: 精确名称映射与专家量化 (PLAN 5.2)
import re
import torch
import torch_npu

RAW_EXPERT = re.compile(
    r"^(?:model.)?layers\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.weight$"
)
HF_EXPERT = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|down_proj|up_proj)\.weight$"
)
PROJECTION = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}

def canonical_expert_name(name):
    """宽松匹配 bridge/HF/带model前缀等专家名变体, 返回规范名; 非专家名返回 None."""
    import re as _re
    m = _re.match(
        r"^(?:model\.)?layers\.(\d+)\.(?:ffn|mlp)\.experts\.(\d+)\."
        r"(w1|w2|w3|gate_proj|down_proj|up_proj)\.weight$",
        name,
    )
    if not m:
        return None
    layer, expert, proj = m.groups()
    proj_map = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj",
                "gate_proj": "gate_proj", "down_proj": "down_proj", "up_proj": "up_proj"}
    return (
        f"model.layers.{layer}.mlp.experts.{expert}."
        f"{proj_map[proj]}.weight"
    )

def build_expert_targets(description):
    targets = set()
    for name, kind in description.items():
        if not isinstance(kind, str) or not kind.startswith("W"):
            continue
        if kind != "W4A8_MXFP":
            raise ValueError(f"Unexpected quantization: {name}={kind}")
        canonical = canonical_expert_name(name)
        if canonical is None:
            raise ValueError(f"Non-routed parameter requested quantization: {name}")
        targets.add(canonical)
    if len(targets) != 4224:
        raise ValueError(f"Expected 4224 expert matrices, got {len(targets)}")
    return targets

def quantize_expert_matrix(weight):
    if weight.device.type != "npu" or weight.dtype != torch.bfloat16:
        raise ValueError((weight.device, weight.dtype))
    if weight.ndim != 2 or weight.shape[-1] % 64:
        raise ValueError(weight.shape)
    out_features, in_features = weight.shape
    packed, scale = torch_npu.npu_dynamic_mx_quant(
        weight.contiguous(), axis=-1,
        dst_type=torch_npu.float4_e2m1fn_x2,
        block_size=32, round_mode="rint", scale_alg=0,
    )
    packed = packed.view(torch.uint8)
    scale = scale.view(torch.uint8).reshape(out_features, in_features // 32)
    if packed.shape != (out_features, in_features // 2):
        raise ValueError(packed.shape)
    return packed, scale

def iter_ascend_dsv4_expert_weights(weights, description):
    targets = build_expert_targets(description)
    seen = set()  # 仅当前调用/bucket 内去重；完整同步覆盖统计见后文
    for name, weight in weights:
        canonical = canonical_expert_name(name)
        if canonical is None:
            # 遇到未支持的融合/3D专家输入时显式报错，不能静默透传
            if ".experts." in name:
                raise ValueError(f"Unsupported routed expert name: {name}")
            yield name, weight
            continue
        if canonical not in targets or canonical in seen:
            raise ValueError(f"Missing target or duplicate expert: {name}")
        seen.add(canonical)
        packed, scale = quantize_expert_matrix(weight)
        yield name, packed
        yield name + "_scale", scale
