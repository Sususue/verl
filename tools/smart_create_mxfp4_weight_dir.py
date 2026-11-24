#!/usr/bin/env python3
# smart_create_mxfp4_weight_dir.py — 在线量化描述目录生成 (PLAN 4.2)
# 只量化路由专家 w1/w2/w3; 其余 FLOAT(原 dtype 原数值); 大文件软链接, 小文件拷贝
import argparse
import json
import re
import shutil
from pathlib import Path
from safetensors import safe_open

EXPERT = re.compile(
    r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.weight$"
)

def create_online_dir(src: Path, dst: Path):
    src, dst = src.resolve(), dst.resolve()
    if not src.is_dir():
        raise ValueError(f"Invalid source: {src}")
    if src == dst or src in dst.parents or dst in src.parents:
        raise ValueError("Source and destination must be independent")
    if dst.exists():
        raise FileExistsError(dst)
    if (src / "quant_model_description.json").exists():
        raise ValueError("Source already has a quantization description")

    config = json.loads((src / "config.json").read_text())
    index = json.loads((src / "model.safetensors.index.json").read_text())
    counts = config["num_experts_per_layer"]
    if len(counts) != config["num_hidden_layers"]:
        raise ValueError("Invalid per-layer expert counts")
    if counts != [256] * 3 + [16] * 40:
        raise ValueError("Unexpected model; verify top16 pruning configuration")
    if config.get("quantization_config"):
        raise ValueError("Expected a high-precision source config")

    names = set(index["weight_map"])
    expected = {
        f"layers.{layer}.ffn.experts.{expert}.{proj}.weight"
        for layer, count in enumerate(counts)
        for expert in range(count)
        for proj in ("w1", "w2", "w3")
    }
    matched = {name for name in names if EXPERT.fullmatch(name)}
    if matched != expected:
        raise ValueError({
            "missing": sorted(expected - matched)[:10],
            "unexpected": sorted(matched - expected)[:10],
        })

    # 分片头部检查: 全部目标矩阵 BF16 + K 可分组(不加载整个模型)
    by_shard = {}
    for name in sorted(expected):
        by_shard.setdefault(index["weight_map"][name], []).append(name)
    for shard, keys in by_shard.items():
        with safe_open(str(src / shard), framework="pt", device="cpu") as f:
            for name in keys:
                tensor = f.get_slice(name)
                shape = tensor.get_shape()
                if tensor.get_dtype() != "BF16":
                    raise ValueError((name, tensor.get_dtype()))
                if len(shape) != 2 or shape[-1] % 64:
                    raise ValueError((name, shape))

    # 临时目录完整生成后再改名; 失败保留临时目录供排错
    staging = dst.with_name(dst.name + ".creating")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    for file in src.rglob("*"):
        if not file.is_file():
            continue
        target = staging / file.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        if file.stat().st_size > 10 * 1024 * 1024:
            target.symlink_to(file.resolve())
        else:
            shutil.copy2(file, target)

    description = {"quant_method": "ascend", "group_size": 32}
    description.update({
        name: "W4A8_MXFP" if name in expected else "FLOAT"
        for name in sorted(names)
    })
    (staging / "quant_model_description.json").write_text(
        json.dumps(description, indent=2) + "\n"
    )
    manifest = {
        "source": str(src),
        "usage": "verl_online_quantization_only",
        "checkpoint_storage": "original_high_precision",
        "target_format": "ascendv1",
        "target_quantization": "W4A8_MXFP",
        "group_size": 32,
        "expert_matrices": len(expected),
        "verl_commit": "263653e749306287a5032140cce3d7833c1959d0",
    }
    (staging / "online_quant_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    staging.rename(dst)
    print(f"Created {dst}; expert matrices={len(expected)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    args = parser.parse_args()
    create_online_dir(args.src, args.dst)
