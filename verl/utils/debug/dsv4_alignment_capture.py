"""Bounded captures from the real rollout MoE backend after initial refit."""
import os
from pathlib import Path
import json

_seen = set()


def capture_moe(layer, x, weights, ids, output, context):
    if os.environ.get("DSV4_ALIGN_CAPTURE") != "1":
        return
    import torch
    root = Path(os.environ["DSV4_QAT_RUN"])
    pid = os.getpid()
    if not (root / "audit" / f"refit_{pid}.jsonl").exists():
        return
    output = output.routed_out if hasattr(output, "routed_out") else output
    phase = "prefill" if x.shape[0] > 16 else "decode"
    if getattr(layer,'layer_name','')=='model.layers.4.mlp.experts' and x.shape[0] in (62,124):
        import hashlib
        from verl.utils.debug.dsv4_alignment_attention_core import _rollout_context
        t=_rollout_context.get('token_ids')
        if t is not None and t.numel()==len(x):
            t=t.flatten().contiguous().cpu()
            tag=hashlib.sha256(t.numpy().tobytes()).hexdigest()[:24]
            folder=Path(os.environ['DSV4_QAT_RUN'])/'mixed_route'/str(os.getpid())
            folder.mkdir(parents=True,exist_ok=True)
            file=folder/(tag+'.pt')
            if not file.exists():
                torch.save(dict(tokens=t,x=x.detach().cpu(),weights=weights.detach().cpu(),ids=ids.detach().cpu(),out=output.detach().cpu()),file)
    if getattr(layer, "layer_name", "") not in ("model.layers.0.mlp.experts","model.layers.8.mlp.experts","model.layers.29.mlp.experts") or x.shape[0] not in (57,65,114,130):
        return
    if torch.distributed.is_initialized() and torch.distributed.get_rank() not in (0,3):
        return
    phase += str(len(x))
    key = (id(layer), phase)
    if key in _seen:
        return
    _seen.add(key)
    directory = root / "captures"
    directory.mkdir(exist_ok=True)
    stem = f"{pid}_{id(layer)}_{phase}"
    attrs = {k: v for k, v in vars(layer).items() if isinstance(v, (str, int, float, bool))}
    metadata = {"pid": pid, "attrs": attrs, "phase": phase, "shape": list(x.shape),
                "activation": str(getattr(layer, "activation", None)),
                "backend": type(context.moe_comm_method).__name__,
                "mega_moe": getattr(context, "use_mega_moe", None)}
    (directory / (stem + ".json")).write_text(json.dumps(metadata, indent=2))
    torch.save({"x": x[:130].detach().cpu(), "topk_weights": weights[:130].detach().cpu(),
                "topk_ids": ids[:130].detach().cpu(), "output": output[:130].detach().cpu()},
               directory / (stem + ".pt"))
