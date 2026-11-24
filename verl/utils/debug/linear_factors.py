"""Replay real first-layer projections from the long/short forward using the resident weights."""
import json
import os
from pathlib import Path

import torch


@torch.no_grad()
def run(worker):
    if worker.rank != 1:
        return None
    root=Path(os.environ["DSV4_QAT_RUN"])
    out=root/"linear_factors";out.mkdir(exist_ok=True)
    modules=worker.actor.engine.module
    modules=modules if isinstance(modules,(list,tuple)) else [modules]
    def load(case,file):
        return torch.load(root/"replay_edges"/case/"1/operators"/file,weights_only=True,map_location="cpu")
    rows={}
    for group in modules:
        for name,m in group.named_modules():
            marker="decoder.layers.0."
            if marker not in name:
                continue
            suffix=name.split(marker,1)[1]
            if suffix not in ("self_attention.linear_q_up_proj","self_attention.linear_q_down_proj",
                              "self_attention.linear_kv_proj","self_attention.linear_proj",
                              "mlp.shared_experts.linear_fc1","mlp.shared_experts.linear_fc2"):
                continue
            f="L00_"+suffix.replace(".","_")+".pt"
            short,long=(load(case,f) for case in ("long_first8","long_full"))
            if "output0" not in short or "input0" not in short:
                continue
            x,y=short["input0"],short["output0"]
            def cmp(x,y):return int((x.cpu().reshape_as(y)!=y).sum())
            stats=dict(input_change=cmp(long["input0"],x),output_change=cmp(long["output0"],y),tests={})
            weight=m.weight.detach()
            if weight.ndim!=2 or x.shape[-1]!=weight.shape[1] or y.shape[-1]!=weight.shape[0]:
                continue
            x=x.reshape(57,-1)
            for n in (57,65,194,285,569):
                inp=x[torch.arange(n)%57].contiguous().to(weight.device)
                for mode in ("linear","invariant"):
                    val=torch.nn.functional.linear(inp,weight) if mode=="linear" else torch.ops.batch_invariant_ops.npu_mm_batch_invariant(inp,weight.T)
                    stats["tests"][f"{n}/{mode}"]=dict(short=cmp(val[:57],y),long=cmp(val[:57],long["output0"]))
            rows[suffix]=stats
    (out/"rank1.json").write_text(json.dumps(rows,indent=2))
    return True
