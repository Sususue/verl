"""Use an identical BF16 output-projection shape for prefill, decode and the training forward."""
from typing import Optional
import torch
import torch_npu
from torch.nn import functional as F


@torch.library.custom_op("dsv4_alignment::fixed_wo_b",mutates_args=())
def fixed_wo_b(x:torch.Tensor,weight:torch.Tensor,bias:Optional[torch.Tensor])->torch.Tensor:
    rows=x.reshape(-1,x.shape[-1])
    if rows.shape[0]==0:
        return x.new_empty((*x.shape[:-1],weight.shape[0]))
    result=[]
    for block in rows.split(64):
        count=len(block)
        block=F.pad(block,(0,0,0,64-count))
        result.append(torch_npu.npu_linear(block.contiguous(),weight,bias)[:count])
    return torch.cat(result).reshape(*x.shape[:-1],weight.shape[0])


@fixed_wo_b.register_fake
def fake_wo_b(x,weight,bias):
    return x.new_empty((*x.shape[:-1],weight.shape[0]))


def install_actor(roots):
    from verl.utils.debug.train_alignment import with_local_backward
    mods=[m for root in roots for n,m in root.named_modules() if n.endswith(".self_attention.linear_proj")]
    assert len(mods)==43
    def project(layer,args,out):
        assert out[1] is None and layer.weight.dtype==args[0].dtype==torch.bfloat16
        assert getattr(layer,"bias", None) is None or not getattr(layer,"use_bias", False)
        with torch.no_grad():
            y=fixed_wo_b(args[0],layer.weight,None)
        return (with_local_backward(y,out[0]) if torch.is_grad_enabled() else y,None)
    for m in mods:
        if not hasattr(m,"_fixed_wo_b_handle"):
            m._fixed_wo_b_handle=m.register_forward_hook(project)
