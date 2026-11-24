"""A shared FP32 routing contract for DSV4 text, before BF16 expert weighting."""
import types
import torch
import torch_npu
from torch.nn import functional as F

TILE=64

@torch.library.custom_op("dsv4_alignment::text_gate_logits",mutates_args=())
def fixed_logits(x:torch.Tensor,weight:torch.Tensor)->torch.Tensor:
    rows=x.reshape(-1,x.shape[-1]).to(torch.bfloat16).float()
    w=weight.float().contiguous()
    chunks=[]
    for block in rows.split(TILE):
        n=len(block)
        if n:
            padded=F.pad(block,(0,0,0,TILE-n)).contiguous()
            chunks.append(torch_npu.npu_linear(padded,w,None)[:n])
    y=torch.cat(chunks) if chunks else rows.new_empty((0,len(w)))
    return y.reshape(*x.shape[:-1],len(w))

@fixed_logits.register_fake
def fake_logits(x,weight):
    return x.new_empty((*x.shape[:-1],weight.shape[0]),dtype=torch.float32)

@torch.library.custom_op("dsv4_alignment::text_gate_weights",mutates_args=())
def dense_coefficients(logits:torch.Tensor,chosen:torch.Tensor,scale:float)->torch.Tensor:
    rows=logits.reshape(-1,logits.shape[-1])
    chosen=chosen.reshape(-1,chosen.shape[-1])
    chunks=[]
    for start in range(0,len(rows),TILE):
        x=rows[start:start+TILE]
        n=len(x)
        ids=chosen[start:start+TILE].long().sort(-1).values
        # Sort before normalization so both sides sum the same ordered operands.
        picked=F.pad(x.gather(1,ids),(0,0,0,TILE-n))
        scores=F.softplus(picked).sqrt()
        parts=(scores/(scores.sum(-1,keepdim=True)+1e-20))*scale
        dense=torch.zeros_like(x)
        # Accumulate duplicate hash slots before the single low-precision cast.
        for j in range(ids.shape[-1]):
            index=ids[:,j:j+1]
            dense.scatter_(1,index,dense.gather(1,index)+parts[:n,j:j+1])
        chunks.append(dense)
    return torch.cat(chunks) if chunks else rows.clone()

@dense_coefficients.register_fake
def fake_weights(logits,chosen,scale):
    return logits.new_empty((chosen.shape[0],logits.shape[-1]))

def serving_slots(logits,chosen,scale,is_hash):
    dense=dense_coefficients(logits,chosen,float(scale))
    if is_hash:
        all_ids=torch.arange(logits.shape[-1],device=logits.device)[None].expand_as(dense)
        selected=torch.zeros_like(dense,dtype=torch.bool).scatter_(1,chosen.long(),True)
        priority=torch.where(selected,all_ids,all_ids+logits.shape[-1])
        desired=priority.topk(chosen.shape[-1],dim=1,largest=False).indices
    else:
        desired=chosen
    ids=desired.long().sort(-1).values
    return dense.gather(1,ids),ids

def install_actor(roots):
    from mindspeed_bridge.models.deepseek_v4_flash.modelling_deepseek_v4_flash.router import _get_input_ids,_is_hash_layer
    from verl.utils.debug.train_alignment import with_local_backward
    routers=[m for root in roots for name,m in root.named_modules() if name.endswith(".mlp.router")]
    assert len(routers)==43
    for router in routers:
        if hasattr(router,"_text_gate_gating"):continue
        router._text_gate_gating=router.gating
        router._text_gate_routing=router.routing
        def gating(self,x):
            assert self.bias is None and self.config.moe_router_dtype=="fp32"
            y=fixed_logits(x,self.weight)
            return with_local_backward(y,self._text_gate_gating(x)) if torch.is_grad_enabled() else y
        def routing(self,logits,padding_mask=None):
            old,mask=self._text_gate_routing(logits,padding_mask)
            rows=logits.reshape(-1,logits.shape[-1])
            token_ids=_get_input_ids() if _is_hash_layer(self) else None
            if token_ids is not None:
                token_ids=token_ids.t().reshape(-1)[:len(rows)].long()
                token_ids=torch.where((token_ids>=0)&(token_ids<len(self.tid2eid)),token_ids,0)
                ids=self.tid2eid[token_ids].long()
            else:
                ids=mask.int().topk(self.topk,dim=-1).indices
            values=dense_coefficients(rows,ids,float(self.config.moe_router_topk_scaling_factor))
            values=values.masked_fill(~mask,0).to(old.dtype)
            return (with_local_backward(values,old) if torch.is_grad_enabled() else values),mask
        router.gating=types.MethodType(gating,router)
        router.routing=types.MethodType(routing,router)
