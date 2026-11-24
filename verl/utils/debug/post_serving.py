"""A forward-only ablation of the two real mHC post implementations on identical Actor operands."""
import os
from pathlib import Path
import torch
from verl.utils.debug.train_alignment import with_local_backward

def configure(worker,label):
    from mindspeed_bridge.models.deepseek_v4_flash.modelling_deepseek_v4_flash import mhc_fused
    if not hasattr(mhc_fused,'_independent_post_base'):
        mhc_fused._independent_post_base=mhc_fused.fused_h_post_bda
    base=mhc_fused._independent_post_base
    active=label.startswith('actor_paged_combined_shared_fc1_post')
    count=0
    @torch.no_grad()
    def serving(h_res,original_residual,h_post,x,bias=None):
        nonlocal count
        assert os.environ.get("DSV4_ALIGN_VARIANT_LABEL")==label and bias is None
        import vllm_ascend.vllm_ascend_C
        op=torch.ops._C_ascend.npu_hc_post
        result=op(x.transpose(0,1).contiguous(),original_residual.transpose(0,1).contiguous(),
                  h_post.float().transpose(0,1).contiguous(),h_res.float().transpose(0,1).contiguous())
        count+=1
        if count in (1,86):
            path=Path(os.environ['DSV4_QAT_RUN'])/'post_hits'/str(os.getpid())/label
            path.mkdir(exist_ok=True,parents=True);(path/'count').write_text(str(count))
        return result.transpose(0,1).reshape_as(original_residual).contiguous()
    def train_post(*args,**kwargs):
        value=serving(*args,**kwargs)
        return with_local_backward(value,base(*args,**kwargs)) if torch.is_grad_enabled() else value
    choice=train_post if active else base
    mhc_fused.fused_h_post_bda=choice
    roots=worker.actor.engine.module
    for root in roots if isinstance(roots,(list,tuple)) else [roots]:
        for layer in root.modules():
            if hasattr(layer,'_h_post_bda_op') and layer._h_post_bda_op is base:
                layer._post_enable_1123=True
            if getattr(layer,'_post_enable_1123',False):layer._h_post_bda_op=choice
