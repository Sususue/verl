"""Capture layer-zero post inputs before any layout changes for an independent comparison."""
import os
from pathlib import Path
import torch

def configure(worker,label):
    from mindspeed_bridge.models.deepseek_v4_flash.modelling_deepseek_v4_flash import mhc_fused
    if not hasattr(mhc_fused,'_post_original_1123'):
        mhc_fused._post_original_1123=mhc_fused.fused_h_post_bda
    real=mhc_fused._post_original_1123
    roots=worker.actor.engine.module
    roots=roots if isinstance(roots,(list,tuple)) else [roots]
    cached=[layer for root in roots for layer in root.modules() if hasattr(layer,'_h_post_bda_op')]
    for layer in cached:
        if not hasattr(layer,'_post_saved_1123'):
            layer._post_saved_1123=layer._h_post_bda_op
        layer._h_post_bda_op=layer._post_saved_1123
    if label!='actor_paged_combined' or worker.rank!=0:
        mhc_fused.fused_h_post_bda=real
        return
    count=0
    def capture(h_res,original_residual,h_post,x,bias=None):
        nonlocal count
        y=real(h_res,original_residual,h_post,x,bias)
        if count<2:
            dest=Path(os.environ['DSV4_QAT_RUN'])/'actor_post'
            dest.mkdir(exist_ok=True)
            torch.save(dict(h_res=h_res.detach().cpu(),residual=original_residual.detach().cpu(),h_post=h_post.detach().cpu(),x=x.detach().cpu(),output=y.detach().cpu()),dest/f'post_{count}.pt')
        count+=1
        return y
    mhc_fused.fused_h_post_bda=capture
    for layer in cached:
        if layer._post_saved_1123 is real:
            layer._h_post_bda_op=capture
