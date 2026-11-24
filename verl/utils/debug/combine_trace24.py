"""Capture the real expert reductions for the first three layers on every participating rank."""
import os
from pathlib import Path

import torch


def install():
    from megatron.core.transformer.moe import token_dispatcher as dispatch
    if getattr(dispatch,'_combine_probe_installed',False):
        return
    dispatch._combine_probe_installed=True
    unpermute=dispatch.unpermute
    calls={}

    def measured(tokens,indices,*args,**kwargs):
        result=unpermute(tokens,indices,*args,**kwargs)
        label=os.environ.get('DSV4_ALIGN_VARIANT_LABEL')
        if label not in ('initial_reference','initial_repeat','initial_inline'):
            return result
        i=calls.get(label,0)
        calls[label]=i+1
        if i>=3:
            return result
        replay=unpermute(tokens,indices,*args,**kwargs)
        directory=Path(os.environ['DSV4_QAT_RUN'])/'combine_trace'/str(os.getpid())
        directory.mkdir(parents=True,exist_ok=True)
        payload=dict(input=tokens.detach().cpu(),indices=indices.detach().cpu(),
                     output=result.detach().cpu(),repeat=replay.detach().cpu(),
                     fused=kwargs.get('fused'),drop_and_pad=kwargs.get('drop_and_pad'))
        routing=kwargs.get('routing_map')
        if isinstance(routing,torch.Tensor):
            payload['routing_map']=routing.detach().cpu()
        torch.save(payload,directory/f'{label}_{i}.pt')
        return result
    dispatch.unpermute=measured
