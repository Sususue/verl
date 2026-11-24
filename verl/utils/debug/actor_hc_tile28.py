"""Measure one shape policy on the actual differentiable Bridge mHC pre operator."""
import json
import os
from pathlib import Path

import torch


def tiled(call, x, *params):
    sequence, batch, streams, hidden = x.shape
    if sequence * batch == 0:
        return call(x, *params)
    rows = x.transpose(0, 1).reshape(-1, 1, streams, hidden)
    pieces = []
    for chunk in rows.split(64):
        n = len(chunk)
        if n < 64:
            chunk = torch.cat([chunk, chunk.new_zeros((64-n, 1, streams, hidden))])
        pieces.append([v[:n] for v in call(chunk.contiguous(), *params)])
    return tuple(torch.cat(parts).reshape(batch, sequence, *parts[0].shape[2:]).transpose(0, 1).contiguous()
                 for parts in zip(*pieces))


def stats(actual, expected):
    x = actual.detach().cpu().reshape_as(expected)
    difference = abs(x.float() - expected.float())
    return dict(n=difference.numel(), neq=int(difference.count_nonzero()), max_abs=float(difference.max()))


def compare_serving(run, call, params, device):
    files = sorted((run/'hc_shape').glob('*/L0_attn_pre_57.pt'))
    if not files:
        return dict(error='missing captured native L0 57-token pre')
    data = torch.load(files[0], map_location='cpu', weights_only=True)
    reference_params = data['params']
    weights = params[:3]
    evidence = [stats(v, t) for v,t in zip(weights,reference_params)]
    results = dict(source=str(files[0]), parameters=evidence, calls={})
    if any(item['neq'] for item in evidence):
        return results
    for n in (57,65,114):
        path = files[0].with_name(f'L0_attn_pre_{n}.pt')
        if not path.exists():
            continue
        capture=torch.load(path,map_location='cpu',weights_only=True)
        x=capture['inputs'][0].to(device).unsqueeze(1)
        native=capture['outputs']
        row={}
        for label,values in (('native_shape',call(x,*params)),('tile64',tiled(call,x,*params))):
            post,comb,collapsed=values
            row[label]=[stats(v,e) for v,e in zip((collapsed,post,comb),native)]
        results['calls'][n]=row
    return results


def configure(worker, mode):
    from mindspeed_bridge.models.deepseek_v4_flash.modelling_deepseek_v4_flash import mhc_fused
    if not hasattr(mhc_fused, '_original_pre_28'):
        mhc_fused._original_pre_28=mhc_fused.fused_pre_sinkhorn
    original=mhc_fused._original_pre_28
    if mode not in ('actor_hc_tile','actor_hc_tile_combined','actor_hc_tile_combined_repeat'):
        mhc_fused.fused_pre_sinkhorn=original
        return
    run=Path(os.environ['DSV4_QAT_RUN'])
    dest=run/'actor_hc'/str(os.getpid())/mode
    dest.mkdir(parents=True,exist_ok=True)
    n=0
    def forward(x,*params):
        nonlocal n
        n+=1
        if n==1 and worker.rank==0:
            try:
                measured=compare_serving(run,original,params,x.device)
            except Exception as exc:
                measured=dict(error=repr(exc))
            (dest/'same_operand_comparison.json').write_text(json.dumps(measured,indent=2))
        if n in (1,86) or n%86==0:
            (dest/'coverage.json').write_text(json.dumps(dict(calls=n,dtype=str(x.dtype),shape=list(x.shape))))
        return tiled(original,x,*params)
    mhc_fused.fused_pre_sinkhorn=forward
