"""Opt-in stage capture. Invoked only from the normal model forward thread."""
from pathlib import Path
import os
import torch

_seen = set()


def save_stage(side, layer, phase, stage, values):
    if os.environ.get('DSV4_ALIGN_EXPERT_STAGES') != '1' or layer != 29:
        return
    if side == 'actor' and os.getenv('DSV4_ALIGN_VARIANT_LABEL') != 'actor_paged_combined_shared_fc1_post_compress_order':
        return
    if side == 'rollout' and phase != 'prefill57':
        return
    key = (side, layer, phase, stage)
    if key in _seen:
        return
    if side == 'rollout':
        audit = Path(os.environ['DSV4_QAT_RUN'])/'audit'/f'refit_{os.getpid()}.jsonl'
        if not audit.exists():
            return

    def cpu(value):
        if isinstance(value, torch.Tensor):
            if 'float8' in str(value.dtype):
                return {'uint8': value.detach().view(torch.uint8).cpu(),
                        'dtype': str(value.dtype), 'shape': list(value.shape)}
            return value.detach().cpu()
        if isinstance(value, dict):
            return {k: cpu(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cpu(v) for v in value]
        return value

    directory = Path(os.environ['DSV4_QAT_RUN'])/'expert_stages'/side/str(os.getpid())
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({'layer':layer,'phase':phase,'stage':stage,'values':cpu(values)},
               directory/f'layer{layer:02d}_{stage}_{phase}.pt')
    _seen.add(key)


def actor_stage(module, stage, values):
    save_stage('actor',getattr(module,'_alignment_layer_number',-1),'full',stage,values)


def rollout_stage(layer, stage, values):
    import re
    match = re.search(r'layers\.(\d+)\.', getattr(layer,'layer_name',''))
    if match:
        save_stage('rollout',int(match.group(1)),getattr(layer,'_alignment_phase','unknown'),stage,values)
