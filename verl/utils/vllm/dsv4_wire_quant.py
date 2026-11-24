"""Encode HF-format routed experts according to the rollout descriptor before IPC bucketing."""
import hashlib
import json
import logging
from pathlib import Path

import torch

from verl.utils.vllm.ascend_dsv4_quant import build_expert_targets, canonical_expert_name, quantize_expert_matrix
from verl.workers.rollout.utils import ensure_async_iterator

logger=logging.getLogger(__name__)


def rollout_description(config,model):
    if config.quantization!='ascend' or getattr(model.hf_config,'model_type',None)!='deepseek_v4':
        return None
    path=Path(config.quantization_config_file) if config.quantization_config_file else Path(model.local_path)/'quant_model_description.json'
    desc=json.loads(path.read_text())
    if 'W4A8_MXFP' not in desc.values():return None
    if int(desc.get('group_size',32))!=32:raise ValueError('Unsupported MXFP4 group size')
    return desc


def spec(description):
    if int(description.get('group_size',32))!=32:
        raise ValueError('Unsupported rollout MXFP4 group size')
    targets=build_expert_targets(description)
    raw=json.dumps(sorted(targets),separators=(',',':')).encode()
    return targets,dict(format='dsv4_w4a8_mxfp_v1',block_size=32,targets_sha256=hashlib.sha256(raw).hexdigest())


class PayloadAudit:
    def __init__(self,targets):
        self.targets=targets
        self.shapes=({}, {})
        self.bytes=0
        self.samples={}
        self.sample_names={min(targets),max(targets)}

    def observe(self,name,value):
        scale=name.endswith('.weight_scale')
        key=canonical_expert_name(name.removesuffix('_scale') if scale else name)
        if key is None:
            if '.experts.' in name:raise ValueError(f'Unknown expert payload {name}')
            return
        shapes=self.shapes[int(scale)]
        if key not in self.targets or key in shapes or value.dtype!=torch.uint8 or value.ndim!=2:
            raise ValueError(f'Invalid or duplicate packed payload {name}: {value.dtype}, {value.shape}')
        shapes[key]=tuple(value.shape)
        self.bytes+=value.nbytes
        if key in self.sample_names:
            self.samples[name]=hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()

    def finish(self,side,step):
        weights,scales=self.shapes
        if not weights or weights.keys()!=scales.keys():raise ValueError(f'Incomplete weight/scale pairs on {side}')
        for key,(rows,width) in weights.items():
            if width%16 or scales[key]!=(rows,width//16):raise ValueError(f'Incorrect MX scale shape: {key}')
        metadata=json.dumps(sorted((key,weights[key],scales[key]) for key in weights),separators=(',',':')).encode()
        rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else None
        event=dict(side=side,step=step,rank=rank,pairs=len(weights),expert_bytes=self.bytes,
                   layout_sha256=hashlib.sha256(metadata).hexdigest(),samples=self.samples)
        logger.warning('DSV4_QUANT_WIRE %s',json.dumps(event,sort_keys=True))


async def encode(weights,targets,step):
    audit=PayloadAudit(targets)
    async for name,value in ensure_async_iterator(weights):
        key=canonical_expert_name(name)
        if key is None:
            if '.experts.' in name:raise ValueError(f'Unexpected non-HF expert payload: {name}')
            yield name,value
            continue
        if key not in targets:raise ValueError(f'Expert absent from rollout quantization descriptor: {name}')
        if value.device.type=='cpu':
            value=value.to(torch.device('npu',torch.npu.current_device()))
        packed,scale=quantize_expert_matrix(value)
        for wire_name,wire_value in ((name,packed),(name+'_scale',scale)):
            audit.observe(wire_name,wire_value)
            yield wire_name,wire_value
    audit.finish('sender_before_bucket',step)
