"""Save bounded real operands for layer four only on the controlled causal-prefix layouts."""
import hashlib
from pathlib import Path
import torch
def pack(v):
    if isinstance(v,torch.Tensor):
        if v.numel()*v.element_size()>32*1024**2:return dict(shape=list(v.shape),dtype=str(v.dtype))
        return v.detach().cpu()
    if isinstance(v,dict):return {k:pack(z) for k,z in v.items() if isinstance(z,(torch.Tensor,dict,tuple,list))}
    if isinstance(v,(tuple,list)):return [pack(z) for z in v if isinstance(z,(torch.Tensor,dict,tuple,list))]
    return None
def save(directory,seen,name,phase,value,ids,pos):
    if not name.startswith('model.layers.4.') or pos.numel() not in (62,124):return
    ids=ids.flatten().contiguous().cpu();pos=pos.flatten().cpu()
    if len(ids)!=len(pos) or int(pos.max())!=61:return
    tag=hashlib.sha256(ids.numpy().tobytes()).hexdigest()[:24]
    key=('depth4_raw',tag,name,phase)
    if key in seen:return
    seen.add(key)
    folder=directory.parents[1]/'depth4_inputs'/directory.name/tag;folder.mkdir(parents=True,exist_ok=True)
    torch.save(dict(ids=ids,pos=pos,value=pack(value)),folder/(name+'_'+phase+'.pt'))
