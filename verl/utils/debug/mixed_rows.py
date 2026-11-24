"""Record a per-token byte digest before dropping traces for alternate causal-prefix shapes."""
import hashlib
import json
from pathlib import Path
import torch

def trace(directory,seen,name,phase,value,ids,pos):
    from verl.utils.debug.depth4_inputs import save
    save(directory,seen,name,phase,value,ids,pos)
    if phase!='output' or pos.numel() not in (62,124) or ids.numel()!=pos.numel():
        return
    pos=pos.flatten()
    if int(pos.max())!=61 or int((pos==0).sum())!=pos.numel()//62:
        return
    ids=ids.reshape(-1).contiguous()
    identity=hashlib.sha256(ids.numpy().tobytes()).hexdigest()[:24]
    key=('mixed',identity,name)
    if key in seen:return
    seen.add(key)
    if isinstance(value,(tuple,list)):value=next((v for v in value if isinstance(v,torch.Tensor) and v.is_floating_point()),None)
    if not isinstance(value,torch.Tensor) or not value.is_floating_point() or value.shape[0]!=len(ids):return
    t=value.detach().cpu().contiguous().reshape(len(ids),-1)
    raw=t.view(torch.uint8).numpy()
    if name.startswith('model.layers.4.') or name in ('model.layers.4','model.layers.3'):
        folder=directory.parents[1]/'mixed_raw'/directory.name/identity
        folder.mkdir(exist_ok=True,parents=True)
        torch.save(dict(ids=ids,pos=pos,t=t),folder/(name+'.pt'))
    digest=[hashlib.sha256(memoryview(row)).hexdigest() for row in raw]
    out=directory.parents[1]/'mixed_rows'/directory.name/identity
    out.mkdir(parents=True,exist_ok=True)
    (out/(name+'.json')).write_text(json.dumps(dict(ids=ids.tolist(),pos=pos.tolist(),digest=digest,shape=list(t.shape),dtype=str(t.dtype))))
