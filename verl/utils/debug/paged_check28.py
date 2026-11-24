"""Compare reconstructed independent pages with real captured native attention on exact operands."""
import json
from pathlib import Path
import torch


def verify(run, paged, device):
    rows=[]
    for depth in (1,2,3):
        for n in (57,65):
            paths=sorted((run/'attention_core/rollout').glob(f'*/layer{depth:02d}_prefill{n}_*.pt'))
            if not paths:
                continue
            p=torch.load(paths[0],weights_only=True,map_location='cpu')
            k=p['kwargs'];ratio=k['cmp_ratio'];seq=k['cu_seqlens_q'].tolist()
            assert seq==[0,n] and k['seqused_kv'].tolist()==[n]
            def unpage(key,size):
                cache=k[key+'_kv'];table=k[key+'_block_table'][0].long()
                return cache[table[:(size+31)//32]].flatten(0,1)[:size].reshape(size,1,cache.shape[-1])
            vals=unpage('ori',n)
            ts=torch.arange(n)
            idx=ts.expand(n,-1).masked_fill(ts[None,:]>ts[:,None],-1).int()
            if ratio>1 and n//ratio>0:
                vals=torch.cat((vals,unpage('cmp',n//ratio)))
                ci=k['cmp_sparse_indices'].reshape(n,-1)
                idx=torch.cat((idx,torch.where(ci>=0,ci+n,ci)),dim=1)
            q=p['q'];sink=k['sinks']
            y=paged(q.to(device).unsqueeze(1),vals.to(device),sink.to(device),idx.to(device).unsqueeze(0),k['softmax_scale'],ratio)
            actual=y.reshape_as(p['output']).detach().cpu();delta=abs(actual.float()-p['output'].float())
            rows.append(dict(capture=str(paths[0]),layer=depth,ratio=ratio,n_rows=n,elements=delta.numel(),neq=int(delta.count_nonzero()),max_abs=float(delta.max())))
    return rows


if __name__=='__main__':
    import sys,subprocess,os
    assert subprocess.check_output(['npu-smi','info'],text=True).count('No running processes found')==8
    torch.set_grad_enabled(False);torch.npu.set_device(0)
    from actor_paged28 import paged
    r=Path('/run/dsv4_alignment24/work/runs')/sys.argv[1]
    result=verify(r,paged,'npu:0')
    (r/'paged_operand_replay.json').write_text(json.dumps(result,indent=2));print(result)
