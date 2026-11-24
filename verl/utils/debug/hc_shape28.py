"""Measure live same-input mHC under two row counts without changing the baseline return value."""
from pathlib import Path
import json
import os
import re
import torch


def install(model, context):
    if getattr(model, '_align_hc_shape_installed',False):
        return
    model._align_hc_shape_installed = True
    fixed = int(os.environ.get('DSV4_ALIGN_HC_TILE','0'))
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    folder = Path(os.environ['DSV4_QAT_RUN'])/'hc_shape'/str(os.getpid())
    seen=set()

    def diff(a,b):
        d=(a.float()-b.float()).abs()
        return {'n':d.numel(),'neq':int(d.count_nonzero()),'max_abs':float(d.max()) if d.numel() else None}

    def outputs(result):
        return list(result) if isinstance(result,(list,tuple)) else [result]

    def run_tiled(call, args, ninputs):
        n=len(args[0])
        if not fixed or n==0:return call(*args)
        parts=[]
        for lo in range(0,n,fixed):
            stop=min(n,lo+fixed);size=stop-lo
            batch=[]
            for x in args[:ninputs]:
                assert len(x)==n
                t=x[lo:stop]
                if size<fixed:t=torch.cat((t,t.new_zeros((fixed-size,*t.shape[1:]))),dim=0)
                batch.append(t.contiguous())
            got=call(*batch,*args[ninputs:])
            parts.append([x[:size] for x in outputs(got)])
        out=[torch.cat(items,dim=0) for items in zip(*parts)]
        return type(got)(out) if isinstance(got,(list,tuple)) else out[0]

    def wrap(module, depth, op, fn):
        ninputs=1 if op=='pre' else 4
        def measured(*args):
            n=len(args[0])
            if op=='pre':
                module._align_hc_branch='ffn' if args[1] is module.hc_ffn_fn else 'attn'
            tag=module._align_hc_branch
            result=run_tiled(fn,args,ninputs)
            if depth==4 and n in (62,124) and rank==0:
                ids=context.get('input_ids');pos=context.get('positions')
                if ids is not None and pos is not None and ids.numel()==n and int(pos.max())==61:
                    import hashlib
                    t=ids.flatten().contiguous().cpu();key=hashlib.sha256(t.numpy().tobytes()).hexdigest()[:24]
                    out=folder.parent.parent/'depth4_hc'/str(os.getpid())/key
                    file=out/(tag+'_'+op+'.pt')
                    if not file.exists():
                        out.mkdir(exist_ok=True,parents=True)
                        torch.save(dict(ids=t,pos=pos.flatten().cpu(),inputs=[x.detach().cpu() for x in args[:ninputs]],outputs=[x.detach().cpu() for x in outputs(result)]),file)

            if depth==8 and rank==0 and n in (57,65):
                tok=context.get('input_ids');pos=context.get('positions')
                if tok is not None and pos is not None and tok.numel()==n and int(pos.max())==n-1:
                    path=folder.parent.parent/'layer8_hc'/str(os.getpid())/f'{tag}_{op}_{n}.pt'
                    if not path.exists():
                        path.parent.mkdir(parents=True,exist_ok=True)
                        torch.save(dict(ids=tok.cpu(),pos=pos.cpu(),inputs=[x.detach().cpu() for x in args[:ninputs]],
                                        outputs=[x.detach().cpu() for x in outputs(result)]),path)
            if depth>2 or rank not in (0,3) or n not in (57,65,114,130):
                return result
            pos=context.get('positions')
            if pos is None or len(pos.flatten())!=n:return result
            key=(depth,op,tag,n)
            if key in seen:return result
            seen.add(key)
            folder.mkdir(parents=True,exist_ok=True)
            name=f'L{depth}_{tag}_{op}_{n}'
            event=dict(rank=rank,depth=depth,op=op,branch=tag,n=n,fixed_tile=fixed,
                       method_module=fn.__module__,method_file=getattr(getattr(getattr(fn,'__func__',fn),'__code__',None),'co_filename',None))
            if n>57:
                short=tuple(v[:57].contiguous() for v in args[:ninputs])+args[ninputs:]
                a=run_tiled(fn,short,ninputs)
                event['identical_operands_57_vs_full']=[diff(x,y[:57]) for x,y in zip(outputs(a),outputs(result))]
            if depth==0:
                snap=dict(token_ids=context['input_ids'],position_ids=pos,
                          inputs=[v.detach().cpu() for v in args[:ninputs]],outputs=[v.detach().cpu() for v in outputs(result)])
                if op=='pre' and n==57:snap['params']=[v.detach().cpu() for v in args[ninputs:]]
                torch.save(snap,folder/(name+'.pt'))
            (folder/(name+'.json')).write_text(json.dumps(event,indent=2))
            return result
        return measured
    for name,layer in model.named_modules():
        match=re.search(r'(?:^|\.)layers\.(\d+)$',name)
        if match and hasattr(layer,'hc_pre') and hasattr(layer,'hc_post'):
            depth=int(match.group(1))
            if fixed or (depth in (0,1,2,4) and rank in (0,3)):
                layer.hc_pre=wrap(layer,depth,'pre',layer.hc_pre)
                layer.hc_post=wrap(layer,depth,'post',layer.hc_post)
