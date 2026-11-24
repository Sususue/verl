"""Match the serving shared SwiGLU rounding on frozen real tokens for attribution."""
import os
import re
from pathlib import Path
import torch


def configure(roots,label):
    active=label in ('actor_shared','actor_shared_route','actor_projections_shared_route','actor_projections_shared_route_repeat')
    found=0
    for root in roots:
        for name,mlp in root.named_modules():
            match=re.search(r'layers\.(\d+)\.mlp\.shared_experts$',name)
            if not match:continue
            found+=1
            mlp._align_shared_label=label if active else None
            if hasattr(mlp,'_align_shared_hooks'):continue
            depth=int(match.group(1))
            def gate(_,args,output,mlp=mlp):
                if mlp._align_shared_label:
                    assert not hasattr(mlp,'_align_gate_operand')
                    mlp._align_gate_operand=output[0] if isinstance(output,(tuple,list)) else output
            def activate(_,args,mlp=mlp,depth=depth):
                mode=mlp._align_shared_label
                if not mode:return
                x=mlp._align_gate_operand
                del mlp._align_gate_operand
                g,u=x.chunk(2,-1)
                assert x.dtype==torch.bfloat16 and g.shape==args[0].shape
                limit=getattr(mlp.config,'activation_func_clamp_value',None)
                if limit is not None:g,u=g.clamp(max=limit),u.clamp(-limit,limit)
                out=(g*torch.sigmoid(g))*u
                if mode not in mlp._align_seen:
                    mlp._align_seen.add(mode)
                    d=Path(os.environ['DSV4_QAT_RUN'])/'shared_hits'/str(os.getpid())/mode
                    d.mkdir(parents=True,exist_ok=True)
                    with (d/'layers.tsv').open('a') as f:f.write(f'{depth}\t{limit}\t{int((args[0]!=out).sum())}\n')
                    if depth==0 and os.getenv('DSV4_ALIGN_LONG_POLICY') != '1':torch.save(dict(gate_up=x.detach().cpu(),before=args[0].detach().cpu(),after=out.detach().cpu()),d/'shared_activation.pt')
                return (out,*args[1:])
            mlp._align_seen=set()
            mlp._align_shared_hooks=(mlp.linear_fc1.register_forward_hook(gate),mlp.linear_fc2.register_forward_pre_hook(activate))
    assert found==43,found
