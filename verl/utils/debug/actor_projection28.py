"""Original-policy forward-only ablation of the two measured BF16 projection boundaries."""
import os
from pathlib import Path
import torch
from verl.utils.debug.train_alignment import with_local_backward


def configure(modules, mode):
    enabled = mode in ('actor_bi_l0','actor_bi_all','actor_bi_all_repeat','actor_projections_shared_route','actor_projections_shared_route_repeat')
    import re
    if enabled:
        import batch_invariant_ops  # register the same AscendC op called by serving
    found = 0
    for root in modules:
        for name,layer in root.named_modules():
            match = re.search(r'(?:^|\.)layers\.(\d+)\.self_attention\.(linear_q_down_proj|linear_kv_proj)$',name)
            if not match:
                continue
            found += 1
            depth=int(match.group(1))
            layer._projection_probe_mode = mode if enabled and (depth==0 or mode!='actor_bi_l0') else None
            if hasattr(layer,'_projection_probe_handle'):
                continue
            def project(module, args, output, depth=depth, kind=match.group(2)):
                active=module._projection_probe_mode
                if active is None:return
                x=args[0];weight=module.weight
                assert isinstance(output,tuple) and len(output)==2 and output[1] is None
                bias=getattr(module,'bias',None)
                assert bias is None or (isinstance(bias,torch.Tensor) and bias.numel()==0 and not getattr(module,'use_bias',False)), (kind,type(module))
                assert x.dtype==weight.dtype==torch.bfloat16,(kind,type(module),x.dtype,weight.dtype)
                real=output[0]
                x2=x.reshape(-1,x.shape[-1])
                y=torch.ops.batch_invariant_ops.npu_mm_batch_invariant(x2,weight.t()).reshape_as(real)
                label=(active,depth,kind)
                if label not in module._projection_probe_seen:
                    module._projection_probe_seen.add(label)
                    where=Path(os.environ['DSV4_QAT_RUN'])/'projection_probe'/str(os.getpid())/active
                    where.mkdir(parents=True,exist_ok=True)
                    with (where/'calls.tsv').open('a') as log:
                        log.write(f'{depth}\t{kind}\t{tuple(x.shape)}\t{int((real!=y).sum())}\n')
                    if depth==0 and os.getenv('DSV4_ALIGN_LONG_POLICY') != '1':
                        torch.save(dict(x=x.detach().cpu(),before=real.detach().cpu(),after=y.detach().cpu()),where/(kind+'.pt'))
                return (with_local_backward(y,real) if torch.is_grad_enabled() else y,None)
            layer._projection_probe_seen=set()
            layer._projection_probe_handle=layer.register_forward_hook(project)
    assert found==86,found
