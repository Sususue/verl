"""On frozen policy scoring, replay the real serving BF16 shared gate/up projection."""
import os
from pathlib import Path
import torch
from verl.utils.debug.train_alignment import with_local_backward


def configure(roots, label):
    active=label.startswith('actor_paged_combined_shared_fc1')
    count=0
    for root in roots:
        for name, mod in root.named_modules():
            if not name.endswith('.mlp.shared_experts.linear_fc1'):
                continue
            count+=1
            mod._shared_fc1_label=label if active else None
            if hasattr(mod,'_shared_fc1_probe'):
                continue
            def project(layer,args,out,layer_name=name):
                mode=layer._shared_fc1_label
                if mode is None:
                    return
                import batch_invariant_ops
                x=args[0];w=layer.weight
                assert x.dtype==w.dtype==torch.bfloat16
                assert isinstance(out,tuple) and len(out)==2 and out[1] is None
                assert not getattr(layer,'use_bias',False)
                result=torch.ops.batch_invariant_ops.npu_mm_batch_invariant(x.reshape(-1,x.shape[-1]),w.T).reshape_as(out[0])
                if mode not in layer._shared_fc1_seen:
                    layer._shared_fc1_seen.add(mode)
                    path=Path(os.environ['DSV4_QAT_RUN'])/'fc1_calls'/str(os.getpid())/mode
                    path.mkdir(parents=True,exist_ok=True)
                    with (path/'layers.txt').open('a') as f:
                        print(layer_name,int((result!=out[0]).sum()),file=f)
                return (with_local_backward(result,out[0]) if torch.is_grad_enabled() else result),None
            mod._shared_fc1_seen=set()
            mod._shared_fc1_probe=mod.register_forward_hook(project,prepend=True)
    assert count==43
