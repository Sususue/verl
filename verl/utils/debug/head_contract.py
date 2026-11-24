"""Compare native output-contraction order using the Actor's own residual and learned weights."""
from pathlib import Path
import os
import torch
from verl.utils.debug.train_alignment import with_local_backward
import torch_npu

def configure(mode):
    from mindspeed_bridge.models.deepseek_v4_flash.modelling_deepseek_v4_flash import hyper_connection_layer as hc
    if not hasattr(hc,'_independent_head_contract'):
        hc._independent_head_contract=hc.learned_output_contract
    original=hc._independent_head_contract
    if mode not in ('head_contract','head_tile','head_sampler','head_all','head_all_repeat',
                    'actor_paged_combined_shared_fc1_post_compress_order'):
        hc.learned_output_contract=original
        return
    def contract(hidden,weight,base,scale,n,eps):
        if mode in ('head_contract','head_all','head_all_repeat'):
            shape=hidden.shape
            x=hidden.reshape(-1,shape[-1]).float()
            norm,_=torch_npu.npu_rms_norm(x,torch.ones_like(x[0]),eps)
            import batch_invariant_ops
            mixes=torch.ops.batch_invariant_ops.npu_mm_batch_invariant(norm.contiguous(),weight.float().T)
            if torch.is_grad_enabled():
                mixes=with_local_backward(mixes,torch.nn.functional.linear(norm,weight.float()))
            gates=torch.sigmoid(mixes*scale.float()+base.float())+eps
            y=(gates.unsqueeze(-1)*x.reshape(len(x),n,-1)).sum(1)
            out=y.to(hidden.dtype).reshape(*shape[:-1],shape[-1]//n)
        else:
            out=original(hidden,weight,base,scale,n,eps)
        if os.getenv('DSV4_ALIGN_LONG_POLICY') == '1': return out
        folder=Path(os.environ['DSV4_QAT_RUN'])/'head_contract'/str(os.getpid())
        path=folder/(mode+'.pt')
        if not path.is_file():
            folder.mkdir(parents=True,exist_ok=True)
            torch.save(dict(residual=hidden.detach().cpu(),contracted=out.detach().cpu(),
                            head_weight=weight.detach().cpu(),base=base.detach().cpu(),
                            scale=scale.detach().cpu(),eps=eps),path)
        return out
    hc.learned_output_contract=contract
