"""Save matched semantic submodule outputs for the first original-policy prefix on each stack."""
from pathlib import Path
import os
import torch

NAMES={'input_layernorm':'attention_norm','self_attn':'attention','self_attention':'attention',
       'pre_mlp_layernorm':'ffn_norm','post_attention_layernorm':'ffn_norm',
       'mlp.shared_experts.linear_fc1':'shared_gate_up','mlp.shared_experts.gate_up_proj':'shared_gate_up',
       'mlp.shared_experts.linear_fc2':'shared_down','mlp.shared_experts.down_proj':'shared_down','mlp':'mlp'}
def install(module,depth,tail,side,directory,context):
    kind=NAMES.get(tail)
    if not kind:return
    if torch.distributed.is_initialized() and torch.distributed.get_rank() not in ((0,) if side=='actor' else (0,3)):return
    saved=False
    def after(mod,args,out):
        nonlocal saved
        if saved:return
        if side=='actor' and os.getenv('DSV4_ALIGN_VARIANT_LABEL')!='actor_paged_combined_shared_fc1_post_compress_order':return
        pos=context.get('position_ids');ids=context.get('token_ids')
        if pos is None or ids is None:return
        pos=pos.flatten();ids=ids.flatten()
        if len(pos)!=(65 if side=='actor' else 57) or len(pos)!=len(ids) or not torch.equal(pos.cpu(),torch.arange(len(pos))):return
        if side=='rollout' and not (Path(os.environ['DSV4_QAT_RUN'])/'audit'/f'refit_{os.getpid()}.jsonl').exists():return
        t=out[0] if isinstance(out,(list,tuple)) else out
        if not isinstance(t,torch.Tensor) or t.shape[0]!=len(pos):return
        folder=Path(os.environ['DSV4_QAT_RUN'])/'branch_edges'/side/str(os.getpid())
        folder.mkdir(exist_ok=True,parents=True)
        torch.save(dict(ids=ids[:57].cpu(),pos=pos[:57].cpu(),x=t[:57].detach().reshape(57,-1).cpu()),folder/f'{depth:02d}_{kind}.pt')
        saved=True
    module.register_forward_hook(after)
