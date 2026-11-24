"""Score original-policy tokens using the native paged attention kernel with independently built Actor pages."""
import json
import os
from pathlib import Path

import torch
from verl.utils.debug.train_alignment import with_local_backward


def page(values):
    n, heads, size = values.shape
    pages = max(1, (n+31)//32)
    padding = values.new_zeros((pages*32-n,heads,size))
    return torch.cat((values,padding)).view(pages,32,heads,size), torch.arange(pages,device=values.device,dtype=torch.int32)[None,:]


def paged(q, kv, sink, indices, scale, ratio):
    from vllm_ascend.attention.sparse_flash_mla import sparse_flash_mla, sparse_flash_mla_metadata
    n,b,h,d=q.shape
    assert b==1 and n in (57,65) and (h,d)==(64,512) and q.dtype==kv.dtype==torch.bfloat16
    extra=len(kv)-n
    assert extra==(0 if ratio==1 else n//ratio)
    x=q[:,0].contiguous()
    k,table=page(kv[:n].reshape(n,1,d))
    cu=torch.tensor([0,n],dtype=torch.int32,device=q.device)
    lengths=cu[1:].contiguous()
    # First verify that the kernel's implicit window covers exactly the Actor's key IDs.
    ids=indices[0]
    serial=torch.arange(n,device=q.device)
    wanted=(serial[None,:]<=serial[:,None])&(serial[None,:]>=serial[:,None]-127)
    regular=(ids>=0)&(ids<n)
    counts=torch.zeros_like(wanted,dtype=torch.int32)
    counts.scatter_add_(1,ids.clamp(0,n-1).long(),regular.int())
    assert torch.equal(counts,wanted.int())
    layout=dict(cmp_ratio=ratio,ori_mask_mode=4,ori_win_left=127,ori_win_right=0,layout_q='TND',layout_kv='PA_BBND')
    kwargs=dict(ori_kv=k,ori_block_table=table,cu_seqlens_q=cu,seqused_kv=lengths,sinks=sink.float().contiguous(),softmax_scale=scale,**layout)
    if ratio>1:
        compressed,comp_table=page(kv[n:].reshape(extra,1,d))
        kwargs.update(cmp_kv=compressed,cmp_block_table=comp_table,cmp_mask_mode=3)
    if ratio==4:
        # Preserve the Actor's selected order, moving invalid and window entries to the end.
        valid=ids>=n
        assert not bool((ids>=n+extra).any())
        slots=valid.int().cumsum(-1)-1
        out=indices.new_full((n,1,512),-1,dtype=torch.int32)
        row=torch.arange(n,device=q.device)[:,None].expand_as(ids)
        out[row[valid],0,slots[valid].long()]=(ids[valid]-n).int()
        kwargs['cmp_sparse_indices']=out
    elif ratio==128:
        assert extra==0
    meta=sparse_flash_mla_metadata(num_heads_q=h,num_heads_kv=1,head_dim=d,cu_seqlens_q=cu,seqused_ori_kv=lengths,
        batch_size=1,max_seqlen_q=n,max_seqlen_ori_kv=n,max_seqlen_cmp_kv=extra,ori_topk=0,cmp_topk=512 if ratio==4 else 0,
        has_ori_kv=True,has_cmp_kv=ratio>1,cmp_mask_mode=3 if ratio>1 else 0,**layout)
    return sparse_flash_mla(x,metadata=meta,**kwargs)[0].reshape(n,b,h*d)


def configure(roots,label):
    from mindspeed_bridge.models.deepseek_v4_flash.modelling_deepseek_v4_flash import csa_smla_fused as csa
    import re
    if not hasattr(csa,'_original_paged28'):
        csa._original_paged28=csa.fused_compressed_sparse_attn
    if label not in ('actor_paged_only','actor_paged_combined','actor_paged_combined_repeat'):
        csa.fused_compressed_sparse_attn=csa._original_paged28
        return
    layers={}
    for root in roots:
        for name,module in root.named_modules():
            match=re.search(r'layers\.(\d+)\.self_attention\.core_attention$',name)
            if match:layers[int(match[1])]=module
    assert len(layers)==43
    context={}
    dest=Path(os.environ['DSV4_QAT_RUN'])/'actor_paged'/str(os.getpid())/label
    dest.mkdir(parents=True,exist_ok=True)
    for depth,module in layers.items():
        module._paged28_context=context
        if getattr(module,'_paged28_hook',None) is None:
            def enter(m,args,depth=depth):
                m._paged28_context.update(depth=depth,ratio=max(1,int(m.compress_ratio)))
            module._paged28_hook=module.register_forward_pre_hook(enter)
    def forward(q,kv,sink,indices,scale,ori_len=None):
        assert ori_len in (None,len(q))
        depth,ratio=context['depth'],context['ratio']
        if not seen and label=='actor_paged_only' and torch.distributed.get_rank()==0:
            from verl.utils.debug.paged_check28 import verify
            try:
                check=verify(dest.parents[2],paged,q.device)
                (dest/'matched_operands.json').write_text(json.dumps(check,indent=2))
            except Exception as exc:
                (dest/'replay_error.json').write_text(json.dumps(dict(error=repr(exc))))
        actual_label=os.environ.get('DSV4_ALIGN_VARIANT_LABEL','')
        if os.getenv('DSV4_ALIGN_LONG_POLICY') != '1' and ratio==4 and actual_label.startswith('actor_paged_combined_shared_fc1_post_compress_order'):
            valid=indices>=len(q)
            sentinel=torch.iinfo(indices.dtype).max
            ordered=indices.masked_fill(~valid,sentinel).sort(dim=-1).values
            slots=(valid.long().cumsum(dim=-1)-1).clamp_min(0)
            indices=torch.where(valid,ordered.gather(-1,slots),indices)
        from verl.utils.debug.long_paged import paged as long_paged
        with torch.no_grad():
            y=long_paged(q,kv,sink,indices,scale,ratio)
        case=os.getenv('DSV4_LONG_REPLAY_CASE')
        rank=torch.distributed.get_rank()
        if case in ('long_first8','long_full') and rank in (0,1) and depth in (0,2,4):
            f=Path(os.environ['DSV4_QAT_RUN'])/'core_operands'/case/str(rank)/f'L{depth:02d}.pt'
            f.parent.mkdir(parents=True,exist_ok=True)
            if not f.exists():
                torch.save(dict(q=q.detach().cpu(),kv=kv.detach().cpu(),sink=sink.detach().cpu(),
                    indices=indices.detach().cpu(),scale=float(scale),ratio=ratio,output=y.detach().cpu()),f)
        if torch.is_grad_enabled():
            reference=csa._original_paged28(q,kv,sink,indices,scale,ori_len)
            y=with_local_backward(y,reference)
        if os.getenv('DSV4_ALIGN_LONG_POLICY') != '1' and actual_label in ('actor_paged_combined_shared_fc1_post','actor_paged_combined_shared_fc1_post_compress','actor_paged_combined_shared_fc1_post_compress_order') and torch.distributed.get_rank()==0 and depth in (0,1,2,3,4,5,6,7,8,9,15,31,42):
            target=Path(os.environ['DSV4_QAT_RUN'])/('actor_kernel_operands_compress_order' if actual_label=='actor_paged_combined_shared_fc1_post_compress_order' else 'actor_kernel_operands_compress' if actual_label=='actor_paged_combined_shared_fc1_post_compress' else 'actor_kernel_operands')/str(os.getpid())
            target.mkdir(parents=True,exist_ok=True)
            file=target/f'layer{depth:02d}.pt'
            if not file.exists():
                torch.save(dict(q=q.detach().cpu(),kv=kv.detach().cpu(),sink=sink.detach().cpu(),indices=indices.detach().cpu(),
                                scale=float(scale),ratio=int(ratio),output=y.detach().cpu()),file)

        if depth not in seen:
            seen.add(depth)
            with (dest/'layers.jsonl').open('a') as f:
                f.write(json.dumps(dict(layer=depth,ratio=ratio,shape=list(q.shape),kv_rows=len(kv),width=indices.shape[-1]))+'\n')
        return y
    seen=set()
    csa.fused_compressed_sparse_attn=forward
