"""Score each Actor sequence on its independently built pages with its original causal mask."""
import torch

def paged(q,kv,sink,indices,scale,ratio):
    from verl.utils.debug.actor_paged28 import page
    from mindspeed_bridge.models.deepseek_v4_flash.modelling_deepseek_v4_flash.csa import (
        get_compress_topk_idxs,get_window_topk_idxs)
    from vllm_ascend.attention.sparse_flash_mla import sparse_flash_mla,sparse_flash_mla_metadata
    n,b,h,d=q.shape
    assert (h,d)==(64,512) and q.dtype==kv.dtype==torch.bfloat16
    assert len(kv)==n+(n//ratio if ratio>1 else 0) and ratio in (1,4,128)
    assert torch.equal(indices[:,:,:128],get_window_topk_idxs(128,b,n,q.device))
    extra=len(kv)-n
    comp=indices[:,:,128:]
    if ratio==128:
        assert torch.equal(comp,get_compress_topk_idxs(ratio,b,n,n,q.device,n_compressed=extra))
    if ratio==4:
        assert comp.shape[-1]<=512
        valid_counts=(torch.arange(n,device=q.device)+1)//ratio
        assert bool(((comp==-1)|((comp>=n)&(comp<n+valid_counts[None,:,None]))).all())
    if ratio==1:assert comp.shape[-1]==0
    cu=torch.tensor([0,n],dtype=torch.int32,device=q.device)
    lengths=cu[1:].contiguous()
    layout=dict(cmp_ratio=ratio,ori_mask_mode=4,ori_win_left=127,ori_win_right=0,layout_q='TND',layout_kv='PA_BBND')
    meta=sparse_flash_mla_metadata(num_heads_q=h,num_heads_kv=1,head_dim=d,cu_seqlens_q=cu,
          seqused_ori_kv=lengths,batch_size=1,max_seqlen_q=n,max_seqlen_ori_kv=n,max_seqlen_cmp_kv=extra,
          ori_topk=0,cmp_topk=512 if ratio==4 else 0,has_ori_kv=True,has_cmp_kv=ratio>1,
          cmp_mask_mode=3 if ratio>1 else 0,**layout)
    result=[]
    for j in range(b):
        k,table=page(kv[:n,j].reshape(n,1,d))
        kw=dict(ori_kv=k,ori_block_table=table,cu_seqlens_q=cu,seqused_kv=lengths,
                sinks=sink.float().contiguous(),softmax_scale=scale,**layout)
        if ratio>1:
            cp,ct=page(kv[n:,j].reshape(extra,1,d))
            kw.update(cmp_kv=cp,cmp_block_table=ct,cmp_mask_mode=3)
        if ratio==4:
            ci=comp[j]
            invalid=torch.iinfo(ci.dtype).max
            ci=torch.where(ci>=n,ci-n,invalid).sort(-1).values
            ci=torch.where(ci==invalid,-1,ci).int()
            kw['cmp_sparse_indices']=torch.nn.functional.pad(ci[:,None],(0,512-ci.shape[-1]),value=-1).contiguous()
        out=sparse_flash_mla(q[:,j].contiguous(),metadata=meta,**kw)[0]
        result.append(out.reshape(n,h*d))
    return torch.stack(result,1)
