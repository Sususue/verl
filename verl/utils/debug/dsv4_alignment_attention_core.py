"""Bounded token-aware attention traces from the normal forward thread."""
from pathlib import Path
import os,json,re,traceback
import torch

_seen=set()
_rollout_context={}

def set_rollout_context(context):
    global _rollout_context
    _rollout_context=context

def _cpu(v):
    if isinstance(v,torch.Tensor):return v.detach().cpu()
    if isinstance(v,dict):return {k:_cpu(x) for k,x in v.items()}
    if isinstance(v,tuple):return tuple(_cpu(x) for x in v)
    if isinstance(v,list):return [_cpu(x) for x in v]
    return v

def compact_pages(cache,table,lengths,first_positions=None):
    table=table.detach().cpu()
    lengths=torch.as_tensor(_cpu(lengths),dtype=torch.int64).reshape(-1)
    assert cache.ndim==4 and table.ndim==2 and len(lengths)==table.shape[0]
    needed=(lengths+cache.shape[1]-1)//cache.shape[1]
    assert bool((needed<=table.shape[1]).all())
    first=torch.zeros_like(lengths) if first_positions is None else torch.as_tensor(first_positions).reshape(-1)//cache.shape[1]
    columns=torch.arange(table.shape[1])[None,:]
    valid=(columns>=first[:,None])&(columns<needed[:,None])
    used=table[valid].long().unique(sorted=True)
    assert bool(((used>=0)&(used<cache.shape[0])).all())
    if not used.numel():used=torch.zeros(1,dtype=torch.int64)
    remap=torch.searchsorted(used,table.long().contiguous()).to(table.dtype)
    remap[~valid]=0
    selected=cache.detach().index_select(0,used.to(cache.device)).cpu()
    meta={'original_shape':list(cache.shape),'original_stride':list(cache.stride()),
          'physical_pages':used,'valid_table_entries':valid,'lengths':lengths}
    return selected,remap,meta

def compact_original_cache(kwargs):
    lengths=torch.as_tensor(_cpu(kwargs['seqused_kv']),dtype=torch.int64).reshape(-1)
    cu_value=kwargs.get('cu_seqlens_q')
    cu=torch.as_tensor(_cpu(cu_value),dtype=torch.int64) if cu_value is not None else None
    qlens=cu[1:]-cu[:-1] if cu is not None else lengths
    window=kwargs.get('ori_win_left',-1)
    first=(lengths-qlens-window).clamp_min(0) if window>=0 else torch.zeros_like(lengths)
    ori,table,meta=compact_pages(kwargs['ori_kv'],kwargs['ori_block_table'],lengths,first)
    copied={**kwargs,'ori_kv':ori,'ori_block_table':table}
    ratio=int(kwargs.get('cmp_ratio',1))
    if kwargs.get('cmp_kv') is not None:
        cmp_lengths=kwargs.get('seqused_cmp_kv')
        if cmp_lengths is None:cmp_lengths=lengths//ratio
        cmp,cmp_table,cmp_meta=compact_pages(kwargs['cmp_kv'],kwargs['cmp_block_table'],cmp_lengths)
        copied.update(cmp_kv=cmp,cmp_block_table=cmp_table)
        meta['compressed']=cmp_meta
    return _cpu(copied),meta

def capture_attention_core(layer_name,q,kwargs,result,cos,sin,has_prefill):
    if os.environ.get('DSV4_ALIGN_ATTN_CORE')!='1':return
    match=re.search(r'layers\.(\d+)\.',layer_name)
    layers={int(x) for x in os.environ.get('DSV4_ALIGN_ATTN_LAYERS','0').split(',')}
    if not match or int(match[1]) not in layers:return
    layer=int(match[1]);phase='prefill' if has_prefill else 'decode'
    if not has_prefill or (torch.distributed.is_initialized() and torch.distributed.get_rank() not in (0,3)):
        return
    prefix_shape_token_count = _rollout_context.get("position_ids")
    if prefix_shape_token_count is None or prefix_shape_token_count.numel() not in (57,62,65,114,124,130):
        return
    phase += str(prefix_shape_token_count.numel())
    if prefix_shape_token_count.numel() in (62,124):
        import hashlib
        token_ids=_rollout_context['token_ids'].flatten().contiguous().cpu()
        phase += '_'+hashlib.sha256(token_ids.numpy().tobytes()).hexdigest()[:24]

    run=Path(os.environ['DSV4_QAT_RUN'])
    if not (run/'audit'/f'refit_{os.getpid()}.jsonl').exists():return
    lengths=torch.as_tensor(_cpu(kwargs['seqused_kv']),dtype=torch.int64).reshape(-1)
    cu=torch.as_tensor(_cpu(kwargs['cu_seqlens_q']),dtype=torch.int64)
    positions=torch.cat([torch.arange(int(lengths[i]-(cu[i+1]-cu[i])),int(lengths[i])) for i in range(len(lengths))])
    requested={int(x) for x in os.environ.get('DSV4_ALIGN_ATTN_POSITIONS','0,57').split(',')}
    selected=sorted(set(positions.tolist())&requested)
    selected=[p for p in selected if (layer,phase,p) not in _seen]
    if not selected:return
    directory=run/'attention_core/rollout'/str(os.getpid());directory.mkdir(parents=True,exist_ok=True)
    stem=f'layer{layer:02d}_{phase}_p'+('-'.join(map(str,selected)))
    try:
        saved_kwargs,cache_meta=compact_original_cache(kwargs)
        context={k:_rollout_context.get(k) for k in ['token_ids','position_ids']}
        data={'layer_name':layer_name,'phase':phase,'layer':layer,'query_positions':positions,
              'token_context':_cpu(context),'q':_cpu(q),'kwargs':saved_kwargs,'output':_cpu(result),
              'cos':_cpu(cos),'sin':_cpu(sin),'cache_meta':cache_meta,'anchors':selected}
        torch.save(data,directory/(stem+'.pt'))
    except Exception:
        (directory/(stem+'_error.json')).write_text(json.dumps({'error':traceback.format_exc()},indent=2))
    _seen.update((layer,phase,p) for p in selected)
