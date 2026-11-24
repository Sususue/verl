"""Observe vLLM's real long prefill and a real decode of an identical causal prefix."""
import json
import os
from pathlib import Path
import re

import torch


def install(model):
    import inspect
    evidence=Path(os.environ["DSV4_QAT_RUN"])/"native_edges"/"observed"
    evidence.mkdir(parents=True,exist_ok=True)
    sample=evidence/f"registration_{os.getpid()}.json"
    if not sample.exists():
        sample.write_text(json.dumps(dict(observer=__file__,model=type(model).__name__,model_file=inspect.getfile(type(model)),
            inner_model=type(getattr(model,'model',None)).__name__,named_layers=[k for k,_ in model.named_modules() if k.endswith("layers.0")][:4])))
    if globals().get("_installed_in_worker",False) or getattr(model,"_long_edges",False):
        return
    globals()["_installed_in_worker"]=True
    model._long_edges=True
    root=Path(os.environ["DSV4_QAT_RUN"])/"native_edges"
    marker=root/"active.json"
    context={}
    seen=set()
    def enter(mod,args,kwargs):
        context.clear()
        if not marker.is_file():return
        obs=json.loads(marker.read_text())
        ids=kwargs.get("input_ids",args[0] if args else None)
        pos=kwargs.get("positions",args[1] if len(args)>1 else None)
        if not isinstance(ids,torch.Tensor) or not isinstance(pos,torch.Tensor):return
        ids,pos=ids.detach().flatten().cpu(),pos.detach().flatten().cpu()
        sequence=obs["ids"];p=obs["position"];tag=obs["tag"]
        if tag in seen or len(ids)!=len(pos):return
        if obs["phase"]=="full":
            n=len(sequence)
            if ids[:n].tolist()!=sequence or pos[:n].tolist()!=list(range(n)):return
            at=p
        else:
            ix=((pos==p)&(ids==sequence[p])).nonzero().flatten()
            if len(ix)!=1 or int((pos>0).sum())!=1:return
            at=int(ix[0])
        dest=root/tag/str(os.getpid());dest.mkdir(exist_ok=True,parents=True)
        (dest/"context.json").write_text(json.dumps(dict(actual_ids=ids.tolist(),actual_positions=pos.tolist(),**obs)))
        context.update(out=dest,index=at,n=len(ids),tag=tag,position=p)
    model.register_forward_pre_hook(enter,with_kwargs=True)
    def hook(label,depth):
        def observe(mod,args,output):
            from verl.utils.debug.native_forward_tap import _ctx
            if not _ctx:return
            context.update(out=_ctx["out"],index=_ctx["at"],n=_ctx["rows"],tag=_ctx["tag"],position=_ctx["position"])
            values={}
            for name,parts in [("input",args),("output",output)]:
                parts=parts if isinstance(parts,(list,tuple)) else (parts,)
                for j,x in enumerate(parts):
                    if isinstance(x,torch.Tensor) and x.ndim and x.shape[0]==context["n"]:
                        values[name+str(j)]=x[context["index"]].detach().cpu().contiguous()
            if values:torch.save(values,context["out"]/(label+".pt"))
            if label=="L42":seen.add(context["tag"])
        return observe
    for name,layer in model.named_modules():
        m=re.search(r"(?:^|\.)layers\.(\d+)(?:\.(self_attn|mlp|mlp\.(?:gate|experts|shared_experts)|self_attn\.(?:wq_a|wkv|wq_b|wo_b|q_norm|kv_norm)))?$",name)
        if m:
            tag="L"+m[1].zfill(2)+("_"+m[2].replace(".","_") if m[2] else "")
            if m[2] is None:
                def before(mod,args,depth=int(m[1])):
                    if context:context["layer"]=depth
                layer.register_forward_pre_hook(before)
            layer.register_forward_hook(hook(tag,int(m[1])))
    from importlib import import_module
    sparse=import_module("vllm_ascend.attention.sparse_flash_mla")
    actual,meta=sparse._get_sparse_flash_mla_ops()
    def measured(q,**kw):
        y=actual(q,**kw)
        from verl.utils.debug.native_forward_tap import _ctx
        if _ctx:
            context.update(out=_ctx['out'],index=_ctx['at'],n=_ctx['rows'],tag=_ctx['tag'],layer=_ctx.get('layer'),position=_ctx['position'])
        else:context.clear()
        if not context or context.get("layer") not in (0,2,3,4,8,10,11,29,42):return y
        idx=context["index"];p=context["position"];depth=context["layer"]
        if q.ndim<2 or idx>=len(q):return y
        f=context["out"]/f"L{depth:02d}_sparse.pt"
        if f.exists():return y
        data=dict(q=q[idx].detach().cpu(),out=y[0][idx].detach().cpu(),p=p)
        for name in ("seqused_ori_kv","cu_seqlens_q"):
            if isinstance(kw.get(name),torch.Tensor):data[name]=kw[name].detach().cpu()
        ratio=int(kw.get("cmp_ratio") or 1);data["ratio"]=ratio
        for label,beg,end in [("ori",max(0,p-127),p+1),("cmp",0,(p+1)//ratio)]:
            cache=kw.get(label+"_kv");table=kw.get(label+"_block_table")
            if cache is None or table is None or end<=beg:continue
            ridx=_ctx["request_index"]
            cu=kw["cu_seqlens_q"].detach().cpu().tolist()
            if ridx+1>=len(cu) or (cu[ridx],cu[ridx+1])!=_ctx["request_span"]:continue
            pages=table[ridx].detach().cpu().long();n=cache.shape[1]
            positions=torch.arange(beg,end);slots=positions//n
            if slots[-1]>=len(pages):continue
            blocks=pages[slots]
            if int(blocks.min())<0 or int(blocks.max())>=len(cache):continue
            data[label]=cache[blocks.to(cache.device),(positions%n).to(cache.device)].detach().cpu()
            data[label+"_logical"]=positions;data[label+"_table"]=pages
        idxs=kw.get("cmp_sparse_indices")
        if isinstance(idxs,torch.Tensor):data["indices"]=idxs[idx].detach().cpu()
        torch.save(data,f)
        return y
    sparse._get_sparse_flash_mla_ops=lambda:(measured,meta)


async def exercise(server,prompt,lora,rank):
    from vllm import SamplingParams
    root=Path(os.environ["DSV4_QAT_RUN"])
    selection=json.loads((root/"probe_specs.json").read_text())
    dest=root/"native_edges";dest.mkdir(exist_ok=True)
    try:
        with (dest/"owner.pid").open("x") as out:
            print(os.getpid(),file=out)
    except FileExistsError:
        return
    marker=dest/"active.json";receipts=[]
    for case in selection:
        tokens=case["ids"];p=case["position"]
        assert tokens[:len(prompt)]==prompt and p+1<len(tokens)
        for phase in ("full","decode"):
            tag=f"row{case['row']}_p{p}_{phase}"
            obs=dict(**case,tag=tag,phase=phase)
            tmp=marker.with_suffix(".tmp");tmp.write_text(json.dumps(obs));tmp.replace(marker)
            if phase=="full":
                pre=tokens;params=SamplingParams(max_tokens=1,prompt_logprobs=0,detokenize=False,temperature=0)
            else:
                pre=tokens[:p];params=SamplingParams(max_tokens=2,min_tokens=2,allowed_token_ids=[tokens[p]],
                                                      detokenize=False,temperature=0)
            last=None
            async for part in server.engine.generate(prompt={"prompt_token_ids":pre},sampling_params=params,
                    request_id="edges-"+tag,lora_request=lora,priority=0,data_parallel_rank=rank):
                last=part
            assert last is not None and list(last.prompt_token_ids)==pre
            if phase=="decode":assert list(last.outputs[0].token_ids)==[tokens[p]]*2
            receipts.append(dict(tag=tag,prompt=len(pre),cached=getattr(last,"num_cached_tokens",None),
                                 generated=list(last.outputs[0].token_ids)))
            (dest/"receipts.json").write_text(json.dumps(receipts))
    marker.unlink()


def install_actor(model,ctx,worker):
    root=Path(os.environ["DSV4_QAT_RUN"])
    cases=json.loads((root/"probe_specs.json").read_text())
    owned=[r for r in cases if r["rank"]==worker.rank]
    if not owned:return
    from verl.utils.debug import long_paged
    original=long_paged.paged
    def paged(q,kv,sink,indices,scale,ratio):
        y=original(q,kv,sink,indices,scale,ratio)
        if not ctx or ctx.get("layer") not in (0,2,3,4,8,10,11,29,42):return y
        depth=ctx["layer"]
        for obs in owned:
            if worker._qualify_case!=obs.get("actor_case","long_full") or len(obs["ids"])!=len(q):continue
            p=obs["position"];n=len(q)
            dest=root/"native_edges"/f"row{obs['row']}_p{p}_actor"/str(worker.rank)
            dest.mkdir(exist_ok=True,parents=True)
            f=dest/f"L{depth:02d}_sparse.pt"
            if f.exists():continue
            start=max(0,p-127);end=(p+1)//ratio if ratio>1 else 0
            data=dict(q=q[p,0].detach().cpu(),out=y[p,0].detach().cpu(),ratio=ratio,p=p,
                      ori=kv[start:p+1,0].detach().cpu(),ori_logical=torch.arange(start,p+1))
            if end:
                data["cmp"]=kv[n:n+end,0].detach().cpu();data["cmp_logical"]=torch.arange(end)
                idc=indices[0,p,128:]
                data["indices"]=(idc[idc>=n]-n).sort().values.detach().cpu()
            torch.save(data,f)
        return y
    long_paged.paged=paged
    for i,layer in enumerate(model.decoder.layers):
        def entering(mod,args,i=i):
            if ctx:ctx["layer"]=i
        layer.register_forward_pre_hook(entering)
    for name,mod in model.named_modules():
        m=re.fullmatch(r"decoder\.layers\.(\d+)(?:\.(self_attention|mlp|mlp\.(?:router|experts|shared_experts)|self_attention\.(?:linear_q_down_proj|linear_kv_proj|linear_q_up_proj|linear_proj|q_layernorm|kv_layernorm)))?",name)
        if not m:continue
        alias={None:"","self_attention":"_self_attn","mlp":"_mlp","mlp.router":"_mlp_gate",
               "mlp.experts":"_mlp_experts","mlp.shared_experts":"_mlp_shared_experts",
               "self_attention.linear_q_down_proj":"_self_attn_wq_a","self_attention.linear_kv_proj":"_self_attn_wkv",
               "self_attention.q_layernorm":"_self_attn_q_norm","self_attention.kv_layernorm":"_self_attn_kv_norm",
               "self_attention.linear_q_up_proj":"_self_attn_wq_b","self_attention.linear_proj":"_self_attn_wo_b"}
        label="L"+m[1].zfill(2)+alias[m[2]]
        def observe(module,args,output,label=label):
            if not ctx:return
            for obs in owned:
                if worker._qualify_case!=obs.get("actor_case","long_full"):continue
                if len(obs["ids"])!=ctx["rows"]:continue
                p=obs["position"];dest=root/"native_edges"/f"row{obs['row']}_p{p}_actor"/str(worker.rank)
                dest.mkdir(exist_ok=True,parents=True)
                if label=="L00":
                    identity=torch.load(ctx["out"]/"context.pt",weights_only=True,map_location="cpu")
                    assert identity["ids"].tolist()==obs["ids"] and identity["pos"].tolist()==list(range(len(obs["ids"])))
                    (dest/"context.json").write_text(json.dumps(obs))
                values={}
                for group,parts in (("input",args),("output",output)):
                    if not isinstance(parts,(list,tuple)):parts=(parts,)
                    for j,x in enumerate(parts):
                        if isinstance(x,torch.Tensor) and x.ndim and x.shape[0]==ctx["rows"]:
                            values[group+str(j)]=x[p].detach().cpu().contiguous()
                if values:torch.save(values,dest/(label+".pt"))
        mod.register_forward_hook(observe)
