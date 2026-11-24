"""Bounded snapshots on independently executed scorer paths at the original policy."""
import os
from pathlib import Path
import torch
import hashlib


def directory(side):
    path=Path(os.environ['DSV4_QAT_RUN'])/'head_trace'/side/str(os.getpid())
    path.mkdir(parents=True,exist_ok=True)
    return path


def install_actor(engine):
    if os.getenv("DSV4_ALIGN_LONG_POLICY") == "1": return
    if getattr(engine,'_initial_head_probe',False) or not hasattr(engine,'_lm_head_logits_processor'):
        return
    engine._initial_head_probe=True
    infer=engine.infer_batch
    def repeat_resident(data,*args,**kwargs):
        if os.environ.get('DSV4_ALIGN_VARIANT_LABEL')!='initial_reference':
            return infer(data,*args,**kwargs)
        # Materialize only this tiny diagnostic batch. Both forwards execute
        # within the same eval_mode and keep the very same weights resident.
        saved=data.clone()
        actual=infer(data,*args,**kwargs)
        os.environ['DSV4_ALIGN_VARIANT_LABEL']='initial_inline'
        try:
            infer(saved,*args,**kwargs)
        finally:
            os.environ['DSV4_ALIGN_VARIANT_LABEL']='initial_reference'
        return actual
    engine.infer_batch=repeat_resident
    process=engine._lm_head_logits_processor
    seen=set()
    def observe(logits,label,temperature,**kwargs):
        mode=os.environ.get('DSV4_TAIL_COMPARISON') or os.environ.get('DSV4_ALIGN_VARIANT_LABEL')
        if mode not in ('initial_reference','initial_repeat','initial_inline','actor_paged_combined_shared_fc1_post_compress_order','head_contract','head_tile','head_sampler','head_all','head_all_repeat') or mode in seen:
            return process(logits,label,temperature,**kwargs)
        seen.add(mode)
        batch=kwargs['batch']
        tokens=batch['input_ids'].unbind(0)[0].reshape(-1)
        flat=logits.reshape(-1,logits.shape[-1])
        stop=min(len(tokens),len(flat))
        idx=torch.arange(max(0,stop-16),stop,device=logits.device)
        raw=flat.index_select(0,idx).detach().cpu()
        labels=label.reshape(-1).index_select(0,idx).detach().cpu()
        temps=temperature.reshape(-1).index_select(0,idx).detach().cpu()
        result=process(logits,label,temperature,**kwargs)
        torch.save(dict(token_ids=tokens.detach().cpu(),labels=labels,temperatures=temps,positions=idx.cpu(),
                        logits=raw,native_log_probs=result['log_probs'].reshape(-1).index_select(0,idx).detach().cpu(),
                        training=[module.training for module in engine.module],
                        cpu_rng=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()),
                   directory('actor')/(mode+'.pt'))
        return result
    engine._lm_head_logits_processor=observe


def install_rollout(model):
    if os.getenv("DSV4_ALIGN_LONG_POLICY") == "1": return
    if getattr(model,'_initial_head_probe',False):
        return
    model._initial_head_probe=True
    context={}
    from verl.utils.debug.hc_shape28 import install as install_hc_shape
    install_hc_shape(model,context)
    def before(mod,args,kwargs):
        for i,key in enumerate(('input_ids','positions')):
            t=kwargs.get(key)
            if t is None and len(args)>i:
                t=args[i]
            if isinstance(t,torch.Tensor):
                context[key]=t.detach().cpu()
    model.register_forward_pre_hook(before,with_kwargs=True)
    for name in ('compute_logits','compute_logits_local'):
        if not hasattr(model,name):
            continue
        function=getattr(model,name)
        counts=[0]
        def observed(hidden_states,*args,op=function,key=name,n=counts,**kwargs):
            logits=op(hidden_states,*args,**kwargs)
            pos = context.get("positions")
            rank_ok = not torch.distributed.is_initialized() or torch.distributed.get_rank() in (0,3)
            if isinstance(logits,torch.Tensor) and n[0]<60 and rank_ok and pos is not None and pos.numel() in (57,65,114,130):
                rows = sorted(set(i for i in [0,1,54,55,56,57,63,64,len(logits)-1] if 0 <= i < len(logits)))
                idx = torch.tensor(rows,device=logits.device)
                torch.save(dict(context=dict(context),logit_rows=rows, total_logit_rows=len(logits), logits=logits.index_select(0,idx).detach().cpu(),
                                hidden=hidden_states.index_select(0,idx).detach().cpu()),
                           directory('rollout')/f'{key}_{n[0]:02d}.pt')
                n[0]+=1
            return logits
        setattr(model,name,observed)
