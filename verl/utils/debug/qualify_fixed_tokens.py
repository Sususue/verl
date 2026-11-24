"""Replay real native scores on frozen original weights using the exact trainable Actor forward."""
import json,os,re
from pathlib import Path
import numpy as np
import torch

def capture(worker,case,prompt):
    from verl.utils.debug.train_alignment import configure_training
    configure_training(worker,1)
    os.environ['DSV4_LONG_REPLAY_CASE']=case
    worker._qualify_case=case
    if getattr(worker,'_qualify_hooks',False):return True
    worker._qualify_hooks=True
    modules=worker.actor.engine.module
    if not isinstance(modules,(list,tuple)):modules=[modules]
    models=[m for root in modules for m in root.modules() if 'decoder' in m._modules and 'embedding' in m._modules and hasattr(m.decoder,'layers')]
    assert len(models)==1
    model=models[0];ctx={}
    def enter(mod,args,kwargs):
        ctx.clear()
        ids=kwargs.get('input_ids',args[0] if args else None)
        pos=kwargs.get('position_ids',args[1] if len(args)>1 else None)
        if not isinstance(ids,torch.Tensor) or not isinstance(pos,torch.Tensor):return
        ids=ids.detach().cpu().flatten();pos=pos.detach().cpu().flatten();n=len(prompt)
        if ids[:n].tolist()!=prompt or pos[:n].tolist()!=list(range(n)):return
        out=Path(os.environ['DSV4_QAT_RUN'])/'replay_edges'/worker._qualify_case/str(worker.rank)
        out.mkdir(parents=True,exist_ok=True)
        torch.save(dict(ids=ids,pos=pos),out/'context.pt')
        ctx.update(out=out,n=n,rows=len(ids))
    model.register_forward_pre_hook(enter,with_kwargs=True)
    from verl.utils.debug.long_actor_edges import install
    install(model,ctx,worker)
    def hook(i):
        def save(mod,args,output):
            if not ctx:return
            tensor=output[0] if isinstance(output,(list,tuple)) else output
            f=ctx['out']/f'L{i:02d}.pt'
            if not f.exists() and isinstance(tensor,torch.Tensor) and tensor.shape[0]>=ctx['n']:
                torch.save(tensor[:ctx['n']].detach().reshape(ctx['n'],-1).cpu().contiguous(),f)
        return save
    assert len(model.decoder.layers)==43
    for i,layer in enumerate(model.decoder.layers):layer.register_forward_hook(hook(i))
    return True

def run(trainer,batch):
    import transfer_queue as tq
    from verl.protocol import DataProto
    from verl.trainer.ppo.v1.trainer_base import response_from_nested
    from tensordict._reductions import _make_td
    from tensordict._td import TensorDict
    from tensordict.tensorclass import NonTensorData,NonTensorStack
    from torch._dynamo.decorators import _DimRange
    from torch.nested._internal.nested_tensor import NestedTensor,_rebuild_njt
    root=Path(os.environ['DSV4_QAT_RUN']);runs=root.parent
    out=root/'same_token';out.mkdir(exist_ok=True)
    live=tq.kv_batch_get(keys=batch.keys,partition_id=batch.partition_id)
    with torch.serialization.safe_globals([TensorDict,NonTensorData,NonTensorStack,_make_td,NestedTensor,_rebuild_njt,_DimRange]):
        short=torch.load(runs/'mixMM1123A/samples/original_batch.pt',weights_only=True,map_location='cpu')
    prompt=short['prompts'][0].tolist()
    assert len(short)==len(live)==8 and len(prompt)==57
    assert all(p.tolist()==prompt for p in short['prompts'].unbind())
    with np.load(runs/'wireTrain2A/samples/step_0001_chosen_token.npz') as z:
        ids=[i for i,pr in enumerate(z['prompts']) if pr.tolist()==prompt+[0]*(len(pr)-len(prompt))]
        assert len(ids)==8
        true=[torch.from_numpy(z['responses'][i,z['response_mask'][i].astype(bool)]).long() for i in ids]
        scores=[torch.from_numpy(z['rollout_log_probs'][i,z['response_mask'][i].astype(bool)]) for i in ids]
    wg=trainer.actor_rollout_wg
    summary={}
    for case in ('live','verified64','long_first8','long_to127','long_to129','long_to136','long_to160','long_full','long_full_repeat','long_current_live','verified64_repeat'):
        data=(live if case=='live' else short).clone()
        if case.startswith('long_'):
            cap={'long_first8':8,'long_to127':127-len(prompt),'long_to129':129-len(prompt),'long_to136':136-len(prompt),'long_to160':160-len(prompt)}.get(case,512)
            seq=[t[:cap] for t in true];ps=[s[:cap] for s in scores];pr=torch.tensor(prompt)
            if case=='long_current_live':
                with np.load(runs/'wireNative1123B/same_token/live.npz') as z:
                    assert all(prs.tolist()==prompt for prs in z['prompts'])
                    masks=z['response_mask'].astype(bool)
                    seq=[torch.from_numpy(t[m]).long() for t,m in zip(z['responses'],masks)]
                    ps=[torch.from_numpy(t[m]) for t,m in zip(z['rollout_log_probs'],masks)]

            ids2=[torch.cat((pr,t)) for t in seq]
            fields=dict(responses=seq,input_ids=ids2,response_mask=[torch.ones_like(t) for t in seq],
                        position_ids=[torch.arange(len(t)) for t in ids2],rollout_log_probs=ps,
                        loss_mask=[torch.ones_like(t) for t in seq],
                        rm_scores=[torch.zeros(len(t)) for t in seq])
            for k,values in fields.items():
                orig=data[k]
                exemplar=orig[0]
                # Preserve the original row layout, including any leading singleton dimensions.
                rows=[v.to(orig.dtype).reshape(*exemplar.shape[:-1],len(v)) for v in values]
                assert exemplar.ndim==1,(k,exemplar.shape)
                data.set(k,torch.nested.nested_tensor(rows,layout=torch.jagged))
        local=tq.kv_batch_put(keys=batch.keys,partition_id=batch.partition_id,fields=data)
        local.extra_info.update(batch.extra_info)
        local.extra_info.update(calculate_entropy=True,compute_loss=False,temperature=trainer.config.actor_rollout_ref.rollout.temperature)
        wg.execute_all_sync('actor_rollout_execute_with_func_generator',capture,case,prompt)
        wg.compute_log_prob(local)
        raw=tq.kv_batch_get(keys=local.keys,partition_id=local.partition_id,
                            select_fields=['log_probs','rollout_log_probs','response_mask','responses','prompts'])
        raw['old_log_probs']=response_from_nested(raw.pop('log_probs'),raw['response_mask'])
        padded=DataProto(batch=raw.to_padded_tensor()).batch
        nums={k:padded[k].detach().cpu().numpy() for k in padded.keys()}
        np.savez_compressed(out/(case+'.npz'),**nums)
        mask=nums['response_mask'].astype(bool)
        x=nums['old_log_probs'][mask].astype(float);y=nums['rollout_log_probs'][mask].astype(float)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        summary[case]=dict(tokens=len(x),neq=int((x!=y).sum()),logp=float(abs(x-y).mean()),
            prob=float(abs(np.exp(x)-np.exp(y)).mean()),pearson=float(np.corrcoef(np.exp(x),np.exp(y))[0,1]))
        (out/'scores.json').write_text(json.dumps(summary,indent=2))
        print('QUALIFY_SAME_TOKENS',case,summary[case],flush=True)
    raise RuntimeError('Completed independent same-token qualification before optimizer')
