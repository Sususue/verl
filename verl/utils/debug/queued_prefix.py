"""Submit both requests to one paused private engine before resuming its scheduler."""
import asyncio
import json
import os
from pathlib import Path
from vllm import SamplingParams


async def probe(server,lora):
    run=Path(os.environ['DSV4_ALIGN_SNAPSHOT']).parents[1]
    corpus=[json.loads(p.read_text()) for p in sorted((run.parent/'hcFixA/samples/rollout_self').glob('*.json'))]
    cases=[next(x for x in corpus if x['response_ids'][0]==v) for v in (5718,2581)]
    results=[];engine=server.engine
    async def collect(q):
        while True:
            out=q.get_nowait() or await q.get()
            if out.finished:return out
    for size in ('prefix','full'):
        for order in ((0,),(1,),(0,0),(0,1),(1,0),(0,1),(1,),(0,)):
            group=[];queues=[];serial=len(results)
            await engine.pause_generation(mode='keep',clear_cache=False)
            try:
                for slot,idx in enumerate(order):
                    case=cases[idx];resp=case['response_ids'];p=case['prompt_ids']
                    ids=p+resp[:5] if size=='prefix' else p+resp
                    opts={'logprob_token_ids':[int(resp[5])],'logprobs':1} if size=='prefix' else {'prompt_logprobs':0}
                    params=SamplingParams(max_tokens=1,temperature=1.,top_p=1.,top_k=-1,detokenize=False,**opts)
                    tag=f'forced-{os.getpid()}-{serial}-{slot}'
                    queue=await engine.add_request(request_id=tag,prompt={'prompt_token_ids':ids},params=params,lora_request=lora,priority=0,data_parallel_rank=0)
                    queues.append(queue);group.append(dict(tag=tag,case=idx,prefix=ids))
            finally:
                await engine.resume_generation()
            completed=await asyncio.wait_for(asyncio.gather(*(collect(q) for q in queues)),timeout=100)
            for rec,out in zip(group,completed):
                assert list(out.prompt_token_ids)==rec['prefix']
                resp=cases[rec['case']]['response_ids'];n=len(cases[rec['case']]['prompt_ids'])
                rec['scores']=[float(out.outputs[0].logprobs[0][resp[5]].logprob)] if size=='prefix' else [float(out.prompt_logprobs[n+i][v].logprob) for i,v in enumerate(resp)]
                rec['cached']=getattr(out,'num_cached_tokens',None)
            results.append(dict(size=size,order=order,results=group))
            path=run/'samples'/f'queued_prefix_{os.getpid()}.json'
            tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(results));tmp.replace(path)
