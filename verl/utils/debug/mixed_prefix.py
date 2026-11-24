"""Paired causal-prefix replay on two DP ranks, holding both IDs and selected labels fixed."""
import asyncio
import json
import os
from pathlib import Path

from vllm import SamplingParams


async def compare(server, lora):
    run=Path(os.environ['DSV4_ALIGN_SNAPSHOT']).parents[1]
    source=run.parent/'hcFixA/samples/rollout_self'
    rows=[json.loads(p.read_text()) for p in sorted(source.glob('*.json'))]
    # These two responses ran concurrently on the same rank in the earlier experiment.
    cases=[next(r for r in rows if r['response_ids'][0]==token) for token in (5718,2581)]
    assert cases[0]['prompt_ids']==cases[1]['prompt_ids']
    results=[]
    async def one(rank,case,offset,serial,slot):
        prefix=case['prompt_ids']+case['response_ids'][:offset]
        label=case['response_ids'][offset]
        cfg=SamplingParams(max_tokens=1,temperature=1.,top_p=1.,top_k=-1,detokenize=False,logprobs=1,logprob_token_ids=[label])
        request_id=f'mixed-prefix-{os.getpid()}-{serial}-{slot}'
        last=None
        async for step in server.engine.generate(prompt={'prompt_token_ids':prefix},sampling_params=cfg,
                request_id=request_id,priority=0,data_parallel_rank=rank,lora_request=lora):
            last=step
        assert last is not None and list(last.prompt_token_ids)==prefix
        return dict(prefix=prefix,label=label,raw_logp=float(last.outputs[0].logprobs[0][label].logprob),
                    request_id=request_id,cached=getattr(last,'num_cached_tokens',None))
    for rank in (0,3,4):
        for offset in (0,4,5,6,7):
            for order in [(0,),(1,),(0,0),(0,1),(1,0),(1,),(0,)]:
                idx=len(results)
                values=await asyncio.gather(*(one(rank,cases[case],offset,idx,slot) for slot,case in enumerate(order)))
                results.append(dict(rank=rank,offset=offset,submitted_case_order=order,observed=values))
                dest=run/'samples'/f'mixed_prefix_{os.getpid()}.json'
                tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(results));tmp.replace(dest)
