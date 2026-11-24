"""Independently teacher-force real normal-batch GSM8K trajectories with the live vLLM model."""
import asyncio,json,os
from pathlib import Path
import numpy as np
from vllm import SamplingParams

async def capture(server,request,prompt_ids,response,native,sampling,lora,priority,rank):
    if not hasattr(server,'_long_teacher_task'):
        server._long_teacher_task=asyncio.create_task(probe(server,list(prompt_ids),lora,rank or 0))
    await asyncio.shield(server._long_teacher_task)

async def probe(server,prompt,lora,rank):
    r=Path(os.environ['DSV4_QAT_RUN'])
    src=r.parent/'wireTrain2A/samples/step_0001_chosen_token.npz'
    with np.load(src) as z:
        rows=[('wireTrain2A',i,z['responses'][i,z['response_mask'][i].astype(bool)].astype(int).tolist(),
               z['rollout_log_probs'][i,z['response_mask'][i].astype(bool)].tolist())
              for i,p in enumerate(z['prompts']) if p.tolist()==prompt+[0]*(len(p)-len(prompt))]
    assert len(rows)==8
    archive=Path(os.environ['DSV4_ALIGN_VERIFIED_REPORT'])
    verified=json.loads(archive.read_text())['rows']
    for item in verified:
        assert item['prompt_length']==len(prompt)
        rows.append(('mixMM1123A',item['batch_row'],item['response_ids'],item['values']['native']))
    path=r/'same_token'/'teacher.json';path.parent.mkdir(exist_ok=True)
    result=[]
    for source,i,reply,actual in rows:
        rec=dict(source=source,row=i,prompt=prompt,response=reply,native=actual,teacher={})
        for name,n in [('first8',8),('full',len(reply))]:
            tokens=prompt+reply[:n]
            params=SamplingParams(max_tokens=1,temperature=1.,top_k=-1,top_p=1.,detokenize=False,prompt_logprobs=0)
            last=None
            async for piece in server.engine.generate(prompt={'prompt_token_ids':tokens},sampling_params=params,
                    request_id=f'wire-long-{source}-{i}-{name}',lora_request=lora,priority=0,data_parallel_rank=rank):
                last=piece
            assert last is not None and list(last.prompt_token_ids)==tokens and len(last.prompt_logprobs)==len(tokens)
            rec['teacher'][name]=dict(scores=[float(last.prompt_logprobs[j][tokens[j]].logprob)
                                      for j in range(len(prompt),len(tokens))],cached=getattr(last,'num_cached_tokens',None))
        result.append(rec)
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,allow_nan=False));tmp.replace(path)

    from verl.utils.debug.long_native_edges import exercise
    await exercise(server,prompt,lora,rank)
