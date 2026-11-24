"""Identify the real active request at the DSV4 model entry and persist typed decoder boundaries."""
import json
import os
import sys
from pathlib import Path
import torch

_ctx={}
_logged=set()


def enter(model,ids,pos):
    _ctx.clear()
    run=Path(os.environ["DSV4_QAT_RUN"]);root=run/"native_edges"
    if "started" not in _logged:
        _logged.add("started")
        dest=root/"observed";dest.mkdir(parents=True,exist_ok=True)
        (dest/f"model_{os.getpid()}.json").write_text(json.dumps(dict(model=type(model).__name__,run=str(run),file=__file__)))
    marker=root/"active.json"
    if not marker.exists():return
    req=json.loads(marker.read_text());tag=req["tag"];p=req["position"]
    diag=root/"observed";diag.mkdir(parents=True,exist_ok=True)
    if tag not in _logged:
        _logged.add(tag)
        with (diag/f"{os.getpid()}.jsonl").open("a") as f:
            print(json.dumps(dict(tag=tag,model=type(model).__name__,file=__file__,
                ids=None if ids is None else list(ids.shape),pos=None if pos is None else list(pos.shape))),file=f)
    if not isinstance(ids,torch.Tensor) or not isinstance(pos,torch.Tensor):return
    ids=ids.detach().cpu().flatten();pos=pos.detach().cpu()
    if pos.ndim>1 and pos.numel()!=len(ids):
        pos=pos.reshape(-1,len(ids))
        if not all(torch.equal(row,pos[0]) for row in pos):return
        pos=pos[0]
    pos=pos.flatten()
    if len(ids)!=len(pos):return
    wanted=req["ids"]
    frame=sys._getframe(1)
    runner=None
    while frame is not None:
        if frame.f_code.co_name=="_model_forward":
            candidate=frame.f_locals.get("self")
            if hasattr(candidate,"input_batch") and hasattr(candidate,"query_start_loc"):
                runner=candidate
                break
        frame=frame.f_back
    del frame
    if runner is None:return
    ids_by_request=list(runner.input_batch.req_ids)
    starts=runner.query_start_loc.np
    schedule=diag/("schedule_"+str(os.getpid())+".jsonl")
    key="schedule_"+tag
    expected="edges-"+tag
    target_present=any(isinstance(r,str) and expected in r for r in ids_by_request)
    if target_present:key+="_target"
    if key not in _logged:
        _logged.add(key)
        with schedule.open("a") as out:
            print(json.dumps(dict(tag=tag,request_ids=ids_by_request,
                query_offsets=starts[:len(ids_by_request)+1].tolist(),token_rows=len(ids))),file=out)
    for ridx,request_id in enumerate(ids_by_request):
        if not isinstance(request_id,str) or (request_id!=expected and not request_id.startswith(expected+"-")):continue
        lo,hi=map(int, starts[ridx:ridx+2])
        if not (0<=lo<hi<=len(ids)):continue
        ix=((pos[lo:hi]==p)&(ids[lo:hi]==wanted[p])).nonzero().flatten()
        if len(ix)!=1:continue
        at=lo+int(ix[0])
        if req["phase"]=="decode" and hi-lo!=1:continue
        if req["phase"]=="full" and hi-lo<2:continue
        start=int(pos[lo])
        if pos[lo:hi].tolist()!=list(range(start,start+hi-lo)):continue
        if ids[lo:hi].tolist()!=wanted[start:start+hi-lo]:continue
        dest=root/tag/str(os.getpid());dest.mkdir(parents=True,exist_ok=True)
        (dest/"context.json").write_text(json.dumps(dict(actual_ids=ids.tolist(),actual_positions=pos.tolist(),request_index=ridx,request_id=request_id,request_span=[lo,hi],**req)))
        _ctx.update(request_index=ridx,request_span=(lo,hi),out=dest,at=at,rows=len(ids),position=p,phase=req["phase"],tag=tag)
        break


def save(depth,label,x,y):
    if not _ctx:return
    dest=_ctx["out"]/(f"L{depth:02d}"+("_"+label if label else "")+".pt")
    if dest.exists():return
    data={}
    for k,t in (("input0",x),("output0",y)):
        if isinstance(t,torch.Tensor) and t.ndim and len(t)==_ctx["rows"]:
            data[k]=t[_ctx["at"]].detach().cpu().contiguous()
    if data:torch.save(data,dest)
