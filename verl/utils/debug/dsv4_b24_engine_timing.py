"""Task-only host wall-time spans. Adds no device barrier or numerical change."""
import functools
import json
import os
from pathlib import Path
import time


def _optimizer_state_metadata(optimizer):
    """Read metadata only; avoid copies, values, or double counting wrappers."""
    pending=[optimizer];seen=set();leaves=[]
    while pending:
        item=pending.pop()
        if item is None or id(item) in seen:
            continue
        seen.add(id(item))
        children=list(getattr(item,'chained_optimizers',[]) or [])
        children+=list(getattr(item,'cpu_optimizers',[]) or [])
        if getattr(item,'gpu_optimizer',None) is not None:
            children.append(item.gpu_optimizer)
        # ChainedOptimizer.optimizer is a property that asserts when >1 sub-optimizer
        # is chained (CPU offload); getattr's default only covers AttributeError.
        try:
            sub=item.optimizer
        except Exception:
            sub=None
        if sub is not None:
            children.append(sub)
        if children:
            pending.extend(children)
            continue
        states={}
        for state in getattr(item,'state',{}).values():
            for name,value in state.items():
                if hasattr(value,'dtype') and hasattr(value,'numel'):
                    key=f'{name}:{value.dtype}:{value.device.type}'
                    states[key]=states.get(key,0)+value.numel()
        leaves.append({'class':type(item).__module__+'.'+type(item).__name__,
                       'state_numel_by_name_dtype_device':states})
    return leaves


def engine_timed(phase):
    def decorate(function):
        @functools.wraps(function)
        def wrapped(self,*args,**kwargs):
            directory=os.environ.get('DSV4_B24_AUDIT_DIR')
            if not directory:
                return function(self,*args,**kwargs)
            started=time.perf_counter()
            ok=False
            try:
                result=function(self,*args,**kwargs)
                ok=True
                return result
            finally:
                ended=time.perf_counter()
                import torch
                rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else None
                record={'phase':phase,'pid':os.getpid(),'rank':rank,'time':time.time(),
                        'host_wall_seconds':ended-started,'success':ok,
                        'forward_only':bool(kwargs.get('forward_only',False)) if phase=='forward_backward_batch' else None,
                        'device_argument':str(kwargs.get('device',args[0] if args and phase=='to' else '')) if phase=='to' else None,
                        'timing_semantics':'host wall only; no added device synchronization; distributed ranks overlap'}
                if phase=='optimizer_step' and ok and not getattr(self,'_b24_optimizer_metadata_recorded',False):
                    record['optimizer_state_metadata']=_optimizer_state_metadata(self.optimizer)
                    self._b24_optimizer_metadata_recorded=True
                folder=Path(directory).parent/'metrics'
                folder.mkdir(parents=True,exist_ok=True)
                with (folder/f'engine_phases_{os.getpid()}.jsonl').open('a') as f:
                    f.write(json.dumps(record)+'\n')
        return wrapped
    return decorate
