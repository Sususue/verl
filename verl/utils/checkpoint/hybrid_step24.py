"""Preserve the serialized Adam bias-correction step on a hybrid optimizer restore."""
import torch


def saved_hybrid_steps(optimizer, state_dict, hybrid_optimizers):
    outer = list(getattr(optimizer, 'chained_optimizers', [optimizer]))
    if len(outer) == 1:
        states = [state_dict]
    elif isinstance(state_dict, dict):
        states = [state_dict[key] for key in sorted(state_dict)]
    else:
        states = state_dict
    assert len(states) == len(outer)
    result = []
    for item, state in zip(outer, states):
        inner = getattr(item, 'optimizer', None)
        if inner not in hybrid_optimizers:
            continue
        groups = state['optimizer']['param_groups']
        steps = {float(g['step']) for g in groups if 'step' in g}
        if len(steps) > 1:
            raise ValueError(f'Conflicting serialized hybrid Adam steps: {steps}')
        if steps:
            result.append((inner, steps.pop()))
    return result


@torch.no_grad()
def restore_hybrid_steps(pairs, rank):
    for hybrid, saved in pairs:
        restored = 0
        for optimizer in hybrid.sub_optimizers:
            if not isinstance(optimizer, (torch.optim.AdamW, torch.optim.Adam)):
                continue
            for state in optimizer.state.values():
                step = state.get('step')
                if isinstance(step, torch.Tensor):
                    step.fill_(saved)
                    restored += 1
        print(f'HYBRID_ADAM_STEP_RESTORE rank={rank} step={saved} parameters={restored}', flush=True)


from contextlib import contextmanager
import json
import os
from pathlib import Path


@contextmanager
def audit_loaded_moments(optimizer, enabled, rank):
    """Compare selected loaded shard values with live state after suboptimizer reconstruction."""
    if not enabled or os.environ.get('DSV4_ALIGN_OPTIMIZER_AUDIT') != '1':
        yield
        return
    wrappers = []
    samples = []
    counts = {}
    for i, outer in enumerate(getattr(optimizer, 'chained_optimizers', [optimizer])):
        setter = getattr(outer, '_set_main_param_and_optimizer_states', None)
        if setter is None:
            continue
        counts[i] = 0
        def observed(param, tensors, setter=setter, outer=outer, i=i):
            index = counts[i]
            counts[i] += 1
            if index in (0,1,5,21,55,103,197,307):
                for key in ('param','exp_avg','exp_avg_sq'):
                    value = tensors.get(key)
                    if isinstance(value, torch.Tensor) and value.numel():
                        points = torch.linspace(0,value.numel()-1,min(value.numel(),37),dtype=torch.int64,device='cpu')
                        expected = value.reshape(-1).index_select(0,points.to(value.device)).detach().cpu().clone()
                        samples.append((outer,i,index,param,key,points,expected))
            return setter(param,tensors)
        outer._set_main_param_and_optimizer_states = observed
        wrappers.append(outer)
    try:
        yield
        report = []
        for outer,i,index,param,key,points,expected in samples:
            values = outer._get_main_param_and_optimizer_states(param)
            actual = values[key].reshape(-1).index_select(0,points.to(values[key].device)).detach().cpu()
            report.append(dict(optimizer=i,parameter=index,key=key,n=len(points),match=torch.equal(expected,actual),
                               max_abs=float((expected.double()-actual.double()).abs().max()),
                               expected_nonzero=int(expected.count_nonzero()),dtype=str(expected.dtype)))
        folder = Path(os.environ['DSV4_QAT_RUN'])/'audit'
        (folder/f'optimizer_load_rank{rank}.json').write_text(json.dumps(dict(counts=counts,samples=report)))
    finally:
        for outer in wrappers:
            del outer._set_main_param_and_optimizer_states
