"""Capture the same complete sampled response in the independent Actor and teacher-forced rollout."""
from pathlib import Path
import json
import os
import re
import torch

MODE = 'actor_paged_combined_shared_fc1_post_compress_order'

def directory():
    return Path(os.environ['DSV4_QAT_RUN']) / 'focus'

def mark(prompt, response):
    folder = directory()
    folder.mkdir(exist_ok=True)
    target = folder / 'selection.json'
    if target.exists():
        return
    import tempfile
    with tempfile.NamedTemporaryFile(dir=folder, mode='w', delete=False) as out:
        json.dump(dict(prompt=list(prompt), response=list(response)), out)
        p = Path(out.name)
    try:
        os.link(p, target)
    except FileExistsError:
        pass
    finally:
        p.unlink()

def install(model, side):
    folder = directory()
    selection = folder / 'selection.json'
    saved = set()
    state = {}
    owned = folder / side / str(os.getpid())

    def enter(mod, args, kwargs):
        state.pop('offset', None)
        if len(saved) == 43 or side == 'actor' and os.getenv('DSV4_ALIGN_VARIANT_LABEL') != MODE:
            return
        if 'ids' not in state:
            if not selection.is_file():
                return
            row = json.loads(selection.read_text())
            state['ids'] = torch.tensor(row['prompt'] + row['response'])
            state['pos'] = torch.arange(len(state['ids']))
        ids = kwargs.get('input_ids')
        pos = kwargs.get('positions')
        if pos is None:
            pos = kwargs.get('position_ids')
        if ids is None and args:
            ids = args[0]
        if pos is None and len(args) > 1:
            pos = args[1]
        if not isinstance(ids, torch.Tensor) or not isinstance(pos, torch.Tensor):
            return
        n = len(state['ids'])
        if ids.numel() < n or pos.numel() != ids.numel() or ids.is_floating_point():
            return
        ids, pos = ids.detach().flatten().cpu(), pos.detach().flatten().cpu()
        for start in (pos == 0).nonzero().flatten().tolist():
            if torch.equal(ids[start:start+n], state['ids']) and torch.equal(pos[start:start+n], state['pos']):
                state['offset'] = start
                state['rows'] = len(ids)
                return

    model.register_forward_pre_hook(enter, with_kwargs=True)
    for name, module in model.named_modules():
        match = re.search(r'layers\.(\d+)$', name)
        if not match:
            continue
        depth = int(match[1])
        def save(mod, args, result, depth=depth):
            if depth in saved or 'offset' not in state:
                return
            t = result[0] if isinstance(result, (tuple, list)) else result
            if not isinstance(t, torch.Tensor) or t.shape[0] != state['rows']:
                return
            i = state['offset']
            owned.mkdir(parents=True, exist_ok=True)
            x = t[i:i+len(state['ids'])].detach().cpu().contiguous()
            torch.save(dict(ids=state['ids'], pos=state['pos'], x=x, physical_rows=state['rows']),
                       owned / f'{depth:02d}.pt')
            saved.add(depth)
        module.register_forward_hook(save)
