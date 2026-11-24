"""Record the first real Actor Q/K rotary inputs, angles and output at original weights."""
import os
from pathlib import Path
import torch


def install():
    from mindspeed_bridge.models.deepseek_v4_flash.modelling_deepseek_v4_flash import rope_utils as module
    if hasattr(module, '_original_trace_installed'):
        return
    module._original_trace_installed = True
    original = module._apply_rotary_pos_emb_bshd
    folder = Path(os.environ['DSV4_QAT_RUN'])/'rotary_actor'/str(os.getpid())
    seen = set()

    def capture(t, freqs, *args, **kwargs):
        out = original(t, freqs, *args, **kwargs)
        label = os.environ.get('DSV4_ALIGN_VARIANT_LABEL')
        if label not in ('initial_reference', 'initial_repeat','initial_inline'):
            return out
        key = (label,t.ndim,tuple(t.shape[2:]),bool(kwargs.get('inverse',False)))
        if key in seen or len(seen) >= 12:
            return out
        seen.add(key)
        # A single logical head and at most 128 sequence positions suffice to test rotation.
        def sample(x):
            x=x.detach()[:128]
            return x.reshape(len(x),-1,x.shape[-1])[:,:1].contiguous().cpu()
        folder.mkdir(parents=True,exist_ok=True)
        torch.save(dict(x=sample(t),freqs=sample(freqs),result=sample(out),shape=list(t.shape),
                        keywords={k:v for k,v in kwargs.items() if isinstance(v,(bool,int,float))},
                        fused=module.rope_fused.fused_enabled(),mode=module.rope_fused.rope_mode()),
                   folder/(label+'_'+str(len(seen))+'.pt'))
        return out
    module._apply_rotary_pos_emb_bshd = capture
