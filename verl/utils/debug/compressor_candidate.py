"""Independent original-weight scoring and request-local compressor execution for a numerical ablation."""
import os
from pathlib import Path

import torch
from verl.utils.debug.train_alignment import with_local_backward


def record_ape(module, side, name):
    if side == 'rollout' and not (Path(os.environ['DSV4_QAT_RUN'])/'audit'/f'refit_{os.getpid()}.jsonl').is_file():
        return
    if getattr(module, '_alignment_ape_saved', False):
        return
    module._alignment_ape_saved = True
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return
    if not any(f'layers.{i}.' in name for i in (2, 4)):
        return
    output = Path(os.environ['DSV4_QAT_RUN']) / 'compressor_weights' / side
    output.mkdir(parents=True, exist_ok=True)
    torch.save(module.ape.detach().cpu(), output / (name + '.pt'))


def split_serving(module, hidden, state, metadata, state_meta, sin, cos, slots):
    """Run each prefill on its own pages and scatter the same valid rows to the original slots."""
    name = getattr(module.wkv, 'prefix', '')
    record_ape(module, 'rollout', name)
    from verl.utils.debug.bounded_compressor import serving
    bounded = serving(module, hidden, state, metadata, state_meta, sin, cos, slots)
    if bounded is not None:return bounded
    cu = metadata.query_start_loc
    batch = cu.numel() - 1
    if batch < 2 or hidden.shape[0] <= batch or torch.npu.is_current_stream_capturing():
        return None
    offsets = cu.tolist()
    starts = metadata.start_pos.reshape(-1).tolist()
    if len(starts) < batch or offsets[-1] > len(hidden):
        raise ValueError((offsets, starts, hidden.shape))
    ratio = module.compress_ratio
    output = hidden.new_zeros((len(slots), module.head_dim))
    consumed = 0
    for i, (lo, hi) in enumerate(zip(offsets, offsets[1:])):
        n = hi - lo
        if n <= 0:
            continue
        count = (starts[i]+n)//ratio - starts[i]//ratio
        capacity = min(n, n//ratio + 1)
        local_cu = cu[i:i+2] - cu[i]
        def rotary(table):
            shape = (capacity, table.shape[-1])
            buf = table.new_zeros(shape)
            buf[:count].copy_(table.reshape(-1, shape[-1])[consumed:consumed+count])
            return buf
        part = torch.ops._C_ascend.compressor(
            hidden[lo:hi].contiguous(), module.wkv.weight, module.wgate.weight, state.squeeze(-2),
            module.ape, module.norm.weight, rotary(sin), rotary(cos),
            state_block_table=state_meta.block_table[i:i+1], cu_seqlens=local_cu, seqused=None,
            start_pos=metadata.start_pos[i:i+1], rope_head_dim=module.rope_head_dim,
            cmp_ratio=ratio, coff=module.coff, norm_eps=module.norm_eps, rotary_mode=2, cache_mode=1)
        output[consumed:consumed+count].copy_(part[:count])
        consumed += count
    return output, slots


def configure(roots, label):
    from mindspeed_bridge.models.deepseek_v4_flash.modelling_deepseek_v4_flash.csa import Compressor
    active = label.startswith('actor_paged_combined_shared_fc1_post_compress')
    for root in roots:
        for name, layer in root.named_modules():
            if not isinstance(layer, Compressor) or layer.rotate:
                continue
            record_ape(layer, 'actor', name)
            if not hasattr(layer, '_alignment_compressor_forward'):
                layer._alignment_compressor_forward = layer.forward
            if not active:
                layer.forward = layer._alignment_compressor_forward
                continue
            @torch.no_grad()
            def forward(x, module=layer):
                assert not torch.is_grad_enabled()
                n, batch, width = x.shape
                ratio, dim = module.compress_ratio, module.head_dim
                if n < ratio:
                    return None
                count = n//ratio
                cap = min(n,count+1)
                freq = module.rotary_pos_emb(n, packed_seq=False)
                if isinstance(freq, tuple):
                    freq = freq[0]
                angles = freq.reshape(n,-1)[:count*ratio:ratio,:module.qk_pos_emb_head_dim//2].float().repeat_interleave(2,-1)
                def table(v):
                    return torch.cat([v,v.new_zeros((cap-count,v.shape[-1]))]).contiguous()
                sin, cos = table(angles.sin()), table(angles.cos())
                pages = (n+1)//2+1
                state_table = torch.arange(pages,device=x.device,dtype=torch.int32)[None]
                cu = torch.tensor([0,n],device=x.device,dtype=torch.int32)
                start = torch.zeros(1,device=x.device,dtype=torch.int32)
                outputs = []
                for j in range(batch):
                    state = torch.zeros((pages,2,2*module.coff*dim),device=x.device,dtype=torch.float32)
                    if ratio==4 and dim==512:
                        from verl.utils.debug.bounded_compressor import compress_rows
                        # Block zero is reserved by the native paged state writer.
                        stable_state=torch.zeros((pages+1,2,2*module.coff*dim),device=x.device,dtype=torch.float32)
                        values=compress_rows(x[:,j],module.linear_wkv.weight,module.linear_wgate.weight,
                            stable_state,module.ape.float(),module.norm.weight.float(),sin,cos,state_table+1,
                            0,ratio,module.coff,module.qk_pos_emb_head_dim,module.config.layernorm_epsilon)
                        outputs.append(values)
                        continue
                    values = torch.ops._C_ascend.compressor(
                        x[:,j].contiguous(),module.linear_wkv.weight,module.linear_wgate.weight,state,
                        module.ape.float(),module.norm.weight.float(),sin,cos,state_block_table=state_table,
                        cu_seqlens=cu,seqused=None,start_pos=start,rope_head_dim=module.qk_pos_emb_head_dim,
                        cmp_ratio=ratio,coff=module.coff,norm_eps=module.config.layernorm_epsilon,rotary_mode=2,cache_mode=1)
                    outputs.append(values[:count])
                return torch.stack(outputs,dim=1)
            def train_compress(x, module=layer, serve=forward):
                value=serve(x)
                if value is None or not torch.is_grad_enabled():return value
                return with_local_backward(value,module._alignment_compressor_forward(x))
            layer.forward = train_compress
