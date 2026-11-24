"""Apply the same short-row compressor projection to long prefills and Actor sequences."""
import torch


def compress_rows(x, wkv, wgate, state, ape, norm, sin, cos, table, start, ratio, coff, rope, eps):
    pieces = []
    for lo in range(0, len(x), 32):
        hi = min(lo + 32, len(x))
        first = (start + lo) // ratio - start // ratio
        last = (start + hi) // ratio - start // ratio
        count = last - first
        capacity = min(hi - lo, (hi - lo) // ratio + 1)
        def rotary(src):
            dst = src.new_zeros((capacity, src.shape[-1]))
            dst[:count].copy_(src[first:last])
            return dst
        cu = torch.tensor([0, hi-lo], device=x.device, dtype=torch.int32)
        pos = torch.tensor([start+lo], device=x.device, dtype=torch.int32)
        y = torch.ops._C_ascend.compressor(
            x[lo:hi].contiguous(), wkv, wgate, state, ape, norm, rotary(sin), rotary(cos),
            state_block_table=table, cu_seqlens=cu, seqused=None, start_pos=pos, rope_head_dim=rope,
            cmp_ratio=ratio, coff=coff, norm_eps=eps, rotary_mode=2, cache_mode=1)
        pieces.append(y[:count])
    return torch.cat(pieces)


def serving(module, hidden, state, metadata, state_meta, sin, cos, slots):
    if module.compress_ratio != 4 or module.head_dim != 512 or torch.npu.is_current_stream_capturing():
        return None
    cu = metadata.query_start_loc
    if len(hidden) <= cu.numel()-1:
        return None
    offsets = cu.tolist()
    starts = metadata.start_pos.flatten().tolist()
    if not any(hi-lo > 32 for lo,hi in zip(offsets,offsets[1:])):
        return None
    assert offsets[-1] <= len(hidden)
    out = hidden.new_zeros((len(slots), module.head_dim))
    used = 0
    for i, (lo, hi) in enumerate(zip(offsets,offsets[1:])):
        if lo == hi:
            continue
        count = (starts[i]+hi-lo)//4 - starts[i]//4
        sin_part = sin.reshape(-1, sin.shape[-1])[used:used+count]
        cos_part = cos.reshape(-1, cos.shape[-1])[used:used+count]
        part = compress_rows(hidden[lo:hi],module.wkv.weight,module.wgate.weight,state.squeeze(-2),
            module.ape,module.norm.weight,sin_part,cos_part,state_meta.block_table[i:i+1],starts[i],
            4,module.coff,module.rope_head_dim,module.norm_eps)
        out[used:used+count].copy_(part)
        used += count
    return out, slots
