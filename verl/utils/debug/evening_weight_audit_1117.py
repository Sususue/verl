"""First-refit, TP=1 DSV4 checkpoint audit. Called on the weight-loading thread."""
import contextlib
import json
import re
import os
from pathlib import Path

import torch
import torch_npu
from safetensors import safe_open

SOURCE = Path(os.environ['DSV4_ALIGNMENT_REFERENCE'])


def inventory(model):
    return {k: [v.data_ptr(), list(v.shape), list(v.stride()), str(v.dtype)]
            for k, v in model.named_parameters()}


def dest_name(name):
    name = 'model.' + name
    for src, dst in (('model.head.', 'lm_head.'), ('model.embed.', 'model.embed_tokens.'),
                     ('.attn.', '.self_attn.'), ('.ffn.', '.mlp.'),
                     ('.ffn_norm.', '.post_attention_layernorm.'), ('.attn_norm.', '.input_layernorm.'),
                     ('.w1.', '.gate_up_proj.'), ('.w3.', '.gate_up_proj.'), ('.w2.', '.down_proj.')):
        name = name.replace(src, dst)
    if name.endswith('.gate.bias'):
        name = name[:-4] + 'e_score_correction_bias'
    return name


def coordinates(flat, shape):
    output = []
    for dim in reversed(shape):
        output.append(flat % dim)
        flat = flat // dim
    return tuple(reversed(output))


def compare_nonexpert(name, src, dst):
    v = dst.detach()
    if name.endswith('.wo_a.weight') and v.ndim == 3:
        v = v.transpose(1, 2)
    if '.shared_experts.' in name and ('.w1.' in name or '.w3.' in name):
        v = v.chunk(2, dim=0)[int('.w3.' in name)]
    n = src.numel()
    if v.numel() != n:
        # vLLM pads the vocabulary of its head and embedding.
        if name in ('head.weight', 'embed.weight') and v.shape[0] >= src.shape[0] and v.shape[1:] == src.shape[1:]:
            v = v[:src.shape[0]]
        else:
            return {'shape_mismatch': [list(src.shape), list(v.shape)]}
    idx = torch.arange(min(17, n), dtype=torch.int64, device="cpu") * (n-1) // max(1, min(17,n)-1)
    a = v[coordinates(idx.to(v.device), v.shape)].cpu()
    b = src[coordinates(idx, src.shape)].to(a.dtype)
    if torch.equal(a,b):
        return None
    return {'observed': a.tolist(), 'expected': b.tolist(), 'max_abs': (a.double()-b.double()).abs().max().item(),
            'matches_bf16_roundtrip': torch.equal(a,b.bfloat16().to(a.dtype))}


@torch.no_grad()
def after(worker, before, receipts):
    from verl.utils.vllm.ascend_dsv4_quant import quantize_expert_matrix
    model = worker.model_runner.model
    rank = model.model.layers[0].mlp.ep_rank
    params = dict(model.named_parameters())
    now = inventory(model)
    folder = Path(os.environ['DSV4_QAT_RUN'])/'audit'
    out = folder / f'checkpoint_rank{rank}.json'
    index = json.loads((SOURCE/'model.safetensors.index.json').read_text())['weight_map']
    report = dict(rank=rank, sources=[], nonexpert_errors={}, expert_errors=[], checked_experts=0,
                  expected_expert_samples=0, parameter_inventory=now, pointer_or_layout_changes={
                      k:[before[k], val] for k,val in now.items() if k in before and before[k]!=val},
                  receipts=sorted(receipts), missing_source_destinations=[], router={}, derived={})
    from vllm_ascend.compilation.acl_graph import _acl_graph_wrappers
    from vllm_ascend.ops.rope_dsv4 import ComplexExpRotaryEmbedding, _ROPE_STATE
    entries = [e for w in list(_acl_graph_wrappers) for e in w.concrete_aclgraph_entries.values()]
    pointers = {p for e in entries for p in (e.input_addresses or [])}
    report['graph_refs'] = dict(entries=len(entries), unique_tensor_addresses=len(pointers),
        live_parameters_seen=sorted(k for k,v in now.items() if v[0] in pointers),
        old_parameter_addresses_seen=sorted(k for k,v in before.items() if v[0] in pointers and now.get(k)!=v))
    report['rope'] = {}
    buffers = {v.data_ptr(): k for k,v in model.named_buffers()}
    for key,(cos,sin) in _ROPE_STATE.full_rope_cache.items():
        fields = re.fullmatch(r'rotary_dim(.+)_max_position_embeddings(.+)_original_seq_len(.+)_apply_yarn(.+)_base(.+)_scaling_factor(.+)_beta_fast(.+)_beta_slow(.+)',key)
        if fields is None:
            report['rope'][key] = dict(unrecognized_key=True)
            continue
        dim,limit,orig,yarn,base,factor,fast,slow = fields.groups()
        with torch.device('cpu'):
            freq = ComplexExpRotaryEmbedding.precompute_freqs_cis(int(dim),int(limit),int(orig),float(base),float(factor),
                       float(fast),float(slow),apply_yarn_scaling=(yarn=='True'))
            positions=torch.tensor([0,1,2,31,min(511,cos.shape[0]-1)],dtype=torch.long)
            phase=(positions.float()[:,None]*freq[None,:]).repeat_interleave(2,-1)
        c=cos.index_select(0,positions.to(cos.device)).view_as(phase).cpu()
        s=sin.index_select(0,positions.to(sin.device)).view_as(phase).cpu()
        report['rope'][key]=dict(cos_error=(c-phase.cos()).abs().max().item(), sin_error=(s-phase.sin()).abs().max().item(),
                                 owner_cos=buffers.get(cos.data_ptr()),owner_sin=buffers.get(sin.data_ptr()))
    covered = set()
    with contextlib.ExitStack() as stack:
        handles = {}
        def get(name):
            shard = index[name]
            if shard not in handles:
                handles[shard] = stack.enter_context(safe_open(str(SOURCE/shard), framework='pt', device='cpu'))
            return handles[shard].get_tensor(name)
        for name in index:
            if name.startswith('mtp.') or '.experts.' in name:
                continue
            target = dest_name(name)
            if target not in params:
                if name.endswith('.gate.bias') and getattr(model.model.layers[int(name.split('.')[1])].mlp.gate, 'tid2eid', None) is not None:
                    continue
                report['missing_source_destinations'].append([name,target])
                continue
            err = compare_nonexpert(name, get(name), params[target])
            if err:
                report['nonexpert_errors'][name] = err
            covered.add(target)
            report['sources'].append(name)
        for li, layer in enumerate(model.model.layers):
            gate = layer.mlp.gate
            if getattr(gate,'weight_fp32',None) is not None:
                report['router'][li] = torch.equal(gate.weight_fp32, gate.weight.float())
                covered.add(f'model.layers.{li}.mlp.gate.weight_fp32')
            mod = layer.mlp.experts.routed_experts
            prefix = f'model.layers.{li}.mlp.experts.routed_experts.'
            nlocal = layer.mlp.n_local_physical_experts
            start = layer.mlp.physical_expert_start
            runtime_map = getattr(mod,'ascend_expert_map',None)
            if isinstance(runtime_map,torch.Tensor):
                observed = runtime_map.detach().cpu().to(torch.int64)
                expected_map = torch.full((layer.mlp.n_routed_experts,),-1,dtype=torch.int64,device='cpu')
                expected_map[start:start+nlocal] = torch.arange(nlocal,device='cpu')
                report.setdefault('expert_maps',{})[li] = dict(first=start,count=nlocal,shape=list(observed.shape),
                    matches_linear=torch.equal(observed,expected_map),actual=observed.tolist())
            else:
                report.setdefault('expert_maps',{})[li] = dict(first=start,count=nlocal,map_is_none=True)
            for stem, projs in [('w13_weight', ('w1','w3')), ('w2_weight', ('w2',))]:
                packed = torch_npu.npu_format_cast(getattr(mod,stem).detach().transpose(1,2), 2).view(torch.uint8)
                scales = getattr(mod,stem+'_scale').detach().transpose(-3,-2).reshape(mod._mxfp4_original_shapes[stem+'_scale']).view(torch.uint8)
                covered.update((prefix+stem, prefix+stem+'_scale'))
                src_parts, packed_parts, scale_parts, keys = [], [], [], []
                for local in range(nlocal):
                    for half, proj in enumerate(projs):
                        name = f'layers.{li}.ffn.experts.{start+local}.{proj}.weight'
                        src = get(name)
                        rows, cols = src.shape
                        assert packed.shape == (nlocal, rows*len(projs), cols//2), (name, packed.shape, src.shape)
                        for row, block in [(0,0),(rows//2,(cols//128)*64),(rows-1,cols-64)]:
                            src_parts.append(src[row,block:block+64])
                            packed_parts.append(packed[local,rows*half+row,block//2:block//2+32])
                            scale_parts.append(scales[local,rows*half+row,block//32:block//32+2])
                            keys.append(name)
                        report['checked_experts'] += 1
                ref_p, ref_s = quantize_expert_matrix(torch.stack(src_parts).to(device=packed.device,dtype=torch.bfloat16))
                bad_p = torch.stack(packed_parts).ne(ref_p).any(-1).cpu().tolist()
                bad_s = torch.stack(scale_parts).ne(ref_s).any(-1).cpu().tolist()
                report['expert_errors'].extend([k,i%3,p,s] for i,(k,p,s) in enumerate(zip(keys,bad_p,bad_s)) if p or s)
                report['expected_expert_samples'] += len(keys)
                del packed, scales, packed_parts, scale_parts
            if li % 8 == 0:
                print(f'DSV4_CHECKPOINT_AUDIT rank={rank} finished_layer={li}', flush=True)
    # Known weight-derived aliases consumed by the router include registered parameters
    # already compared above. Inspect other actual tensor aliases by storage identity.
    all_ptrs = {v.data_ptr():k for k,v in params.items()}
    for module_name, module in model.named_modules():
        if 'router' in module_name or 'routed_experts' in module_name:
            for attr in ('e_score_correction_bias','hash_indices_table','bias_vl'):
                val = getattr(module,attr,None)
                if isinstance(val, torch.Tensor):
                    report['derived'][module_name+'.'+attr] = all_ptrs.get(val.data_ptr())
    report['uncovered_parameters'] = sorted(params.keys()-covered)
    report['without_load_receipt'] = sorted(covered-set(receipts)-{k for k in covered if k.endswith('weight_fp32')})
    tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(report))
    tmp.replace(out)
    print('DSV4_CHECKPOINT_AUDIT_DONE', rank, len(report['nonexpert_errors']), len(report['expert_errors']), flush=True)
