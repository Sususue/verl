"""Opt-in GSM8K evaluation evidence; no sampling/weight arithmetic changes."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def append(name, record):
    folder = Path(os.environ['DSV4_GSM8K_RUN_DIR']) / 'audit'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f'{name}_{os.getpid()}.jsonl').open('a') as handle:
        handle.write(json.dumps({'time': time.time(), 'pid': os.getpid(), **record},
                                ensure_ascii=False, default=str) + '\n')


def setup_process():
    root = Path(os.environ['DSV4_GSM8K_ROOT'])
    cache = root / 'c/extensions' / str(os.getpid())
    cache.mkdir(parents=True, exist_ok=True)
    marker=cache/'.task_seed_ready'
    seed=root/'c/extension_seed'
    if not marker.exists():
        if seed.is_dir():
            shutil.copytree(seed,cache,dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns('lock','*.lock','*_lock','.nfs*'))
        marker.touch()
    os.environ['TORCH_EXTENSIONS_DIR'] = str(cache)


def strict_determinism(role, seed=None):
    import torch
    setup_process()
    torch.use_deterministic_algorithms(True, warn_only=False)
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert os.environ.get('HCCL_DETERMINISTIC', '').lower() in ('true', 'strict')
    append('determinism', {'role': role, 'rank': os.environ.get('RANK'),
                          'local_rank': os.environ.get('LOCAL_RANK'), 'seed': seed,
                          'deterministic': True, 'warn_only': False,
                          'hccl': os.environ.get('HCCL_DETERMINISTIC'),
                          'batch_invariant': os.environ.get('VLLM_BATCH_INVARIANT'),
                          'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
                          'extension_cache': os.environ['TORCH_EXTENSIONS_DIR']})


def record_native(final_res, sampling_params, replica_rank, priority):
    assert sampling_params.ignore_eos is False, 'ignore_eos must be False'
    assert sampling_params.min_tokens == 0, 'min_tokens must be zero'
    assert sampling_params.temperature == 0.0, 'greedy required'
    assert sampling_params.max_tokens == 512, 'response limit changed'
    assert len(final_res.outputs) == 1
    output = final_res.outputs[0]
    prompt = list(final_res.prompt_token_ids)
    record = {'sample_id': f'gsm8k-train-{int(priority):05d}',
              'request_id': final_res.request_id, 'priority': int(priority),
              'replica_rank': replica_rank, 'dp_rank': int(priority) % 8,
              'prompt_token_ids': prompt, 'response_token_ids': list(output.token_ids),
              'prompt_hash': hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
              'finish_reason': output.finish_reason, 'stop_reason': output.stop_reason,
              'num_preempted': getattr(final_res, 'num_preempted', None),
              'sampling': {key: getattr(sampling_params,key,None) for key in
                           ['ignore_eos','min_tokens','max_tokens','temperature','top_p','top_k',
                            'seed','stop','stop_token_ids','repetition_penalty',
                            'presence_penalty','frequency_penalty']}}
    append('native', record)


def record_validation_batch(keys, data):
    rewards = data['rm_scores'].sum(dim=1).tolist()
    uid_values=data['uid']
    target_values=data['reward_model']
    uids=uid_values.tolist() if hasattr(uid_values,'tolist') else list(uid_values)
    ground_truth=target_values.tolist() if hasattr(target_values,'tolist') else list(target_values)
    assert len(keys)==len(rewards)==len(uids)==len(ground_truth),'Validation metadata alignment mismatch'
    for key, uid, reward, target in zip(keys, uids, rewards, ground_truth, strict=True):
        append('scores', {'key': key, 'sample_id': str(uid), 'reward': reward,
                          'ground_truth': target.get('ground_truth')})
    append('progress', {'completed_batch': len(uids), 'batch_correct': sum(rewards)})


def resolve_expert_quant_method(runner):
    routed=getattr(runner,'routed_experts',runner)
    method=getattr(routed,'quant_method',None)
    if method is None:
        method=getattr(runner,'_quant_method',None)
    if method is None:
        raise AssertionError(f'No quant method on actual routed expert module: {type(routed).__name__}')
    return getattr(method,'quant_method',method)


def source_values_equal(actual,expected,name,index_table=False):
    import torch
    if index_table:
        assert actual.dtype in (torch.int32,torch.int64) and expected.dtype in (torch.int32,torch.int64)
        info=torch.iinfo(actual.dtype)
        assert int(expected.min())>=info.min and int(expected.max())<=info.max,'index conversion overflow'
        return bool(torch.equal(actual.to(torch.int64),expected.to(torch.int64)))
    if name=='lm_head.weight' and actual.dtype==torch.float32 and expected.dtype in (torch.bfloat16,torch.float16):
        return bool(torch.equal(actual,expected.to(torch.float32)))
    return actual.dtype==expected.dtype and bool(torch.equal(actual,expected))


def packed_tensor_sha256(tensor):
    """Read logical packed bytes without cloning an NPU internal-format view."""
    import torch
    value=tensor.detach()
    assert value.numel()*value.element_size()<=512*1024**2,'Packed parameter exceeds bounded host staging budget'
    # The installed W4A8 post-load transform transposes NZ weights on device.
    # Restore that view before D2H; perform contiguity conversion only on CPU.
    if value.ndim==3 and not value.is_contiguous() and value.transpose(1,2).is_contiguous():
        host=value.transpose(1,2).cpu().transpose(1,2).contiguous()
    else:
        host=value.cpu().contiguous()
    buffer=memoryview(host.reshape(-1).view(torch.uint8).numpy())
    digest=hashlib.sha256()
    for start in range(0,len(buffer),1048576):digest.update(buffer[start:start+1048576])
    return digest.hexdigest()


def audit_loaded_model(worker):
    import torch
    from safetensors import safe_open
    from verl.utils.megatron.dsv4_source_precision import _source_name

    started=time.perf_counter()
    model=worker.model_runner.model
    config=worker.model_runner.vllm_config
    is_quant=config.quant_config is not None
    assert model.lm_head.weight.dtype==torch.float32
    layers=model.model.layers
    assert len(layers)==43
    model.eval()
    caches=[]
    schemes=[]
    for i,layer in enumerate(layers):
        gate=layer.mlp.gate
        if hasattr(gate,'weight_fp32'):
            equal=bool(torch.equal(gate.weight_fp32,gate.weight.float()))
            assert equal,f'Stale router cache at layer {i}'
            caches.append({'layer':i,'equal':equal})
        inner=resolve_expert_quant_method(layer.mlp.experts)
        schemes.append(type(inner).__name__)
    parameters=dict(model.named_parameters())
    experts={name:p for name,p in parameters.items() if '.experts.' in name and 'weight' in name}
    assert experts
    if not is_quant:
        assert all(p.dtype==torch.bfloat16 for p in experts.values())
    else:
        assert all('W4A8' in name for name in schemes),schemes

    # Compare small logical samples from every layer; this is explicitly a sample audit.
    source=Path('/mnt/share/s00956100/dsv4_mxfp4/prune/out/top16')
    index=json.loads((source/'model.safetensors.index.json').read_text())['weight_map']
    comparisons=[]
    source_mismatches=[]
    for name,p in parameters.items():
        if not (name.endswith('lm_head.weight') or '.mlp.gate.' in name or '.hc_' in name):
            continue
        key=_source_name(name)
        if key not in index:continue
        with safe_open(str(source/index[key]),framework='pt',device='cpu') as handle:
            part=handle.get_slice(key)
            if list(p.shape)!=part.get_shape():
                raise AssertionError((name,list(p.shape),part.get_shape()))
            full_index_table=key.endswith('.tid2eid')
            expected=part[:] if p.ndim==0 or full_index_table else part[:1]
        actual=p.detach() if p.ndim==0 or full_index_table else p.detach()[:1]
        actual=actual.reshape(-1).cpu()
        expected=expected.reshape(-1)
        if not full_index_table:
            actual=actual[:256];expected=expected[:256]
        equal=source_values_equal(actual,expected,name,index_table=full_index_table)
        if not equal:source_mismatches.append({'name':name,'runtime_dtype':str(actual.dtype),'source_dtype':str(expected.dtype)})
        comparisons.append({'name':name,'dtype':str(p.dtype),'source_dtype':str(expected.dtype),
                            'comparison_scope':'full_integer_index_table' if full_index_table else 'first_row_up_to256_values',
                            'sample_count':actual.numel(),
                            'equal':equal,'sha256':hashlib.sha256(actual.view(torch.uint8).numpy().tobytes()).hexdigest()})
    hashes={}
    if is_quant:
        # Hash all local packed expert parameters and scales with bounded staging buffers.
        for name,p in experts.items():
            hashes[name]={'dtype':str(p.dtype),'shape':list(p.shape),'sha256':packed_tensor_sha256(p)}
    append('loaded_model',{'local_rank':worker.local_rank,'quantized':is_quant,
                           'head_dtype':str(model.lm_head.weight.dtype),'schemes':schemes,
                           'router_caches':caches,'non_expert_samples':comparisons,
                           'expert_metadata':{name:{'dtype':str(p.dtype),'shape':list(p.shape)} for name,p in experts.items()},
                           'packed_expert_hashes':hashes,'source_mismatches':source_mismatches,
                           'audit_seconds':time.perf_counter()-started})
    assert not source_mismatches,f'Non-expert source samples differ: {source_mismatches}'
