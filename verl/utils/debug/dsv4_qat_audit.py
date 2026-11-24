"""Task-scoped evidence for MoE QAT training and refit."""
import functools
import hashlib
import json
import os
from pathlib import Path
import time

import torch


def append(kind, record):
    run = os.environ.get('DSV4_QAT_RUN')
    if not run:
        return
    folder = Path(run) / 'audit'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f'{kind}_{os.getpid()}.jsonl').open('a') as f:
        f.write(json.dumps({'pid': os.getpid(), 'time': time.time(), **record}, default=str) + '\n')


def sample(tensor, count=2048):
    tensor = tensor.detach()
    if tensor.device.type != 'cpu' and tensor.dtype == torch.uint8:
        assert tensor.numel() * tensor.element_size() <= 512 * 1024**2
        if tensor.ndim == 3 and not tensor.is_contiguous() and tensor.transpose(1, 2).is_contiguous():
            host = tensor.transpose(1, 2).cpu().transpose(1, 2)
        else:
            host = tensor.cpu()
        return sample(host, count)
    count = min(count, tensor.numel())
    index = torch.arange(count, device=tensor.device, dtype=torch.int64)
    index = index * (tensor.numel() - 1) // max(1, count - 1)
    if tensor.is_contiguous():
        values = tensor.view(-1).index_select(0, index)
    else:
        coordinates = []
        remaining = index
        for size in reversed(tensor.shape):
            coordinates.append(remaining.remainder(size))
            remaining = remaining // size
        values = tensor[tuple(reversed(coordinates))]
    return values.float().cpu()


def engine_snapshot(engine):
    chosen = {}
    for i, module in enumerate(engine.module):
        params = dict(module.named_parameters())
        experts = sorted(n for n in params if '.experts.local_experts.' in n and n.endswith('.weight'))
        others = sorted(n for n in params if 'hyper_connection' in n or 'output_layer.weight' in n)
        for name in dict.fromkeys(experts[:2] + experts[-2:] + others[:2]):
            p = params[name]
            chosen[f'{i}:{name}'] = sample(p)
    return chosen


def audit_engine_init(engine):
    from verl.utils.debug.head_trace24 import install_actor
    install_actor(engine)
    if not os.environ.get('DSV4_QAT_RUN'):
        return
    import mindspeed_bridge
    import verl
    from mindspeed_bridge.models.deepseek_v4_flash.modelling_deepseek_v4_flash import gmm_fused as g
    expected = os.environ['DSV4_QAT_EXPECT_SCHEME']
    assert os.environ.get('QAT_SCHEME', '') == expected
    assert 'DSA_SCHEME' not in os.environ, 'DSA_SCHEME must remain unset in this task'
    repo_root = Path(__file__).resolve().parents[3]
    bridge_root = os.environ.get("DSV4_MINDSPEED_BRIDGE_ROOT")
    if bridge_root:
        assert Path(mindspeed_bridge.__file__).resolve().is_relative_to(Path(bridge_root).resolve())
    assert Path(verl.__file__).resolve().is_relative_to(repo_root)
    if os.environ.get("DSV4_ALIGN_LAYER_DUMP") == "1":
        from verl.utils.debug.dsv4_alignment_layer_dump import install_layer_dump
        for chunk in engine.module:
            install_layer_dump(chunk, "actor")
    modules = []
    for chunk in engine.module:
        for name, module in chunk.named_modules():
            if isinstance(module, g.DSv4GroupedSequentialMLP):
                if os.environ.get("DSV4_ALIGN_EXPERT_STAGES") == "1":
                    module._alignment_layer_number = int(name.split("layers.")[-1].split(".")[0])
                modules.append({'name':name,'module_id':id(module),'local_experts':module.num_local_experts,
                                'clamp':module.config.activation_func_clamp_value})
    assert len(modules) == 43, f'Expected 43 routed-expert modules, got {len(modules)}'
    policy_mode = os.environ.get("DSV4_ALIGN_POLICY_MODE")
    if policy_mode:
        allowed = {"qat_fp32_combine", "qat_flash_original_rope", "qat_flash_fp32_rope",
                   "qat_original_attn_bf16_router_weight", "qat_flash_fp32_rope_bf16_router_weight",
                   "qat_fp32_attention_fp32_rope_router_combine"}
        if policy_mode not in allowed:
            raise ValueError(f"Unsupported policy numerical mode: {policy_mode}")
        from types import SimpleNamespace
        from verl.utils.debug.dsv4_alignment_variants import configure_mode
        proxy = SimpleNamespace(actor=SimpleNamespace(engine=engine), rank=torch.distributed.get_rank())
        evidence = configure_mode(proxy, policy_mode)
        append("policy_numerics", {**evidence, "flags": {key: os.environ.get(key) for key in
               ("DSV4_ALIGN_FLASH_SOFTMAX", "DSV4_ALIGN_FP32_ROPE", "DSV4_ALIGN_FP32_ATTN")}})
    append('engine_init', {'rank':torch.distributed.get_rank(), 'bridge_class':type(engine.bridge).__name__,
                          'provider_class':type(engine.provider).__name__, 'modules':modules,
                          'QAT_SCHEME':expected,'DSA_SCHEME':None,'gmm_enabled':g.fused_enabled(),
                          'bridge_path':mindspeed_bridge.__file__,'verl_path':verl.__file__})



def gradient_evidence(engine):
    records = {}
    for chunk_id, chunk in enumerate(engine.module):
        params = dict(chunk.named_parameters())
        names = sorted(n for n in params if '.experts.local_experts.' in n and n.endswith('.weight'))
        for name in dict.fromkeys(names[:2] + names[-2:]):
            param = params[name]
            grad = getattr(param, 'main_grad', None)
            source = 'main_grad'
            if not isinstance(grad, torch.Tensor):
                grad = param.grad
                source = 'grad'
            if isinstance(grad, torch.Tensor):
                v = sample(grad)
                records[f'{chunk_id}:{name}'] = {
                    'source': source, 'dtype': str(grad.dtype), 'device': str(grad.device),
                    'finite': bool(torch.isfinite(v).all()),
                    'sample_abs_max': float(v.abs().max()) if v.numel() else 0.,
                }
            else:
                records[f'{chunk_id}:{name}'] = {'source': None}
    return records


def optimizer_steps(optimizer):
    todo = [optimizer]
    seen = set()
    records = []
    while todo:
        item = todo.pop()
        if item is None or id(item) in seen:
            continue
        seen.add(id(item))
        children = list(getattr(item, 'chained_optimizers', []) or [])
        children += list(getattr(item, 'cpu_optimizers', []) or [])
        sub = getattr(item, 'gpu_optimizer', None)
        if sub is not None:
            children.append(sub)
        try:
            sub = item.optimizer
        except Exception:
            sub = None
        if sub is not None:
            children.append(sub)
        if children:
            todo.extend(children)
            continue
        steps = []
        for state in getattr(item, 'state', {}).values():
            value = state.get('step')
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                steps.append(float(value.item()))
            elif isinstance(value, (float, int)):
                steps.append(float(value))
            if len(steps) == 8:
                break
        records.append({'class': type(item).__name__, 'step_samples': steps})
    return records


def audit_phase(phase):
    def decorate(fn):
        @functools.wraps(fn)
        def wrapped(self, *args, **kwargs):
            if not os.environ.get('DSV4_QAT_RUN'):
                return fn(self, *args, **kwargs)
            before = engine_snapshot(self) if phase == 'optimizer' else None
            gradient = gradient_evidence(self) if phase == 'optimizer' else None
            steps_before = optimizer_steps(self.optimizer) if phase == 'optimizer' else None
            started = time.perf_counter()
            result = fn(self, *args, **kwargs)
            from mindspeed_bridge.models.deepseek_v4_flash.modelling_deepseek_v4_flash import gmm_fused as g
            g.flush_qat_audit()
            record = {'rank':torch.distributed.get_rank(),'phase':phase,
                      'elapsed_s':time.perf_counter()-started,'counts':dict(g._QAT_COUNTS)}
            if before is not None:
                after = engine_snapshot(self)
                record['parameter_delta'] = {n:float((after[n]-v).abs().max()) for n,v in before.items()}
                record['grad_norm'] = float(result)
                record['expert_gradient_samples'] = gradient
                record['optimizer_steps_before'] = steps_before
                record['optimizer_steps_after'] = optimizer_steps(self.optimizer)
            append('engine_phase', record)
            return result
        return wrapped
    return decorate


def audit_refit(worker):
    from verl.utils.debug.long_native_edges import install
    install(worker.model_runner.model)
    from verl.utils.debug.head_trace24 import install_rollout
    install_rollout(worker.model_runner.model)
    model = worker.model_runner.model
    if os.environ.get("DSV4_ALIGN_LAYER_DUMP") == "1":
        from verl.utils.debug.dsv4_alignment_layer_dump import install_layer_dump
        install_layer_dump(model, "rollout")
    from verl.utils.debug.dsv4_gsm8k_audit import resolve_expert_quant_method
    assert model.lm_head.weight.dtype == torch.float32
    schemes=[];router=[]
    for i, layer in enumerate(model.model.layers):
        method=resolve_expert_quant_method(layer.mlp.experts)
        schemes.append({'layer':i,'method':type(method).__name__})
        assert type(method).__name__=='AscendW4A8MXFPDynamicFusedMoEMethod', schemes[-1]
        gate=layer.mlp.gate
        if isinstance(getattr(gate,'weight_fp32',None),torch.Tensor):
            equal=bool(torch.equal(gate.weight_fp32,gate.weight.float()))
            assert equal,f'Stale router cache at layer {i}'
            router.append({'layer':i,'cache_matches':equal})
    params=dict(model.named_parameters());samples={}
    for name,p in params.items():
        if name=='lm_head.weight' or any(f'layers.{i}.' in name for i in [0,42]):
            if any(x in name for x in ['w13_weight','w2_weight','hc_attn_fn','hc_ffn_fn','gate.weight','lm_head.weight']):
                values=sample(p)
                samples[name]={'dtype':str(p.dtype),'shape':list(p.shape),
                               'hash':hashlib.sha256(values.numpy().tobytes()).hexdigest()}
    index=getattr(worker,'_task_qat_refit_index',0)
    append('refit',{'index':index,'head_dtype':str(model.lm_head.weight.dtype),
                    'schemes':schemes,'router':router,'samples':samples})
    worker._task_qat_refit_index=index+1
