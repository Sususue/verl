"""Task-scoped BF16 dtype/cache and refit-storage audit; no weight mutation."""
import hashlib
import json
import os
from pathlib import Path
import time

import torch


def audit_refit(worker):
    run = os.environ.get('B24_RUN')
    if not run and os.environ.get('DSV4_B24_AUDIT_DIR'):
        run = str(Path(os.environ['DSV4_B24_AUDIT_DIR']).parent)
    if not run:
        return
    started = time.perf_counter()
    model = worker.model_runner.model
    config = worker.model_runner.vllm_config
    if config.model_config.hf_config.model_type != 'deepseek_v4':
        return
    assert config.quant_config is None, 'BF16 test unexpectedly has a quantization config'
    assert model.lm_head.weight.dtype == torch.float32, 'Expected actual FP32 rollout LM head'
    layers = model.model.layers
    assert len(layers) == 43, f'Expected 43 decoder layers; got {len(layers)}'
    parameters = dict(model.named_parameters())
    expert_metadata = []
    for name, parameter in parameters.items():
        if 'experts' in name and 'weight' in name and parameter.ndim >= 2:
            assert parameter.dtype == torch.bfloat16, (name, str(parameter.dtype))
            expert_metadata.append({'name': name, 'shape': list(parameter.shape), 'dtype': str(parameter.dtype)})
    assert expert_metadata, 'No routed/shared expert weights found for BF16 dtype audit'
    router_caches = []
    schemes = []
    for index, layer in enumerate(layers):
        gate = layer.mlp.gate
        if isinstance(getattr(gate, 'weight_fp32', None), torch.Tensor):
            equal = bool(torch.equal(gate.weight_fp32, gate.weight.float()))
            router_caches.append({'layer': index, 'cache_matches_weight': equal,
                                  'cache_ptr': gate.weight_fp32.data_ptr()})
            assert equal, f'Stale FP32 router cache at layer {index}'
        experts = getattr(layer.mlp, 'experts', None)
        method = getattr(experts, 'quant_method', None)
        schemes.append({'layer': index, 'method': type(method).__name__,
                        'local_experts': getattr(layer.mlp, 'n_local_physical_experts', None)})
    # Reserve slots for both ends of the decoder before adding mHC/shared
    # matrices. A global alphabetical truncation would miss layer42.
    names = []
    def add_matches(predicate, limit=None):
        matches = [name for name in sorted(parameters) if predicate(name) and name not in names]
        names.extend(matches if limit is None else matches[:limit])
    add_matches(lambda name: name.endswith('lm_head.weight'))
    for layer_index in [0, 42]:
        prefix = f'layers.{layer_index}.'
        add_matches(lambda name: prefix in name and name.endswith('mlp.gate.weight'))
        add_matches(lambda name: prefix in name and 'experts' in name and
                    'shared_experts' not in name and 'weight' in name, limit=2)
    add_matches(lambda name: 'layers.0.' in name and name.endswith(('hc_attn_fn', 'hc_ffn_fn')))
    add_matches(lambda name: 'layers.42.' in name and name.endswith(('hc_attn_fn', 'hc_ffn_fn')))
    names = names[:12]
    parameter_storage = {name: parameter.data_ptr() for name, parameter in parameters.items()}
    prior_storage = getattr(worker, '_b24_last_parameter_storage', None)
    moved_all = sorted(name for name in parameter_storage if prior_storage is not None
                       and name in prior_storage and parameter_storage[name] != prior_storage[name])
    samples = {}
    for name in names:
        parameter = parameters[name]
        entry = {'dtype': str(parameter.dtype), 'shape': list(parameter.shape),
                 'data_ptr': parameter.data_ptr(), 'contiguous': parameter.is_contiguous()}
        if parameter.numel() and parameter.is_contiguous():
            flat = parameter.detach().view(-1)
            count = min(8192, flat.numel())
            # Integer arithmetic avoids float32 rounding numel-1 upward for
            # the 529M-element head tensor.
            index = (torch.arange(count, dtype=torch.int64, device='cpu')
                     * (flat.numel()-1) // max(1, count-1)).to(flat.device)
            if 'lm_head.weight' in name and parameter.ndim == 2 and parameter.shape[0] > 2:
                # Include common BOS/EOS rows in addition to a global grid.
                special = torch.arange(min(parameter.shape[1]*2, flat.numel()), device=flat.device)
                index = torch.cat([index, special])
            values = flat.index_select(0, index).float().cpu().numpy()
            entry.update(sample_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                         sample_count=len(values), first_values=values[:8].tolist())
        samples[name] = entry
    previous = getattr(worker, '_b24_last_refit_samples', None)
    changed = []
    moved = []
    if previous is not None:
        for name in samples.keys() & previous.keys():
            if samples[name].get('sample_sha256') != previous[name].get('sample_sha256'):
                changed.append(name)
            if samples[name]['data_ptr'] != previous[name]['data_ptr']:
                moved.append(name)
    index = getattr(worker, '_b24_refit_index', 0)
    record = {'refit_index': index, 'pid': os.getpid(), 'local_rank': worker.local_rank,
              'time': time.time(), 'head_dtype': str(model.lm_head.weight.dtype),
              'quantization': None, 'decoder_layers': len(layers), 'expert_weights': expert_metadata,
              'schemes': schemes, 'router_caches': router_caches, 'sampled_parameters': samples,
              'changed_parameter_samples': sorted(changed), 'moved_parameter_storage': sorted(moved),
              'all_parameter_storage': parameter_storage, 'moved_all_parameter_storage': moved_all,
              'audit_seconds': time.perf_counter()-started}
    worker._b24_last_parameter_storage = parameter_storage
    worker._b24_last_refit_samples = samples
    worker._b24_refit_index = index + 1
    folder = Path(run)/'samples'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/f'refit_{os.getpid()}.jsonl').open('a') as stream:
        stream.write(json.dumps(record)+'\n')
    print('B24_BF16_REFIT_AUDIT '+json.dumps({k: record[k] for k in
          ['refit_index','pid','local_rank','head_dtype','decoder_layers','changed_parameter_samples','moved_parameter_storage']}), flush=True)
