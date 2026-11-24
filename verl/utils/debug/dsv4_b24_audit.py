"""Task-scoped chosen-token audit. Original verl metrics remain authoritative."""
import json
import os
from pathlib import Path
import time

import numpy as np


def audit_batch(data, metrics, step):
    directory = os.environ.get('DSV4_B24_AUDIT_DIR')
    if not directory:
        return
    started = time.perf_counter()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name in ['old_log_probs', 'rollout_log_probs', 'responses', 'response_mask', 'prompts']:
        value = data.batch.get(name)
        if value is not None:
            if 'log_probs' in name:
                value = value.float()
            arrays[name] = value.detach().cpu().numpy()
    actor = arrays['old_log_probs'].astype(np.float64)
    rollout = arrays['rollout_log_probs'].astype(np.float64)
    mask = arrays['response_mask'].astype(bool)
    assert actor.shape == rollout.shape == mask.shape
    finite = bool(np.isfinite(actor[mask]).all() and np.isfinite(rollout[mask]).all())
    pa, pr = np.exp(actor[mask]), np.exp(rollout[mask])
    valid = len(pa) > 1 and finite and np.std(pa) > 0 and np.std(pr) > 0
    corr = float(np.corrcoef(pa, pr)[0, 1]) if valid else None
    delta = np.abs(pa - pr)
    native_diff = float(metrics.get('training/rollout_probs_diff_mean', float('nan')))
    native_corr = float(metrics.get('training/rollout_actor_probs_pearson_corr', float('nan')))
    passed = bool(valid and np.isfinite(native_diff) and np.isfinite(native_corr)
                  and metrics.get('training/rollout_probs_diff_valid') == 1
                  and native_diff < 0.005 and native_corr > 0.999)
    record = {
        'global_step': int(step), 'policy_version_expected': int(step) - 1,
        'samples': int(mask.shape[0]), 'valid_tokens': int(mask.sum()), 'finite': finite,
        'response_lengths': mask.sum(axis=1).tolist(),
        'native_diff_mean': native_diff, 'native_pearson': native_corr,
        'audit_fp64_diff_mean': float(delta.mean()) if len(delta) else None,
        'audit_fp64_pearson': corr,
        'probability_diff_p95': float(np.quantile(delta, .95)) if len(delta) else None,
        'probability_diff_p99': float(np.quantile(delta, .99)) if len(delta) else None,
        'probability_diff_max': float(delta.max()) if len(delta) else None,
        'logprob_mae': float(np.abs(actor[mask] - rollout[mask]).mean()) if len(delta) else None,
        'pass': passed, 'token_ranges': {},
    }
    for lo, hi in [(0, 512), (512, 1024), (1024, 2048), (2048, 3072), (3072, 4096)]:
        m = mask[:, lo:hi]
        a, r = actor[:, lo:hi][m], rollout[:, lo:hi][m]
        if not len(a):
            continue
        p, q = np.exp(a), np.exp(r)
        record['token_ranges'][f'{lo+1}-{hi}'] = {
            'tokens': int(len(a)), 'diff_mean': float(np.abs(p-q).mean()),
            'pearson': float(np.corrcoef(p,q)[0,1]) if len(a)>1 and np.std(p)>0 and np.std(q)>0 else None,
        }
    if 'prompts' in arrays:
        # Prompts may be padded with pad==eos; exact input lengths are also
        # independently checked from the known prompt token manifest.
        record['prompt_padded_width'] = int(arrays['prompts'].shape[1])
    np.savez_compressed(directory/f'step_{int(step):04d}_chosen_token.npz', **arrays)
    record['audit_seconds'] = time.perf_counter() - started
    (directory/f'step_{int(step):04d}_audit.json').write_text(json.dumps(record, indent=2))
    metrics['audit/valid_response_tokens'] = int(mask.sum())
    metrics['timing_s/chosen_token_audit'] = record['audit_seconds']
    metrics['audit/passed'] = int(passed)
    print('B24_CONSISTENCY_AUDIT ' + json.dumps({k: record[k] for k in ['global_step','valid_tokens','native_diff_mean','native_pearson','pass']}), flush=True)
    if not passed and os.environ.get('DSV4_B24_FAIL_ON_MISMATCH', '1') == '1':
        raise RuntimeError('B24 consistency acceptance failed; chosen-token evidence saved to '+str(directory))


def audit_optimizer_update(metrics, step):
    start = os.environ.get('DSV4_B24_EXACT_FROM_POLICY')
    directory = os.environ.get('DSV4_B24_AUDIT_DIR')
    if start is None or not directory:
        return
    required = int(step) <= int(start)
    norm = float(metrics.get('actor/grad_norm', float('nan')))
    passed = bool(np.isfinite(norm) and (norm > 0 if required else True))
    record = {'global_step': int(step), 'grad_norm': norm,
              'nonzero_update_required': required, 'pass': passed}
    (Path(directory)/f'step_{int(step):04d}_update_gate.json').write_text(json.dumps(record, indent=2))
    print('B24_UPDATE_GATE ' + json.dumps(record), flush=True)
    if not passed:
        raise RuntimeError('Natural-EOS update prelude has no finite nonzero gradient; stop before claiming updated-weight full-length verification')
