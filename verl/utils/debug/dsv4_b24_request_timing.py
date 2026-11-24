"""Native request timing and an explicit task-only length-test schedule."""
import json
import math
import os
from pathlib import Path


def apply_length_schedule(params, policy_version):
    start = os.environ.get('DSV4_B24_EXACT_FROM_POLICY')
    if start is None:
        return
    assert os.environ.get('DSV4_B24_AUDIT_DIR'), 'Length schedule requires task audit evidence'
    start = int(start)
    assert start >= 0 and isinstance(policy_version, int) and policy_version >= 0
    # Only stop-on-EOS behaviour changes; logprobs stay raw and the response
    # budget, seed, temperature and all model/optimizer math stay as declared.
    params['ignore_eos'] = policy_version >= start


def record_request_timing(result, policy_version, replica_rank, sampling_params=None):
    run = os.environ.get('B24_RUN')
    if not run and os.environ.get('DSV4_B24_AUDIT_DIR'):
        run = str(Path(os.environ['DSV4_B24_AUDIT_DIR']).parent)
    if not run:
        return
    stats = getattr(result, 'metrics', None)
    names = ['num_generation_tokens', 'num_preemptions', 'arrival_time', 'queued_ts',
             'scheduled_ts', 'first_token_ts', 'last_token_ts', 'first_token_latency',
             'is_corrupted']
    native = {name: getattr(stats, name, None) for name in names} if stats is not None else {}
    valid = bool(stats is not None and all(
        isinstance(native.get(name), (float, int)) and math.isfinite(native[name])
        for name in ['queued_ts', 'scheduled_ts', 'first_token_ts', 'last_token_ts', 'first_token_latency']
    ))
    if valid:
        valid = 0 < native['queued_ts'] <= native['scheduled_ts'] <= native['first_token_ts'] <= native['last_token_ts']
    outputs = getattr(result, 'outputs', [])
    count = sum(len(output.token_ids) for output in outputs)
    record = {
        'request_id': result.request_id, 'policy_version': int(policy_version),
        'replica_rank': int(replica_rank), 'pid': os.getpid(),
        'prompt_tokens': len(getattr(result, 'prompt_token_ids', []) or []),
        'returned_tokens': count, 'num_outputs': len(outputs), 'native': native,
        'timing_valid': valid,
        'sampling': {key: getattr(sampling_params, key, None) for key in ['ignore_eos', 'max_tokens', 'temperature', 'top_p', 'top_k']},
        'ttft_s': native['first_token_latency'] if valid else None,
        'queue_s': native['scheduled_ts']-native['queued_ts'] if valid else None,
        'prefill_s': native['first_token_ts']-native['scheduled_ts'] if valid else None,
        'decode_s': native['last_token_ts']-native['first_token_ts'] if valid else None,
    }
    generated = native.get('num_generation_tokens')
    record['tpot_s'] = record['decode_s']/(generated-1) if valid and generated and generated>1 else None
    folder = Path(run)/'metrics'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/f'request_timings_{os.getpid()}.jsonl').open('a') as stream:
        stream.write(json.dumps(record)+'\n')
