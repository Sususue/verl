"""Teacher-force the actual response on the same native engine, preserving rollout scores."""
import json
import os
from pathlib import Path
import traceback

from vllm import SamplingParams


async def capture(server, request, prompt_ids, response, native, sampling, lora, priority, rank):
    folder = Path(os.environ["DSV4_ALIGN_SNAPSHOT"]).parent / "rollout_self"
    folder.mkdir(parents=True, exist_ok=True)
    tokens = list(prompt_ids) + list(response)
    offset = len(prompt_ids)
    row = dict(request=str(request), prompt_ids=list(prompt_ids), response_ids=list(response),
               native=list(native), replica=server.replica_rank, data_parallel_rank=rank,
               pid=os.getpid(), temperature=sampling.temperature, top_p=sampling.top_p,
               top_k=sampling.top_k, min_p=sampling.min_p, logprob_mode=str(getattr(
                   getattr(server.engine, "model_config", None), "logprobs_mode", None)))
    out = folder / (str(os.getpid()) + "-" + __import__("hashlib").sha256(str(request).encode()).hexdigest()[:24] + ".json")

    def save():
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(row, allow_nan=False))
        tmp.replace(out)

    async def score(ids, tag, **options):
        params = SamplingParams(max_tokens=1, temperature=sampling.temperature, top_k=sampling.top_k,
                                top_p=sampling.top_p, min_p=sampling.min_p, detokenize=False, **options)
        last = None
        async for result in server.engine.generate(prompt={"prompt_token_ids": ids}, sampling_params=params,
                request_id=f"{request}-align28-{tag}", lora_request=lora, priority=priority, data_parallel_rank=rank):
            last = result
        assert last is not None and list(last.prompt_token_ids) == ids
        return last

    from verl.utils.debug.focus_trace import mark
    mark(prompt_ids, response)
    save()
    for tag in ("prefill1", "prefill2"):
        try:
            result = await score(tokens, tag, prompt_logprobs=0)
            assert len(result.prompt_logprobs) == len(tokens)
            row[tag] = [float(result.prompt_logprobs[i][tokens[i]].logprob) for i in range(offset, len(tokens))]
            row[tag + "_cached"] = getattr(result, "num_cached_tokens", None)
        except Exception:
            row[tag + "_error"] = traceback.format_exc()[-1600:]
        save()
    # One fresh prefill per causal prefix. This also compares first-token
    # sampling scores with the input-token scoring path above.
    row["prefix"] = []
    for i, label in enumerate(response):
        try:
            result = await score(tokens[:offset+i], f"prefix{i}", logprob_token_ids=[int(label)], logprobs=1)
            row["prefix"].append(float(result.outputs[0].logprobs[0][label].logprob))
        except Exception:
            row["prefix_error"] = traceback.format_exc()[-1600:]
            break
        save()
    row["finished"] = True
    save()
