"""Optional task diagnostics preserving full token and position metadata."""
import os
from pathlib import Path
import torch
import transfer_queue as tq


def prepare_replay(batch, step):
    destination = os.environ.get("DSV4_ALIGN_SNAPSHOT")
    source = os.environ.get("DSV4_ALIGN_REPLAY")
    if not destination and not source:
        return batch
    frozen_cp80 = (int(step) == 81 and destination and not source
                   and os.getenv("DSV4_ALIGN_LONG_POLICY") == "1"
                   and os.getenv("DSV4_ALIGN_VARIANTS") == "1")
    if int(step) != 1 and not frozen_cp80:
        raise RuntimeError("Alignment replay is restricted to the initial policy")
    if source:
        from tensordict._td import TensorDict
        from tensordict.tensorclass import NonTensorData, NonTensorStack
        from tensordict._reductions import _make_td
        from torch.nested._internal.nested_tensor import NestedTensor, _rebuild_njt
        from torch._dynamo.decorators import _DimRange
        with torch.serialization.safe_globals([
            TensorDict, NonTensorData, NonTensorStack, _make_td,
            NestedTensor, _rebuild_njt, _DimRange,
        ]):
            data = torch.load(source, map_location="cpu", weights_only=True)
        assert len(data) == len(batch)
        restored = tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data)
        restored.extra_info.update(batch.extra_info)
        batch = restored
    if destination:
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id)
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, target)
        print("ALIGNMENT_SNAPSHOT", str(target), list(data.keys()), flush=True)
    return batch
