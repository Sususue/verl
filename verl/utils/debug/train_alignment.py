"""Use independently evaluated Actor operators in GRPO and retain their training Jacobians."""
import json
import os
from pathlib import Path
import torch


class LocalOperatorBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, native, differentiable):
        assert native.shape==differentiable.shape and native.dtype==differentiable.dtype
        return native

    @staticmethod
    def backward(ctx, grad):
        return None,grad


def with_local_backward(native, differentiable):
    # The reference is the same local mathematical operator on the same Actor operands.
    # Its graph supplies the existing Bridge VJP when the serving kernel lacks autograd.
    return LocalOperatorBackward.apply(native.detach(),differentiable)


def configure_training(worker,step):
    from verl.utils.debug.dsv4_alignment_variants import configure_mode
    os.environ['DSV4_ALIGNMENT_TRAIN_STEP']=str(step)
    return configure_mode(worker,'head_all')


@torch.no_grad()
def record_loss_scores(current,old,mask):
    delta=(current[mask].double()-old[mask].double()).abs()
    if not delta.numel():return
    event=dict(step=int(os.environ['DSV4_ALIGNMENT_TRAIN_STEP']),tokens=delta.numel(),
               different=int(delta.count_nonzero()),sum_abs=float(delta.sum()),max_abs=float(delta.max()))
    file=Path(os.environ['DSV4_QAT_RUN'])/'audit'/f'loss_logp_rank{torch.distributed.get_rank()}.jsonl'
    with file.open('a') as out:print(json.dumps(event),file=out)
