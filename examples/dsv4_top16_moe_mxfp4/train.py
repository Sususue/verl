"""Apply the reference process settings and launch the normal verl entry point."""

import ctypes
from importlib.util import find_spec
import os
from pathlib import Path
import resource
import runpy


def main():
    # PR_SET_THP_DISABLE matches the process setting used by the long-running job.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(41, 1, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))

    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft_limit < 65536 and (hard_limit == resource.RLIM_INFINITY or hard_limit >= 65536):
        resource.setrlimit(resource.RLIMIT_NOFILE, (65536, hard_limit))

    verl_root = Path(__file__).resolve().parents[2]
    source_root = verl_root.parent
    expected_roots = {
        "verl": verl_root,
        "mindspeed_bridge": source_root / "MindSpeed-Bridge",
        "vllm_ascend": source_root / "vllm-ascend",
    }
    for name, root in expected_roots.items():
        spec = find_spec(name)
        origin = spec.origin if spec else None
        if not origin or not Path(origin).resolve().is_relative_to(root.resolve()):
            raise RuntimeError(f"{name} resolved to {origin}, expected the checkout under {root}")

    runpy.run_module("verl.trainer.main_ppo", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
