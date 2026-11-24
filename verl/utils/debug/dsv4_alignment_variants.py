"""Same-policy, same-token diagnostic comparisons using existing worker RPC."""
import json
import os
from pathlib import Path
import numpy as np
import transfer_queue as tq


def configure_mode(worker, mode):
    requested = mode
    tail_modes = ['head_contract', 'head_tile', 'head_sampler', 'head_all', 'head_all_repeat']
    os.environ['DSV4_TAIL_COMPARISON'] = requested if requested in tail_modes else ''
    from verl.utils.debug.head_contract import configure as configure_tail
    configure_tail(requested)
    if requested in tail_modes:
        mode = 'actor_paged_combined_shared_fc1_post_compress_order'
    label = mode
    if mode in ('initial_reference','initial_repeat','actor_bi_l0','actor_bi_all','actor_bi_all_repeat','actor_shared','actor_shared_route','actor_projections_shared_route','actor_projections_shared_route_repeat','actor_hc_tile','actor_hc_tile_combined','actor_hc_tile_combined_repeat','actor_paged_only','actor_paged_combined','actor_paged_combined_repeat','actor_paged_combined_shared_fc1','actor_paged_combined_shared_fc1_repeat','actor_paged_combined_shared_fc1_post','actor_paged_combined_shared_fc1_post_repeat','actor_paged_combined_shared_fc1_post_compress','actor_paged_combined_shared_fc1_post_compress_repeat','actor_paged_combined_shared_fc1_post_compress_order','actor_paged_combined_shared_fc1_post_compress_order_repeat'):
        from verl.utils.debug.combine_trace24 import install as install_combine
        install_combine()
        from verl.utils.debug.original_rope24 import install
        install()
        mode = ('qat_flash_fp32_rope_bf16_router_weight' if label in ('actor_shared_route','actor_projections_shared_route','actor_projections_shared_route_repeat','actor_hc_tile_combined','actor_hc_tile_combined_repeat','actor_paged_combined','actor_paged_combined_repeat','actor_paged_combined_shared_fc1','actor_paged_combined_shared_fc1_repeat','actor_paged_combined_shared_fc1_post','actor_paged_combined_shared_fc1_post_repeat','actor_paged_combined_shared_fc1_post_compress','actor_paged_combined_shared_fc1_post_compress_repeat','actor_paged_combined_shared_fc1_post_compress_order','actor_paged_combined_shared_fc1_post_compress_order_repeat') else 'qat_flash_fp32_rope')
    os.environ["DSV4_ALIGN_VARIANT_LABEL"] = label
    precision_names = ("DSV4_ALIGN_FLASH_SOFTMAX", "DSV4_ALIGN_FP32_ROPE", "DSV4_ALIGN_FP32_ATTN")
    if not hasattr(worker, "_alignment_precision_initial"):
        worker._alignment_precision_initial = {name: os.environ.get(name) for name in precision_names}
    if mode in ("qat_flash_original_rope", "qat_flash_fp32_rope", "qat_flash_fp32_rope_bf16_router_weight"):
        os.environ["DSV4_ALIGN_FLASH_SOFTMAX"] = "1"
        os.environ["DSV4_ALIGN_FP32_ROPE"] = "1" if mode in ("qat_flash_fp32_rope", "qat_flash_fp32_rope_bf16_router_weight") else "0"
        os.environ["DSV4_ALIGN_FP32_ATTN"] = "0"
    elif mode == "qat_fp32_attention_fp32_rope_router_combine":
        os.environ["DSV4_ALIGN_FLASH_SOFTMAX"] = "0"
        os.environ["DSV4_ALIGN_FP32_ROPE"] = "1"
        os.environ["DSV4_ALIGN_FP32_ATTN"] = "1"
    else:
        for name, value in worker._alignment_precision_initial.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
    from mindspeed_bridge.models.deepseek_v4_flash.modelling_deepseek_v4_flash.spec import (
        DSv4PerLayerMoELayer, DSv4QATAlltoAllTokenDispatcher,
    )
    from mindspeed_bridge.models.deepseek_v4_flash.modelling_deepseek_v4_flash.gmm_fused import _qat_record
    modules = worker.actor.engine.module
    if not isinstance(modules, (list, tuple)):
        modules = [modules]
    from verl.utils.debug.actor_projection28 import configure
    from verl.utils.debug.actor_hc_tile28 import configure as configure_hc
    from verl.utils.debug.actor_paged28 import configure as configure_paged
    from verl.utils.debug.shared_fc1 import configure as configure_fc1
    configure_fc1(modules,label)
    from verl.utils.debug.fixed_wo_b import install_actor
    install_actor(modules)
    from verl.utils.debug.text_gate import install_actor as install_text_gate
    install_text_gate(modules)
    from verl.utils.debug.post_serving import configure as configure_post
    configure_post(worker,label)
    from verl.utils.debug.compressor_candidate import configure as configure_compress
    configure_compress(modules,label)
    configure_paged(modules, 'actor_paged_combined' if label.startswith('actor_paged_combined_shared_fc1') else label)
    hc_mode='actor_hc_tile_combined' if label.startswith('actor_paged_combined') else label
    configure_hc(worker,hc_mode)
    hook_mode='actor_projections_shared_route' if label.startswith(('actor_hc_tile_combined','actor_paged_combined')) else label
    configure(modules, hook_mode)
    from verl.utils.debug.actor_shared28 import configure as configure_shared
    configure_shared(modules,hook_mode)
    import torch_npu
    norm_count = 0
    for root in modules:
        for block in root.modules():
            mlp = getattr(block, "mlp", None)
            norm = getattr(block, "pre_mlp_layernorm", None)
            if not isinstance(mlp, DSv4PerLayerMoELayer) or norm is None:
                continue
            router = mlp.router
            router._alignment_fp32_router_enabled = mode in (
                "qat_fp32_router_input", "qat_fp32_router_and_combine", "qat_fp32_attention_fp32_rope_router_combine"
            )
            if not hasattr(block, "_alignment_router_hook_handles"):
                def capture_norm(module, args, output, router=router):
                    if not router._alignment_fp32_router_enabled:
                        return
                    weight = module._gamma() if hasattr(module, "_gamma") else module.weight
                    router._alignment_fp32_input = torch_npu.npu_rms_norm(
                        args[0].float(), weight.float(), epsilon=module.eps
                    )[0]
                def route_fp32(module, args):
                    if not module._alignment_fp32_router_enabled:
                        return
                    value = module._alignment_fp32_input
                    del module._alignment_fp32_input
                    assert value.shape == args[0].shape
                    return (value, *args[1:])
                block._alignment_router_hook_handles = (
                    norm.register_forward_hook(capture_norm),
                    router.register_forward_pre_hook(route_fp32),
                )
            norm_count += 1
    assert norm_count == 43, norm_count
    count = 0
    for root in modules:
        for layer in root.modules():
            if not isinstance(layer, DSv4PerLayerMoELayer):
                continue
            experts = layer.experts
            experts._alignment_router_weight_bf16_enabled = mode in (
                "qat_original_attn_bf16_router_weight", "qat_flash_fp32_rope_bf16_router_weight"
            )
            if not hasattr(experts, "_alignment_router_weight_round_hook"):
                def round_router_weights(module, args, kwargs):
                    if not module._alignment_router_weight_bf16_enabled:
                        return
                    if len(args) >= 3:
                        args = (*args[:2], args[2].bfloat16().float(), *args[3:])
                    else:
                        kwargs = {**kwargs, "permuted_probs": kwargs["permuted_probs"].bfloat16().float()}
                    return args, kwargs
                experts._alignment_router_weight_round_hook = experts.register_forward_pre_hook(
                    round_router_weights, with_kwargs=True
                )
            enabled = mode in ("qat_fp32_attention_fp32_rope_router_combine", "qat_fp32_combine", "qat_fp32_combine_repeat", "qat_fp32_router_and_combine", "qat_flash_original_rope",
                               "qat_flash_fp32_rope", "qat_original_attn_bf16_router_weight",
                               "qat_flash_fp32_rope_bf16_router_weight")
            layer.config.align_qat_fp32_combine = enabled
            layer.experts.config.align_qat_fp32_combine = enabled
            layer.token_dispatcher.__class__ = (
                DSv4QATAlltoAllTokenDispatcher if enabled else MoEAlltoAllTokenDispatcher
            )
            count += 1
    assert count == 43, count
    os.environ["QAT_SCHEME"] = "" if mode == "bf16_actor_vs_w4a8_rollout" else "w4a8-moe-only"
    _qat_record("alignment_variant", mode=mode, layers=count)
    return {"rank": worker.rank, "mode": requested, "layers": count,
            "precision_flags": {name: os.environ.get(name) for name in precision_names},
            "model_training": [root.training for root in modules],
            "config_metadata": {name: getattr(experts.config, name, None) for name in (
                "hidden_dropout", "attention_dropout", "moe_input_jitter_eps",
                "moe_router_force_load_balancing", "moe_router_force_biased",
                "moe_router_enable_expert_bias", "moe_router_bias_update_rate",
                "moe_router_dtype", "deterministic_mode")}}


def compare_variants(trainer, batch):
    from verl.protocol import DataProto
    from verl.trainer.ppo.v1.trainer_base import response_from_nested, calculate_debug_metrics
    root = Path(os.environ["DSV4_QAT_RUN"]) / "variants"
    root.mkdir(exist_ok=True)
    wg = trainer.actor_rollout_wg
    reports = []
    batch.extra_info.update(calculate_entropy=True, compute_loss=False,
                            temperature=trainer.config.actor_rollout_ref.rollout.temperature)
    try:
        supported = ["qat_bf16_combine", "qat_fp32_combine", "qat_fp32_router_input",
                     "qat_fp32_router_and_combine", "bf16_actor_vs_w4a8_rollout"]
        modes = os.environ.get("DSV4_ALIGN_VARIANT_MODES", ",".join(supported)).split(",")
        extra_file = root.parent / "additional_variants.json"
        extra = json.loads(extra_file.read_text()) if extra_file.exists() else []
        modes += [mode for mode in extra if mode not in modes]
        (root / "mode_selection.json").write_text(json.dumps({"modes": modes, "extra_file": str(extra_file)}, indent=2))
        extra_supported = ["qat_fp32_attention_fp32_rope_router_combine", "qat_fp32_combine_repeat", "qat_flash_original_rope", "qat_flash_fp32_rope", "qat_original_attn_bf16_router_weight", "qat_flash_fp32_rope_bf16_router_weight"]
        extra_supported += ['initial_reference','initial_repeat','actor_bi_l0','actor_bi_all','actor_bi_all_repeat','actor_shared','actor_shared_route','actor_projections_shared_route','actor_projections_shared_route_repeat','actor_hc_tile','actor_hc_tile_combined','actor_hc_tile_combined_repeat','actor_paged_only','actor_paged_combined','actor_paged_combined_repeat','actor_paged_combined_shared_fc1','actor_paged_combined_shared_fc1_repeat','actor_paged_combined_shared_fc1_post','actor_paged_combined_shared_fc1_post_repeat','actor_paged_combined_shared_fc1_post_compress','actor_paged_combined_shared_fc1_post_compress_repeat','actor_paged_combined_shared_fc1_post_compress_order','actor_paged_combined_shared_fc1_post_compress_order_repeat']
        extra_supported += ['head_contract', 'head_tile', 'head_sampler', 'head_all', 'head_all_repeat']
        assert modes and all(mode in supported + extra_supported for mode in modes), modes
        for mode in modes:
            worker_evidence = wg.execute_all_sync(
                "actor_rollout_execute_with_func_generator", configure_mode, mode
            )
            wg.compute_log_prob(batch)
            data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id,
                                  select_fields=["log_probs", "rollout_log_probs", "response_mask", "responses"])
            data["old_log_probs"] = response_from_nested(data.pop("log_probs"), data["response_mask"])
            padded = DataProto(batch=data.to_padded_tensor())
            metrics = calculate_debug_metrics(padded)
            arrays = {k: padded.batch[k].float().cpu().numpy()
                      for k in ["old_log_probs", "rollout_log_probs", "response_mask"]}
            np.savez_compressed(root / (mode + ".npz"), **arrays)
            record = {"mode": mode, "metrics": metrics, "workers": worker_evidence,
                      "acceptance_scope": "same current restored-policy tokens; frozen before optimizer"}
            (root / (mode + ".json")).write_text(json.dumps(record, indent=2))
            reports.append(record)
            print("ALIGNMENT_VARIANT", json.dumps(record), flush=True)
    finally:
        wg.execute_all_sync("actor_rollout_execute_with_func_generator", configure_mode, "qat_bf16_combine")
    (root / "summary.json").write_text(json.dumps(reports, indent=2))
    raise RuntimeError("Fixed-token variant comparisons completed before optimizer")
