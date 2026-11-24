"""Task-gated first-forward tensor dumps with token identity."""
from pathlib import Path
import os,json,re
import torch


def install_layer_dump(model, side):
    if os.environ.get("DSV4_ALIGN_LAYER_DUMP") != "1" or getattr(model, "_alignment_dump_installed", False):
        return
    from verl.utils.debug.focus_trace import install as install_focus
    install_focus(model, side)
    if torch.distributed.is_initialized() and torch.distributed.get_rank() not in (0,3):
        return
    model._alignment_dump_installed = True
    if side == "rollout":
        os.environ["DSV4_ALIGN_CAPTURE"] = "1"
    directory = Path(os.environ["DSV4_QAT_RUN"]) / "layer_dump" / side / str(os.getpid())
    directory.mkdir(parents=True, exist_ok=True)
    context = {}
    if side == "rollout" and os.environ.get("DSV4_ALIGN_ATTN_CORE") == "1":
        from verl.utils.debug.dsv4_alignment_attention_core import set_rollout_context
        set_rollout_context(context)
    seen = set()
    def tensors(value):
        if isinstance(value, torch.Tensor):
            default_limit = "67108864" if os.environ.get("DSV4_ALIGN_LAYER_DUMP_SCOPE") == "alignment_trace" else "0"
            limit = int(os.environ.get("DSV4_ALIGN_MODULE_DUMP_MAX_BYTES", default_limit))
            if limit and value.numel() * value.element_size() > limit:
                return {"omitted": "large_tensor", "shape": list(value.shape), "dtype": str(value.dtype),
                        "stride": list(value.stride()), "bytes": value.numel() * value.element_size()}
            return value.detach().cpu()
        if isinstance(value, (list, tuple)):
            return [tensors(v) for v in value if isinstance(v, (torch.Tensor, list, tuple, dict))]
        if isinstance(value, dict):
            return {k: tensors(v) for k,v in value.items() if isinstance(v, (torch.Tensor, list, tuple, dict))}
        if hasattr(value, "routed_out"):
            return {"routed_out": tensors(value.routed_out)}
        return None
    def save(name, stage, value):
        phase = "full" if side == "actor" else ("prefill" if context.get("tokens", 0) > 16 else "decode")
        if side == "rollout":
            ids = context.get("token_ids")
            pos = context.get("position_ids")
            if ids is None or pos is None:
                return
            pos = pos.reshape(-1)
            if pos.numel() in (62,124):
                from verl.utils.debug.mixed_rows import trace as trace_mixed
                trace_mixed(directory,seen,name,stage,value,ids,pos)
                return
            if len(pos) not in (57, 65, 114, 130) or not (pos == 0).any():
                return
            phase = "prefill" + str(len(pos))
        label = os.environ.get("DSV4_ALIGN_VARIANT_LABEL", "") if side == "actor" else ""
        if side == 'actor' and label not in ('initial_reference','initial_repeat','initial_inline','actor_hc_tile','actor_projections_shared_route','actor_hc_tile_combined','actor_paged_only','actor_paged_combined','actor_paged_combined_shared_fc1','actor_paged_combined_shared_fc1_post','actor_paged_combined_shared_fc1_post_compress','actor_paged_combined_shared_fc1_post_compress_order'):
            return
        key = (name, stage, phase, label)
        if key in seen:
            return
        seen.add(key)
        stem = re.sub(r"[^a-zA-Z0-9_-]", "_", name) + "_" + stage + "_" + phase
        target = directory / label if label else directory
        target.mkdir(parents=True, exist_ok=True)
        torch.save({"name": name, "stage": stage, "phase": phase,
                    "token_ids": context.get("token_ids"), "position_ids": context.get("position_ids"),
                    "tensor": tensors(value)}, target / (stem + ".pt"))
    def root_pre(module, args, kwargs):
        for key in ["input_ids", "position_ids", "positions"]:
            value = kwargs.get(key)
            if isinstance(value, torch.Tensor):
                context["token_ids" if key == "input_ids" else "position_ids"] = value.detach().cpu()
        if args and isinstance(args[0], torch.Tensor) and not args[0].is_floating_point():
            context["token_ids"] = args[0].detach().cpu()
        if len(args) > 1 and isinstance(args[1], torch.Tensor) and not args[1].is_floating_point():
            context["position_ids"] = args[1].detach().cpu()
    model.register_forward_pre_hook(root_pre, with_kwargs=True)
    count = 0
    for name,module in model.named_modules():
        embedding = name.endswith(("word_embeddings", "embed_tokens"))
        match = re.search(r"layers\.(\d+)(?:\.(.*))?$", name)
        tail = match.group(2) or "" if match else ""
        selected = embedding or (match and tail in (
            "", "input_layernorm", "pre_mlp_layernorm", "post_attention_layernorm",
            "self_attention", "self_attn", "mlp", "mlp.router", "mlp.gate", "mlp.experts",
            "attention_hyper_connection", "mlp_hyper_connection",
        ))
        if os.environ.get("DSV4_ALIGN_LAYER_DUMP_SCOPE") == "l0_internal":
            selected = embedding or (match and int(match.group(1)) == 0 and (
                selected or tail.startswith(("self_attention.", "self_attn."))
            ))
        if os.environ.get("DSV4_ALIGN_LAYER_DUMP_SCOPE") == "alignment_trace":
            selected = selected or (match and int(match.group(1)) in (0, 2, 3, 40, 41, 42) and
                                     tail.startswith(("self_attention.", "self_attn.")))
        if os.environ.get("DSV4_ALIGN_LAYER_DUMP_SCOPE") == "original_trace":
            selected = embedding or (match and (
                (tail == "" and int(match.group(1)) in (0,1,2,3,7,15,21,31,41,42)) or
                (int(match.group(1)) == 0 and
                 (tail in ("input_layernorm","self_attention","self_attn","mlp","mlp.router","mlp.gate") or
                  tail.endswith(("kv_layernorm","q_layernorm","core_attention","wo_a","wo_b"))))))
        if os.environ.get("DSV4_ALIGN_LAYER_DUMP_SCOPE") == "prefix28":
            selected = embedding or (match and (
                (int(match.group(1)) in (0,1,2,3,7,15,31,41,42) and tail == "") or
                (int(match.group(1)) in (0,1,2,3,4,5,6,7) and (tail in ("mlp","mlp.router","mlp.experts","mlp.shared_experts",
                "self_attention","self_attn","input_layernorm","pre_mlp_layernorm","post_attention_layernorm",
                "self_attn.q_norm","self_attn.kv_norm","self_attn.wq_a","self_attn.wkv","self_attn.q_norm_without_weight","self_attn.wo_b","self_attention.q_layernorm","self_attention.kv_layernorm") or
                tail.startswith("mlp.shared_experts.")))))
        if os.environ.get("DSV4_ALIGN_LAYER_DUMP_SCOPE")=="prefix28" and match and int(match.group(1)) in (0,1,2,3,4,5,6,7):
            selected = selected or tail in ("self_attention.linear_q_down_proj","self_attention.linear_kv_proj",
                "self_attention.linear_q_up_proj","self_attention.core_attention","self_attn.wq_b","self_attn.wq_a","self_attn.wkv")
        if match and tail == '' and os.environ.get('DSV4_ALIGN_LAYER_DUMP_SCOPE')=='prefix28':
            selected = True
        if side=='rollout' and match and int(match.group(1))==4:
            selected = True
        if match and int(match.group(1)) in (8,29):
            selected = True
        if match:
            from verl.utils.debug.branch_edges import install as install_edge
            install_edge(module, int(match.group(1)), tail, side, directory, context)
        if not selected:
            continue
        def before(mod,args,kwargs,name=name,embedding=embedding):
            if embedding and args and isinstance(args[0],torch.Tensor):
                context["token_ids"] = args[0].detach().cpu()
                context["tokens"] = args[0].numel()
            save(name,"input",{"args":args,"kwargs":kwargs})
            if hasattr(mod, "attn_sink"):
                save(name, "parameters", {"attn_sink": mod.attn_sink})
        def after(mod,args,kwargs,output,name=name):
            save(name,"output",output)
        if side == "actor" and name.endswith("mlp.router") and hasattr(module, "gating"):
            original_gating = module.gating
            def gating(*args, original=original_gating, name=name, **kwargs):
                save(name + ".gating", "input", {"args":args,"kwargs":kwargs})
                output = original(*args, **kwargs)
                save(name + ".gating", "output", output)
                return output
            module.gating = gating
        module.register_forward_pre_hook(before,with_kwargs=True)
        module.register_forward_hook(after,with_kwargs=True)
        count += 1
    (directory/"manifest.json").write_text(json.dumps({"side":side,"pid":os.getpid(),"hooked_modules":count},indent=2))
