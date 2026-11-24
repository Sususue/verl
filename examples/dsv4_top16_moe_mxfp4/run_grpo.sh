#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'HELP'
Run the DSV4 Flash top16 MoE MXFP4 single-node eager GRPO recipe.

Required environment variables (absolute paths)
  MODEL_PATH   Top16 model directory with quant_model_description.json
  TRAIN_FILE   Training parquet
  VAL_FILE     Fixed development parquet
  OUTPUT_DIR   New directory for logs, artifacts and checkpoints
  NET_IFACE    HCCL and Gloo network interface

Optional environment variables
  RESUME_FROM  Complete global_step_N checkpoint directory
  TOTAL_STEPS  Cumulative target (default 2 from original weights or checkpoint step + 10)
  PYTHON_BIN   Interpreter in the pinned environment (default python3)

Usage: run_grpo.sh [--dry-run]
HELP
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    --dry-run) dry_run=1; shift ;;
    '') dry_run=0 ;;
    *) usage >&2; exit 2 ;;
esac
[[ $# -eq 0 ]] || { usage >&2; exit 2; }

: "${MODEL_PATH:?Set the path to the top16 online MXFP4 model directory}"
: "${TRAIN_FILE:?Set the training parquet path}"
: "${VAL_FILE:?Set the fixed development parquet path}"
: "${OUTPUT_DIR:?Set a fresh output directory on local NVMe}"
: "${NET_IFACE:?Set the HCCL and Gloo network interface}"

for path in "$MODEL_PATH" "$TRAIN_FILE" "$VAL_FILE" "$OUTPUT_DIR"; do
    [[ "$path" = /* ]] || { printf 'Expected an absolute path: %s\n' "$path" >&2; exit 2; }
done
[[ -f "$MODEL_PATH/config.json" && -f "$MODEL_PATH/model.safetensors.index.json" &&
   -f "$MODEL_PATH/quant_model_description.json" ]] || {
    printf 'Missing top16 model metadata or the online quantization description.\n' >&2
    exit 2
}
[[ -r "$TRAIN_FILE" && -f "$TRAIN_FILE" && -r "$VAL_FILE" && -f "$VAL_FILE" ]] || {
    printf 'Both parquet files must exist and be readable.\n' >&2
    exit 2
}

hydra_string() {
    local value=${1//\\/\\\\}
    value=${value//\"/\\\"}
    printf '"%s"' "$value"
}

resume_mode=disable
resume_path=null
resume_step=0
test_freq=20
reference_model=$MODEL_PATH
if [[ -n "${RESUME_FROM:-}" ]]; then
    RESUME_FROM=${RESUME_FROM%/}
    if [[ "$RESUME_FROM" != /* || ! "${RESUME_FROM##*/}" =~ ^global_step_[0-9]+$ ||
          ! -f "$RESUME_FROM/checkpoint_complete.json" || ! -f "$RESUME_FROM/data.pt" ]]; then
        printf 'RESUME_FROM must point to a complete global_step_N checkpoint.\n' >&2
        exit 2
    fi
    resume_mode=resume_path
    resume_step=${RESUME_FROM##*/global_step_}
    resume_step=$((10#$resume_step))
    resume_path=$(hydra_string "$RESUME_FROM")
    reference_model="$RESUME_FROM/actor/model/huggingface"
    [[ -f "$reference_model/model.safetensors.index.json" && -f "$RESUME_FROM/actor/ckpt_contents.json" ]] || {
        printf 'The resume checkpoint must contain the actor model and checkpoint manifest.\n' >&2
        exit 2
    }
    test_freq=-1
fi

default_steps=2
if [[ "$resume_mode" = resume_path ]]; then
    default_steps=$((resume_step + 10))
    if (( default_steps > 100 )); then default_steps=100; fi
fi
TOTAL_STEPS=${TOTAL_STEPS:-$default_steps}
if [[ ! "$TOTAL_STEPS" =~ ^[1-9][0-9]{0,2}$ ]] || (( TOTAL_STEPS <= resume_step || TOTAL_STEPS > 100 )); then
    printf 'TOTAL_STEPS must be greater than the starting step and at most 100.\n' >&2
    exit 2
fi

recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_dir=$(cd -- "$recipe_dir/../.." && pwd -P)
python_bin=${PYTHON_BIN:-python3}
OUTPUT_DIR=${OUTPUT_DIR%/}
if [[ "$dry_run" -eq 1 ]]; then
    ray_tmp=/tmp/dsv4-mxfp4-dry-run
    export DSV4_CONFIG_ONLY=1
else
    [[ -d "/sys/class/net/$NET_IFACE" ]] || { printf 'Network interface is unavailable: %s\n' "$NET_IFACE" >&2; exit 2; }
    unset DSV4_CONFIG_ONLY
    mkdir -p -- "$(dirname -- "$OUTPUT_DIR")"
    mkdir -- "$OUTPUT_DIR"
    ray_link_dir=$(mktemp -d /tmp/dsv4-ray.XXXXXX)
    ray_tmp="$ray_link_dir/run"
    trap 'if [[ -L "$ray_tmp" ]]; then unlink "$ray_tmp"; fi; rmdir "$ray_link_dir"' EXIT
    mkdir -p -- "$OUTPUT_DIR"/{audit,metrics,monitor,reports,samples,ray/tmp}
    ln -s -- "$OUTPUT_DIR/ray" "$ray_tmp"
fi

unset DSA_SCHEME NON_MEGATRON DSV4_LP_DEBUG
source_root=$(dirname -- "$repo_dir")
export DSV4_MINDSPEED_BRIDGE_ROOT="$source_root/MindSpeed-Bridge"
export PYTHONPATH="$source_root/vllm-ascend:$recipe_dir:$repo_dir:$DSV4_MINDSPEED_BRIDGE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DSV4_RESUMABLE=1
export DSV4_LP_RECOMPUTE=0
export DSV4_CONSISTENCY_DIR="$OUTPUT_DIR/outputs/consistency"
export DSV4_TRACE_DIR=''
export VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY=DSV4_
export DSV4_XFER_NO_RECV_EXPERT_FILTER=1
export DSV4_XFER_SENDER_EXPERT_FILTER=0
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS=1
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT=600
export TQ_NUM_THREADS=1
export VERL_USE_MEGATRON_ADAPTOR=enabled
export VLLM_DISABLE_COMPILE_CACHE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export KIMI_TRAIN_USE_PARTITION_INVARIANT_KDA_FORWARD=0
export KIMI_FUSED_KDA_BACKEND=ascend_triton
export KIMI_VLLM_USE_PARTITION_INVARIANT_KDA=0
export KIMI_VLLM_USE_TRAINING_CAUSAL_CONV1D=0
export KIMI_KDA_OPROJ_FP32_REDUCE=0
export VERL_FAIL_ON_NONFINITE_GRAD=1
export KIMI_RUN_TOKEN
KIMI_RUN_TOKEN=$("$python_bin" -S -c 'import uuid; print(uuid.uuid4())')
export DSV4_FORCE_MOE_ALLTOALL=0
export DBG_XFER=0
export DSV4_LAYER_TRACE=0
export DSV4_LAYER_DUMP=0
export PROMPTS_ONLY=0
export ASCEND_LAUNCH_BLOCKING=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export DSV4_MOE_TRACE=0
export DSV4_A2A_PAD=0
export DSV4_A2A_LIST=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export VLLM_RANDOMIZE_DP_DUMMY_INPUTS=0
export HCCL_SOCKET_IFNAME="$NET_IFACE"
export GLOO_SOCKET_IFNAME="$NET_IFACE"
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export HCCL_NPU_SOCKET_PORT_RANGE=auto
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=17340
export DSV4_B24_RELEASE_CACHE_BEFORE_FIRST_OPT=1
export VLLM_LOGGING_LEVEL=INFO
export RAY_DEDUP_LOGS=0
export PYTHONPYCACHEPREFIX="$OUTPUT_DIR/c/pycache"
export DSV4_B24_PROCESS_CACHE=1
export DSV4_B24_AUDIT_DIR="$OUTPUT_DIR/samples"
export DSV4_B24_FAIL_ON_MISMATCH=0
export TMPDIR="$ray_tmp/tmp"
export TMP="$ray_tmp/tmp"
export TEMP="$ray_tmp/tmp"
export XDG_CACHE_HOME="$OUTPUT_DIR/c/xdg"
export HF_HOME="$OUTPUT_DIR/c/hf"
export TORCH_HOME="$OUTPUT_DIR/c/torch"
export TORCH_EXTENSIONS_DIR="$OUTPUT_DIR/c/extensions"
export TORCHINDUCTOR_CACHE_DIR="$OUTPUT_DIR/c/inductor"
export TRITON_CACHE_DIR="$OUTPUT_DIR/c/triton"
export VLLM_CACHE_ROOT="$OUTPUT_DIR/c/vllm"
export VLLM_CONFIG_ROOT="$OUTPUT_DIR/c/vllm_config"
export ASCEND_CACHE_PATH="$OUTPUT_DIR/c/ascend"
export ASCEND_WORK_PATH="$OUTPUT_DIR/c/ascend_work"
export B24_ROOT="$OUTPUT_DIR"
export B24_RUN="$OUTPUT_DIR"
export VERL_FILE_LOGGER_ROOT="$OUTPUT_DIR/metrics"
export VERL_FILE_LOGGER_PATH="$OUTPUT_DIR/metrics/steps.jsonl"
export ASCEND_PROCESS_LOG_PATH="$OUTPUT_DIR/monitor/ascend"
export DSV4_QAT_ROOT="$OUTPUT_DIR"
export DSV4_QAT_RUN="$OUTPUT_DIR"
export DSV4_QAT_EXPECT_SCHEME=w4a8-moe-only
export DSV4_QAT_AUDIT_DIR="$OUTPUT_DIR/audit/qat"
export QAT_SCHEME=w4a8-moe-only
export HCCL_DETERMINISTIC=true
export LCCL_DETERMINISTIC=1
export CLOSE_MATMUL_K_SHIFT=1
export VLLM_BATCH_INVARIANT=1
export VERL_LOGGING_LEVEL=INFO
export DSV4_QAT_TMP="$ray_tmp"
export DSV4_CAMEM_PIN_BACKUP=0
export DSV4_ALIGN_CHECKPOINT_AUDIT=1
export DSV4_ALIGN_POLICY_MODE=qat_flash_fp32_rope
export DSV4_ALIGN_MOE_GRAPH_OUTPUTS=1
export DSV4_ALIGN_GRAPH_REFIT_STORAGE=1
export RAY_memory_usage_threshold=0.99
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export MINDSPEED_BRIDGE_AUTOREG_MODE=on
export VLLM_USE_V1=1
export VLLM_VERSION=0.30.0
export VLLM_ASCEND_ENABLE_FLASHCOMM=1
export VLLM_ASCEND_ENABLE_NZ=0
export KIMI_USE_FUSED_KDA=False
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export EXPECTED_NUM_HIDDEN_LAYERS=43
export EXPECTED_NUM_EXPERTS=16
export DSV4_QAT_FAIL_ON_MISMATCH=0

export DSV4_ALIGNMENT_REFERENCE="$reference_model"
export DSV4_ALIGN_LONG_POLICY=1
export DSV4_ALIGN_LAYER_DUMP=0
export DSV4_ALIGN_EXPERT_STAGES=0
export DSV4_ALIGN_ROLLOUT_SELF=0
export DSV4_ALIGN_ATTN_CORE=0
unset DSV4_ALIGN_OPTIMIZER_AUDIT

# Hydra overrides used by the successful eager training and resume runs.
config=(
    "data.train_files=$(hydra_string "$TRAIN_FILE")"
    "data.val_files=$(hydra_string "$VAL_FILE")"
    data.train_batch_size=32
    data.max_prompt_length=1024
    data.max_response_length=512
    data.filter_overlong_prompts=False
    data.filter_overlong_prompts_workers=null
    data.truncation=error
    data.shuffle=False
    data.validation_shuffle=False
    data.trust_remote_code=True
    data.reward_fn_key=data_source
    data.val_batch_size=128

    "actor_rollout_ref.model.path=$(hydra_string "$MODEL_PATH")"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.use_fused_kernels=False

    model_engine=megatron

    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False

    reward.custom_reward_function.path=null
    reward.custom_reward_function.name=compute_score
    reward.reward_manager.source=register
    reward.reward_manager.name=naive
    reward.num_workers=1

    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_decay_style=constant
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0
    actor_rollout_ref.actor.optim.weight_decay=0.01
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.optim.use_precision_aware_optimizer=True
    actor_rollout_ref.actor.optim.main_grads_dtype=bf16
    actor_rollout_ref.actor.optim.exp_avg_dtype=bf16
    actor_rollout_ref.actor.optim.exp_avg_sq_dtype=bf16
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1.0
    +actor_rollout_ref.actor.optim.override_optimizer_config.store_param_remainders=True
    actor_rollout_ref.actor.ppo_mini_batch_size=32
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=1536
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.calculate_entropy=False
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1
    actor_rollout_ref.actor.megatron.context_parallel_size=1
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=8
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1
    actor_rollout_ref.actor.megatron.sequence_parallel=False
    actor_rollout_ref.actor.megatron.use_remove_padding=False
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.optimizer_offload=True
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.align_mhc_fp32=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=False
    ++actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_csa_smla=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_mla_rope=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_experts_glu=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_mhc_add3=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_dsa_kernel_fusion=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.dsa_indexer_loss_coeff=0.0
    +actor_rollout_ref.actor.megatron.override_transformer_config.dsa_indexer_use_sparse_loss=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.csa_dense_mode=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.mtp_num_layers=null
    +actor_rollout_ref.actor.megatron.override_transformer_config.seq_length=1536
    actor_rollout_ref.actor.megatron.forward_only=False
    actor_rollout_ref.actor.megatron.pad_bshd_to_minibatch_max=False
    ++actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_gmm=True
    ++actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_csa_indexer=False
    ++actor_rollout_ref.actor.optim.override_optimizer_config.pin_cpu_grads=False
    ++actor_rollout_ref.actor.megatron.full_determinism=True
    ++actor_rollout_ref.actor.megatron.seed=1234
    ++actor_rollout_ref.actor.megatron.override_transformer_config.align_qat_fp32_combine=True
    actor_rollout_ref.actor.profiler.enable=False
    ++actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_masked_select=True
    ++actor_rollout_ref.actor.megatron.override_transformer_config.align_qat_router_weight_bf16=False
    ++actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_mhc=true
    'actor_rollout_ref.actor.checkpoint.load_contents=["model","optimizer","extra"]'
    'actor_rollout_ref.actor.checkpoint.save_contents=["model","optimizer","extra"]'
    actor_rollout_ref.actor.checkpoint.async_save=false
    actor_rollout_ref.actor.optim.use_checkpoint_opt_param_scheduler=true
    actor_rollout_ref.actor.optim.total_training_steps=100

    ++actor_rollout_ref.ref.megatron.override_transformer_config.align_mhc_fp32=True
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=1536
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.megatron.use_mbridge=True
    actor_rollout_ref.ref.megatron.use_remove_padding=False
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=1
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1
    actor_rollout_ref.ref.megatron.context_parallel_size=1
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=8
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=1
    actor_rollout_ref.ref.megatron.sequence_parallel=False
    actor_rollout_ref.ref.megatron.param_offload=True
    ++actor_rollout_ref.ref.megatron.override_transformer_config.seq_length=1536
    +actor_rollout_ref.ref.megatron.override_transformer_config.apply_rope_fusion=False
    +actor_rollout_ref.ref.megatron.override_transformer_config.use_fused_csa_smla=False
    +actor_rollout_ref.ref.megatron.override_transformer_config.use_fused_mla_rope=False
    +actor_rollout_ref.ref.megatron.override_transformer_config.use_fused_experts_glu=False
    +actor_rollout_ref.ref.megatron.override_transformer_config.use_fused_mhc_add3=False
    +actor_rollout_ref.ref.megatron.override_transformer_config.apply_dsa_kernel_fusion=False
    +actor_rollout_ref.ref.megatron.override_transformer_config.dsa_indexer_loss_coeff=0.0
    +actor_rollout_ref.ref.megatron.override_transformer_config.dsa_indexer_use_sparse_loss=False
    +actor_rollout_ref.ref.megatron.override_transformer_config.csa_dense_mode=False
    +actor_rollout_ref.ref.megatron.override_transformer_config.mtp_num_layers=null

    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.data_parallel_size=8
    actor_rollout_ref.rollout.expert_parallel_size=8
    actor_rollout_ref.rollout.n=8
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.logprobs_mode=raw_logprobs
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    actor_rollout_ref.rollout.max_model_len=1536
    actor_rollout_ref.rollout.max_num_seqs=16
    actor_rollout_ref.rollout.max_num_batched_tokens=2048
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.enforce_eager=true
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.load_format=dummy
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=1536
    '++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="NONE"'
    '++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes=[1, 2, 4, 8, 16]'
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.max_cudagraph_capture_size=16
    +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.rl_config.enabled=True
    +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.dump_config_path=null
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=bfloat16
    actor_rollout_ref.rollout.quantization=ascend
    actor_rollout_ref.rollout.quantization_config_file=null
    actor_rollout_ref.rollout.ignore_eos=False
    actor_rollout_ref.rollout.seed=1234
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.head_dtype=float32
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=False
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.do_sample=False
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.profiler.enable=False
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.mode=0
    '++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.splitting_ops=["vllm::unified_attention_with_output", "vllm::unified_mla_attention_with_output", "vllm::mamba_mixer2", "vllm::mamba_mixer", "vllm::short_conv", "vllm::qwen4_exp_ple_short_conv", "vllm::qwen4_exp_qsa_with_output", "vllm::linear_attention", "vllm::qwen_gdn_attention_core", "vllm::qwen_gdn_attention_core_fused_norm_packed", "vllm::gdn_attention_core_xpu", "vllm::olmo_hybrid_gdn_full_forward", "vllm::sparse_attn_indexer", "vllm::rocm_aiter_sparse_attn_indexer", "vllm::deepseek_v4_attention", "vllm::hpc_rope_norm_forward", "vllm::unified_kv_cache_update", "vllm::unified_mla_kv_cache_update", "vllm::moe_forward", "vllm::moe_forward_shared", "vllm::dsv4_moe_shared_out"]'

    'trainer.logger=["console","file"]'
    trainer.project_name=verl_deepseek_v4
    trainer.experiment_name=dsv4-top16-moe-mxfp4-aligned-eager
    trainer.nnodes=1
    trainer.n_gpus_per_node=8
    trainer.device=npu
    trainer.balance_batch=False
    trainer.val_before_train=false
    "trainer.resume_mode=$resume_mode"
    trainer.save_freq=-1
    trainer.max_actor_ckpt_to_keep=1
    trainer.max_critic_ckpt_to_keep=1
    "trainer.test_freq=$test_freq"
    trainer.total_epochs=1
    "trainer.total_training_steps=$TOTAL_STEPS"
    "trainer.default_local_dir=$(hydra_string "$OUTPUT_DIR/outputs/checkpoints")"
    "trainer.rollout_data_dir=$(hydra_string "$OUTPUT_DIR/outputs/rollouts")"
    trainer.v1.sampler.sync_refill_failed_groups=True
    trainer.val_only=False
    trainer.v1.trainer_mode=sync
    "trainer.validation_data_dir=$(hydra_string "$OUTPUT_DIR/outputs/validation")"
    "trainer.resume_from_path=$resume_path"
    trainer.del_local_ckpt_after_load=false

    transfer_queue.backend.SimpleStorage.num_data_storage_units=2

    ++ray_kwargs.ray_init.address=local
    ++ray_kwargs.ray_init.resources.NPU=8
    ray_kwargs.ray_init.num_cpus=64
    ++ray_kwargs.ray_init.object_store_memory=4294967296
    ++ray_kwargs.ray_init.include_dashboard=False
    "++ray_kwargs.ray_init._temp_dir=$(hydra_string "$ray_tmp/r")"

    "hydra.run.dir=$(hydra_string "$OUTPUT_DIR/outputs/hydra")"

    global_profiler.tool=null
    global_profiler.steps=null
)

# Forward the runtime settings to Ray workers.
ray_env=(
    DSV4_RESUMABLE DSV4_LP_RECOMPUTE DSV4_CONSISTENCY_DIR DSV4_TRACE_DIR
    VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY DSV4_XFER_NO_RECV_EXPERT_FILTER DSV4_XFER_SENDER_EXPERT_FILTER ASCEND_RT_VISIBLE_DEVICES
    OMP_NUM_THREADS PYTHONHASHSEED TOKENIZERS_PARALLELISM TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT
    TQ_NUM_THREADS VERL_USE_MEGATRON_ADAPTOR VLLM_DISABLE_COMPILE_CACHE CUDA_DEVICE_MAX_CONNECTIONS
    KIMI_TRAIN_USE_PARTITION_INVARIANT_KDA_FORWARD KIMI_FUSED_KDA_BACKEND KIMI_VLLM_USE_PARTITION_INVARIANT_KDA KIMI_VLLM_USE_TRAINING_CAUSAL_CONV1D
    KIMI_KDA_OPROJ_FP32_REDUCE VERL_FAIL_ON_NONFINITE_GRAD KIMI_RUN_TOKEN DSV4_FORCE_MOE_ALLTOALL
    DBG_XFER DSV4_LAYER_TRACE DSV4_LAYER_DUMP PROMPTS_ONLY
    ASCEND_LAUNCH_BLOCKING PYTORCH_NPU_ALLOC_CONF DSV4_MOE_TRACE DSV4_A2A_PAD
    DSV4_A2A_LIST ASCEND_GLOBAL_LOG_LEVEL VLLM_RANDOMIZE_DP_DUMMY_INPUTS HCCL_SOCKET_IFNAME
    GLOO_SOCKET_IFNAME HCCL_HOST_SOCKET_PORT_RANGE HCCL_NPU_SOCKET_PORT_RANGE HCCL_CONNECT_TIMEOUT
    HCCL_EXEC_TIMEOUT DSV4_B24_RELEASE_CACHE_BEFORE_FIRST_OPT VLLM_LOGGING_LEVEL RAY_DEDUP_LOGS
    PYTHONPYCACHEPREFIX DSV4_B24_PROCESS_CACHE DSV4_B24_AUDIT_DIR DSV4_B24_FAIL_ON_MISMATCH
    PYTHONPATH TMPDIR TMP TEMP
    XDG_CACHE_HOME HF_HOME TORCH_HOME TORCH_EXTENSIONS_DIR
    TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR VLLM_CACHE_ROOT VLLM_CONFIG_ROOT
    ASCEND_CACHE_PATH ASCEND_WORK_PATH B24_ROOT B24_RUN
    VERL_FILE_LOGGER_ROOT VERL_FILE_LOGGER_PATH ASCEND_PROCESS_LOG_PATH DSV4_QAT_ROOT
    DSV4_QAT_RUN DSV4_QAT_EXPECT_SCHEME DSV4_QAT_AUDIT_DIR QAT_SCHEME
    HCCL_DETERMINISTIC LCCL_DETERMINISTIC CLOSE_MATMUL_K_SHIFT VLLM_BATCH_INVARIANT
    VERL_LOGGING_LEVEL DSV4_QAT_TMP DSV4_CAMEM_PIN_BACKUP DSV4_ALIGN_CHECKPOINT_AUDIT
    DSV4_ALIGN_POLICY_MODE DSV4_ALIGN_MOE_GRAPH_OUTPUTS DSV4_ALIGN_GRAPH_REFIT_STORAGE DSV4_ALIGNMENT_REFERENCE
    DSV4_ALIGN_LONG_POLICY DSV4_ALIGN_EXPERT_STAGES DSV4_ALIGN_LAYER_DUMP DSV4_ALIGN_ROLLOUT_SELF
    DSV4_ALIGN_ATTN_CORE DSV4_MINDSPEED_BRIDGE_ROOT
)
for name in "${ray_env[@]}"; do
    config+=("++ray_kwargs.ray_init.runtime_env.env_vars.${name}=$(hydra_string "${!name}")")
done
if [[ "$resume_mode" = resume_path ]]; then
    export DSV4_ALIGN_OPTIMIZER_AUDIT=1
    config+=('++ray_kwargs.ray_init.runtime_env.env_vars.DSV4_ALIGN_OPTIMIZER_AUDIT="1"')
fi

cd -- "$repo_dir"
if [[ "$dry_run" -eq 1 ]]; then
    exec "$python_bin" -m verl.trainer.main_ppo "${config[@]}" --cfg job
fi

ulimit -l unlimited 2>/dev/null || true
"$python_bin" "$recipe_dir/train.py" "${config[@]}" 2>&1 | tee "$OUTPUT_DIR/train.log"
