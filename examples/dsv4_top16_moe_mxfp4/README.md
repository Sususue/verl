# DSV4 Flash top16 MoE MXFP4 单机训推对齐

这套配置用 Ascend 950DT 单机八卡复现 DSV4 Flash top16 MoE MXFP4 的 eager GRPO。发送端将 Actor 更新后的权重按 HF 格式流式导出，再按 32 元素分组生成 MXFP4 权重和 E8M0 scale，随后分桶传给 rollout。接收端校验并直接加载打包载荷。

| 验收 | 完整优化步 | 有效 token | 原生 rollout 对独立 Actor 的 logp MAE / 概率 MAE / Pearson |
| --- | ---: | ---: | --- |
| 从原始 top16 权重起训 | 2 | 114,817 | `0 / 0 / 1` |
| 从第 80 步 checkpoint 续训 | 10（第 81–90 步） | 287,027 | `0 / 0 / 1` |

两轮实际 PPO 损失前向与独立 Actor 重算也逐 token 相等。续训保留模型和 Adam 状态恢复修复，第 81 步的 Adam 计数从 80 更新到 81。

## 1. 拉取基础镜像并创建构建容器

在 Ascend 950DT 的 Linux ARM64 宿主机上，拉取 `quay.io/ascend/vllm-ascend:nightly-releases-v0.30.0-a5`，然后在独立容器里检出源码并安装训练依赖。以下路径按宿主机实际位置填写。

| 宿主机变量 | 容器内路径 | 用途 |
| --- | --- | --- |
| `HOST_INPUT_ROOT` | `/inputs` | 原始裁剪权重和数据，只读挂载 |
| `HOST_WORK_ROOT` | `/workspace/dsv4-mxfp4` | 本地 NVMe 上的源码、缓存、日志和 checkpoint |

```bash
export BASE_IMAGE=quay.io/ascend/vllm-ascend:nightly-releases-v0.30.0-a5
export HOST_INPUT_ROOT=/abs/path/dsv4-inputs
export HOST_WORK_ROOT=/abs/local-nvme/dsv4-mxfp4

mkdir -p "$HOST_WORK_ROOT"
docker pull --platform linux/arm64 "$BASE_IMAGE"
docker image inspect --format '{{json .RepoDigests}}' "$BASE_IMAGE"

npu_devices=()
for i in {0..7}; do npu_devices+=(--device "/dev/davinci$i"); done

docker run -it --name dsv4-mxfp4-env \
  --network host --shm-size=64g --ulimit nofile=65536:65536 \
  "${npu_devices[@]}" --device /dev/davinci_manager \
  --device /dev/devmm_svm --device /dev/hisi_hdc \
  --mount type=bind,src=/usr/local/Ascend/driver,dst=/usr/local/Ascend/driver,readonly \
  --mount type=bind,src=/usr/local/dcmi,dst=/usr/local/dcmi,readonly \
  --mount type=bind,src=/etc/ascend_install.info,dst=/etc/ascend_install.info,readonly \
  --mount "type=bind,src=$(command -v npu-smi),dst=/usr/local/bin/npu-smi,readonly" \
  --mount "type=bind,src=$HOST_INPUT_ROOT,dst=/inputs,readonly" \
  --mount "type=bind,src=$HOST_WORK_ROOT,dst=/workspace/dsv4-mxfp4" \
  --workdir /workspace/dsv4-mxfp4 --entrypoint /bin/bash "$BASE_IMAGE"
```

进入容器后执行第 2、3 节。再次进入已退出的容器可用 `docker start -ai dsv4-mxfp4-env`，并重新加载 CANN 和 ATB。nightly 标签会更新，建议记录本次拉取显示的仓库摘要。

下表是历史长跑环境的对照值。基础镜像的 ARM64 清单摘要与历史迁移镜像的本地 ID 分别记录在 [versions.lock.yaml](versions.lock.yaml)。

| 项目 | 参考环境 |
| --- | --- |
| 历史容器系统与硬件 | aarch64 openEuler 24.03 LTS-SP4，Ascend 950DT × 8 |
| 驱动与 CANN | 驱动 `26.2.0.b020`，CANN `9.2.1` |
| Python | `3.12.16` |
| PyTorch / Torch-NPU | `2.10.0+cpu` / `2.10.0.post7` |
| Ray / Transformers | `2.48.0` / `5.14.1` |
| vLLM / vLLM-Ascend 包版本 | `0.30.0+empty` / `0.30.0rc2.dev316+g3509a6d15` |
| Triton / Triton Ascend | `3.6.0` / `3.6.0.post2` |
| 原生组件 | 镜像中的 HCCL、`batch_invariant_ops 1.0.0` 和与 CANN 匹配的 Ascend 自定义算子 |

## 2. 拉取源码并检出指定版本

`SOURCE_ROOT` 是容器内新的源码根目录。五个修改仓的 `dsv4-top16-moe-mxfp4` 分支均发布后，使用下方 HTTPS 地址获取固定提交。

| 仓库 | 来源 | 训推对齐提交 |
| --- | --- | --- |
| `verl` | [Sususue/verl](https://github.com/Sususue/verl)，包含训练入口 | `81261489` |
| `MindSpeed-Bridge` | [qq_44739108/MindSpeed-Bridge](https://gitcode.com/qq_44739108/MindSpeed-Bridge) | `5cd84eef` |
| `Megatron-LM` | [Sususue/Megatron-LM](https://github.com/Sususue/Megatron-LM) | `add3dcf1` |
| `Megatron-Bridge` | [Sususue/Megatron-Bridge](https://github.com/Sususue/Megatron-Bridge) | `219ee113` |
| `vllm-ascend` | [Sususue/vllm-ascend](https://github.com/Sususue/vllm-ascend) | `5514c452` |
| `vllm` | vLLM 上游 | `ced6857a` |
| `MegatronAdaptor` | Ascend 上游 | `04951c8b` |
| `MindSpeed` | Ascend 上游 | `bafa47af` |
| `MindSpeed-Ops` | Ascend 上游 | `e0d16d28` |
| `TransformerEngineNPU` | Ascend 上游 | `1ac83362` |

下面的命令固定到完整 SHA，便于在分支后续追加修复后复现这一版的 eager 配置。MindSpeed 系列依赖采用实跑时的 editable 源码 SHA。

```bash
export SOURCE_ROOT=/workspace/dsv4-mxfp4/sources

bash <<'BASH'
set -euo pipefail
mkdir -p "$SOURCE_ROOT"
cd "$SOURCE_ROOT"

checkout_at() {
    local repo=$1 source=$2 commit=$3
    mkdir "$repo"
    git -C "$repo" init -q
    git -C "$repo" remote add origin "$source"
    if ! git -C "$repo" fetch --no-tags --depth=1 origin "$commit"; then
        # For servers that require fetching a named ref before checkout by SHA.
        git -C "$repo" fetch --tags origin
    fi
    git -C "$repo" checkout --detach "$commit"
}

checkout_at verl https://github.com/Sususue/verl.git 812614890654630da022cbbd6ea3ae769482e27f
checkout_at MindSpeed-Bridge https://gitcode.com/qq_44739108/MindSpeed-Bridge.git 5cd84eef2e39c39254723677d30a6a6dee0ba206
checkout_at Megatron-LM https://github.com/Sususue/Megatron-LM.git add3dcf1f348d66280a861bd475e7f2648dd21f9
checkout_at Megatron-Bridge https://github.com/Sususue/Megatron-Bridge.git 219ee113ac7cd6b714d71805ee3e74d5a74fcaa3
checkout_at vllm-ascend https://github.com/Sususue/vllm-ascend.git 5514c452e70ac05071656f6717071e56b024c873

checkout_at vllm https://github.com/vllm-project/vllm.git ced6857afa0ea7b2e3f0846a62e1394e90f15607
checkout_at MegatronAdaptor https://gitcode.com/ascend/MegatronAdaptor.git 04951c8b8919efda9b9465017d2e0eaad6d344e0
checkout_at MindSpeed https://gitcode.com/ascend/MindSpeed.git bafa47afb8bdfa89bcdfb12b91461def43f6c391
checkout_at MindSpeed-Ops https://gitcode.com/ascend/MindSpeed-Ops.git e0d16d2891f35f1504104fbe9e94b1a1a536b2ba
checkout_at TransformerEngineNPU https://gitcode.com/ascend/TransformerEngineNPU.git 1ac83362fb48cb55efb2bf0cf4bcbe299f79ce0a
BASH
```

## 3. 构建训练环境

先加载 CANN 和 ATB，补齐 GRPO 训练依赖。约束文件保留基础镜像中已安装的 Torch、Torch-NPU 和 Triton 版本。

```bash
cd "$SOURCE_ROOT"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

python3 - <<'PY' > "$SOURCE_ROOT/base-npu-constraints.txt"
from importlib.metadata import PackageNotFoundError, version

for name in ("torch", "torch-npu", "torchvision", "torchaudio", "triton", "triton-ascend"):
    try:
        print(f"{name}=={version(name)}")
    except PackageNotFoundError:
        continue
PY

python3 -m pip install -c "$SOURCE_ROOT/base-npu-constraints.txt" \
  'ray[default]==2.48.0' 'hydra-core==1.3.8' 'omegaconf==2.3.1' \
  'transformers==5.14.1' 'accelerate==1.15.0' 'tensordict==0.10.0' \
  'torchdata==0.11.0' 'TransferQueue==0.1.11' 'peft==0.21.2' \
  'numpy==1.26.4' 'pandas==3.0.6' 'datasets==5.1.0' 'pyarrow==24.0.0' \
  'dill==0.4.1' 'codetiming==1.4.0' 'einops==0.8.2' \
  'mathruler==0.1.0' 'pylatexenc==2.11' 'qwen-vl-utils==0.0.14'
```

再按顺序安装上节检出的源码。vLLM-Ascend 在容器内用 A5 工具链编译自定义算子，安装时沿用历史运行的 vLLM 包版本标识。

```bash
python3 -m pip install --no-deps --no-build-isolation \
  -e ./Megatron-LM -e ./Megatron-Bridge -e ./MegatronAdaptor \
  -e ./MindSpeed -e ./MindSpeed-Ops -e ./TransformerEngineNPU -e ./MindSpeed-Bridge

VLLM_TARGET_DEVICE=empty VLLM_VERSION_OVERRIDE=0.30.0+empty \
  python3 -m pip install --no-deps --no-build-isolation -e ./vllm
git -C ./vllm-ascend submodule update --init --recursive
SOC_VERSION=ascend950dt_9582 COMPILE_CUSTOM_KERNELS=1 VLLM_BATCH_INVARIANT=1 \
  SETUPTOOLS_SCM_PRETEND_VERSION=0.30.0rc2.dev316+g3509a6d15 \
  python3 -m pip install --no-deps --no-build-isolation -e ./vllm-ascend
python3 -m pip install --no-deps --no-build-isolation -e ./verl
```

历史长跑使用 `batch_invariant_ops 1.0.0`。如基础镜像未带此包，从可访问的 wheel 源获取匹配当前 Python 和 CANN 的 AArch64 wheel，再用 `python3 -m pip install --no-deps /abs/path/to/batch_invariant_ops.whl` 安装。

```bash
python3 - <<'PY'
import torch_npu
import batch_invariant_ops
import megatron.bridge
import megatron.core
import mindspeed_bridge
import verl
import vllm
import vllm_ascend

for module in (verl, megatron.core, megatron.bridge, mindspeed_bridge, vllm, vllm_ascend):
    print(module.__name__, module.__file__)
PY
npu-smi info
```

核对源码导入路径、`ASCEND_CUSTOM_OPP_PATH`、自定义算子加载和八张 NPU 的可见性。

## 4. 训练和续训

| 配置项 | 默认值 |
| --- | --- |
| 模型 | DSV4 Flash 43 层，前 3 层各 256 个路由专家，后 40 层各 16 个 |
| 量化 | MoE-only W4A8 QAT，发送端在分桶前打包 MXFP4 专家权重和 E8M0 scale，激活使用 MXFP8 |
| Actor 并行 | TP1、PP1、CP1、EP8、ETP1 |
| Rollout 并行 | TP1、DP8、EP8 |
| 批量 | GBS 32，rollout `n=8`，Actor 与 old-logprob MBS 均为 2 |
| 序列与采样 | 输入上限 1024，输出上限 512，正常 EOS，生成时记录 raw logprob |
| 实际算子路径 | Actor 融合 mHC 和 Grouped GEMM，关闭训练侧 CSA SMLA 和 CSA indexer 量化 |
| 推理模式 | eager rollout，BF16 KV cache、FP32 LM head |
| 步数与评测 | 原始权重默认跑 2 步，起训前关闭评测。续训默认再跑 10 步，关闭评测 |
| 优化器与 checkpoint | 优化器调度总步数为 100。复现配置关闭保存；续训读取指定 checkpoint 的模型、优化器、LR、RNG 和数据状态 |

已有在线量化目录时，直接设置 `MODEL_PATH`。若只有原始裁剪权重，先生成带 `quant_model_description.json` 的目录。

```bash
cd "$SOURCE_ROOT/verl"
python3 tools/smart_create_mxfp4_weight_dir.py /inputs/top16 /workspace/dsv4-mxfp4/top16-online-mxfp4
```

| 变量 | 用途 |
| --- | --- |
| `MODEL_PATH` | 在线 MXFP4 模型目录 |
| `TRAIN_FILE` / `VAL_FILE` | 训练和固定 dev 的 parquet 文件 |
| `NET_IFACE` | HCCL 和 Gloo 使用的网卡 |
| `OUTPUT_DIR` | 本地 NVMe 上尚未创建的独立运行目录 |
| `ASCEND_RT_VISIBLE_DEVICES` | 可选，默认 `0,1,2,3,4,5,6,7` |
| `RESUME_FROM` / `TOTAL_STEPS` | 可选，完整 `global_step_N` 的绝对路径和累计目标步数。目标须大于起始步且不超过 100 |

```bash
cd "$SOURCE_ROOT/verl"
export MODEL_PATH=/workspace/dsv4-mxfp4/top16-online-mxfp4
export TRAIN_FILE=/inputs/data/train.parquet
export VAL_FILE=/inputs/data/dev.parquet
export NET_IFACE=your_network_interface
export OUTPUT_DIR=/workspace/dsv4-mxfp4/runs/long-run

bash examples/dsv4_top16_moe_mxfp4/run_grpo.sh --dry-run
bash examples/dsv4_top16_moe_mxfp4/run_grpo.sh
```

续训沿用模型、数据和网卡设置。填写已有完整的第 80 步 checkpoint 路径，并指定新的输出目录。

```bash
export RESUME_FROM=/workspace/dsv4-mxfp4/checkpoints/global_step_80
export OUTPUT_DIR=/workspace/dsv4-mxfp4/runs/resume
export TOTAL_STEPS=90
bash examples/dsv4_top16_moe_mxfp4/run_grpo.sh
```

## 5. 查看产物与验证状态

| 产物 | 内容 |
| --- | --- |
| `train.log` | 训练和权重回灌日志 |
| `metrics/steps.jsonl` | reward、`training/rollout_probs_diff_mean`、`training/rollout_actor_probs_pearson_corr` |
| `outputs/hydra/` | 实际运行配置 |
| `audit/` 与 `samples/` | 权重收发、checkpoint 恢复审计及用于逐 token 复核的样本 |

整理后的入口已用原始起训与 checkpoint 续训的生效配置做 CPU 对照。上表 NPU 结果来自归档的 eager 实跑。验收以同策略、同 token、同位置和同有效 mask 下的 Actor 与原生 rollout logprob 为准。
