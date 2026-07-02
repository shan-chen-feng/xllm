# Qwen Image Edit Plus 部署

本文档介绍如何基于 xLLM 在 Ascend NPU 环境中部署 Qwen Image Edit Plus 图像编辑服务。该模型服务由两个独立模块组成：

- **Text Encoder**：接收文本和图片，输出多模态 embedding。
- **DiT**：接收 Text Encoder 生成的 embedding 和输入图片，执行图像生成/编辑。

## 环境准备

### 硬件与软件

| 项目 | 要求 |
| ---- | ---- |
| 硬件 | Ascend NPU |
| 工具链 | ascend-toolkit、nnal/atb |
| 服务程序 | 已编译的 xLLM 服务，例如 `./build/xllm/core/server/xllm` |

### Python 依赖

请求脚本需要额外安装以下依赖：

```bash
pip install torch numpy Pillow requests diffusers
```

## 权重准备

模型根目录需要包含 DiT 服务加载所需的组件目录。典型目录结构如下：

```text
Qwen_Image_Edit_Plus/
├── model_index.json
├── processor/
├── text_encoder/
├── tokenizer/
├── transformer/
└── vae/
```

Text Encoder 单独启动时会直接读取 `text_encoder/` 目录。若下载的权重中 tokenizer 文件不在 `text_encoder/` 下，需要将 tokenizer 相关文件复制到 `text_encoder/` 目录，确保 Text Encoder 能正常解析图片和文本输入。

## 启动 Text Encoder

Text Encoder 使用 VLM embedding 服务，对外提供 `/v1/embeddings` 接口。

```bash
#!/bin/bash
set -e

export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/
export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
export LIBTORCH_ROOT="${PYTORCH_INSTALL_PATH}"
export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:${LD_LIBRARY_PATH}

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export ASDOPS_LOG_TO_STDOUT=1
export ASDOPS_LOG_LEVEL=ERROR
export ASDOPS_LOG_TO_FILE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.98
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export OMP_NUM_THREADS=12
export HCCL_CONNECT_TIMEOUT=7200
export INF_NAN_MODE_ENABLE=0
export INF_NAN_MODE_FORCE_DISABLE=1
export HCCL_IF_BASE_PORT=43438

XLLM_PATH="./build/xllm/core/server/xllm"
MODEL_PATH="/path/to/Qwen_Image_Edit_Plus/text_encoder"
MODEL_ID="text_encoder"
MASTER_NODE_ADDR="127.0.0.1:9798"
START_PORT=18007
START_DEVICE=8
LOG_DIR="log_text_encoder"
NNODES=1

rm -rf core.*
rm -rf ${LOG_DIR}/node_*.log
mkdir -p ${LOG_DIR}

for (( i=0; i<${NNODES}; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="${LOG_DIR}/node_${i}.log"
  ${XLLM_PATH} \
    --model ${MODEL_PATH} \
    --model_id ${MODEL_ID} \
    --backend="vlm" \
    --task="embed" \
    --devices="npu:${DEVICE}" \
    --port ${PORT} \
    --master_node_addr=${MASTER_NODE_ADDR} \
    --nnodes=${NNODES} \
    --node_rank=${i} \
    --max_memory_utilization=0.86 \
    --max_tokens_per_batch=40000 \
    --max_seqs_per_batch=256 \
    --block_size=128 \
    --communication_backend="hccl" \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=true \
    --enable_schedule_overlap=false \
    --enable_shm=true \
    --enable_return_mm_full_embeddings=1 \
    > ${LOG_FILE} 2>&1 &
done
```

启动命令：

```bash
bash start_text_encoder.sh
```

## 启动 DiT

DiT 使用 diffusion 后端，对外提供 `/v1/image/generation` 接口。Text Encoder 与 DiT 需要使用不同的 NPU 设备和服务端口。

```bash
#!/bin/bash
set -e

export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/
export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
export LIBTORCH_ROOT="${PYTORCH_INSTALL_PATH}"
export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:${LD_LIBRARY_PATH}

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export ASDOPS_LOG_TO_STDOUT=1
export ASDOPS_LOG_LEVEL=ERROR
export ASDOPS_LOG_TO_FILE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.98
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export OMP_NUM_THREADS=12
export HCCL_CONNECT_TIMEOUT=7200
export INF_NAN_MODE_ENABLE=0
export INF_NAN_MODE_FORCE_DISABLE=1
export HCCL_IF_BASE_PORT=43432

XLLM_PATH="./build/xllm/core/server/xllm"
MODEL_ROOT="/path/to/Qwen_Image_Edit_Plus"
MODEL_ID="Qwen_Image_Edit_2509"
MASTER_NODE_ADDR="127.0.0.1:19757"
START_PORT=18002
START_DEVICE=0
LOG_DIR="log_dit"
NNODES=8

rm -rf core.*
rm -rf ${LOG_DIR}/node_*.log
mkdir -p ${LOG_DIR}

for (( i=0; i<${NNODES}; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="${LOG_DIR}/node_${i}.log"
  ${XLLM_PATH} \
    --model="${MODEL_ROOT}" \
    --model_id="${MODEL_ID}" \
    --backend="dit" \
    --dit_cache_policy="TaylorSeer" \
    --dit_cache_warmup_steps=0 \
    --cfg_size=2 \
    --sp_size=4 \
    --devices="npu:${DEVICE}" \
    --master_node_addr=${MASTER_NODE_ADDR} \
    --nnodes=${NNODES} \
    --node_rank=${i} \
    --port ${PORT} \
    --communication_backend="hccl" \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --enable_schedule_overlap=false \
    --enable_shm=true \
    --use_contiguous_input_buffer=false \
    --dit_debug_print=true \
    --max_memory_utilization=0.6 \
    > ${LOG_FILE} 2>&1 &
done
```

启动命令：

```bash
bash start_dit.sh
```

## DiT 参数说明

| 参数 | 说明 | 默认值 | 取值 |
| ---- | ---- | ------ | ---- |
| `--sp_size` | Sequence Parallel 并行度 | `1` | 正整数，如 `1`、`2`、`4`、`8` |
| `--cfg_size` | Classifier-Free Guidance 并行度 | `1` | `1` 或 `2` |
| `--dit_cache_policy` | DiT Cache 性能优化策略 | `TaylorSeer` | `TaylorSeer`、`None` |
| `--dit_cache_warmup_steps` | 前多少步保持完整计算，之后启用 DiT Cache 预测优化 | `0` | 非负整数 |
| `--dit_vae_image_size` | Qwen Image Edit Plus VAE 尺寸计算使用的图像面积 | `1048576` | 正整数 |

`NNODES` 必须等于 `sp_size * cfg_size`。

| `sp_size` | `cfg_size` | `NNODES` | 说明 |
| --------- | ---------- | -------- | ---- |
| `1` | `1` | `1` | 单卡部署 |
| `2` | `1` | `2` | 仅开启 SP |
| `1` | `2` | `2` | 仅开启 CFG 并行 |
| `2` | `2` | `4` | 同时开启 SP 和 CFG |
| `4` | `2` | `8` | 8 卡部署 |

`TaylorSeer` 可提升推理速度，但可能对生成精度有轻微影响；如果需要优先保证精度，可设置 `--dit_cache_policy="None"`。

## 请求调用

请求流程分两步：

1. 调用 Text Encoder 的 `/v1/embeddings`，分别生成正向和负向 embedding。
2. 调用 DiT 的 `/v1/image/generation`，传入 embedding、源图和参考图，生成编辑后的图片。

### 基本用法

```bash
python request_qwen_image_edit_plus.py \
  --source_image ./source.png \
  --condition_image ./condition.jpg \
  --encoder_host 127.0.0.1 \
  --encoder_port 18007 \
  --encoder_model text_encoder \
  --dit_host 127.0.0.1 \
  --dit_port 18002 \
  --dit_model_id Qwen_Image_Edit_2509 \
  --positive_prompt "将第一张图中的冲锋衣穿到第二张图片中的模特身上。" \
  --negative_prompt "hat" \
  --size 896*1184 \
  --num_inference_steps 40 \
  --guidance_scale 1.0 \
  --true_cfg_scale 4.0 \
  --seed 0 \
  --output_image ./result.png
```

### 请求参数

| 参数 | 说明 | 默认值 |
| ---- | ---- | ------ |
| `--encoder_host` | Text Encoder 服务地址 | `127.0.0.1` |
| `--encoder_port` | Text Encoder 服务端口 | `18007` |
| `--encoder_model` | Text Encoder 模型名称，需要与 `--model_id` 一致 | `text_encoder` |
| `--dit_host` | DiT 服务地址 | `127.0.0.1` |
| `--dit_port` | DiT 服务端口 | `18002` |
| `--dit_model_id` | DiT 模型 ID，需要与 DiT 服务 `--model_id` 一致 | `Qwen_Image_Edit_2509` |
| `--source_image` | 源图路径，即待编辑图片 | 必填 |
| `--condition_image` | 条件图路径，即参考图片 | 必填 |
| `--positive_prompt` | 正向提示词 | `将第一张图中的冲锋衣穿到第二张图片中的模特身上。` |
| `--negative_prompt` | 负向提示词 | `hat` |
| `--size` | 输出图像尺寸，格式为 `宽*高` | `896*1184` |
| `--num_inference_steps` | 推理步数 | `40` |
| `--guidance_scale` | guidance scale | `1.0` |
| `--true_cfg_scale` | true CFG scale | `4.0` |
| `--num_images_per_prompt` | 每条提示生成的图片数 | `1` |
| `--seed` | 随机种子 | `0` |
| `--max_sequence_length` | 最大序列长度 | `2048` |
| `--skip_tokens` | 从 embedding 头部跳过的 token 数 | `64` |
| `--output_image` | 输出图片路径 | `./result.png` |
| `--timeout` | DiT 请求超时时间，单位秒 | `300` |

### Python 调用示例

下面的示例展示核心请求格式。实际使用时可按需补充图片 resize、异常处理和参数解析。

```python
import base64
import json
import time
from pathlib import Path

import requests
import torch


def image_to_base64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


def create_tensor(tensor: torch.Tensor, name: str) -> dict:
    data = tensor.to(torch.float32).cpu().numpy()
    return {
        "name": name,
        "datatype": "FP32",
        "shape": list(data.shape),
        "contents": {"fp32_contents": data.flatten().tolist()},
    }


def request_embedding(host, port, model, prompt, image_base64_list, skip_tokens=64):
    content = []
    for idx, image_base64 in enumerate(image_base64_list):
        content.append({"type": "text", "text": f"Picture {idx + 1}: "})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_base64}"},
        })
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": model,
        "input": "",
        "encoding_format": "float",
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Describe the input images and follow the edit instruction."}],
            },
            {"role": "user", "content": content},
        ],
    }

    response = requests.post(f"http://{host}:{port}/v1/embeddings", json=payload)
    response.raise_for_status()
    embedding = response.json()["data"][0]["mm_embeddings"][0]["embedding"]
    raw = base64.b64decode(embedding["contents"]["bytes_contents"])
    tensor = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(embedding["shape"])
    return tensor[skip_tokens:] if skip_tokens > 0 else tensor


def request_image_generation(host, port, model_id, pos_embed, neg_embed, source_image, condition_image):
    payload = {
        "model": model_id,
        "input": {
            "prompt": "将第一张图中的冲锋衣穿到第二张图片中的模特身上。",
            "negative_prompt": "hat",
            "prompt_embed": create_tensor(pos_embed, "prompt_embeds"),
            "negative_prompt_embed": create_tensor(neg_embed, "negative_prompt_embed"),
            "images": [source_image, condition_image],
        },
        "parameters": {
            "size": "896*1184",
            "num_inference_steps": 40,
            "guidance_scale": 1.0,
            "true_cfg_scale": 4.0,
            "num_images_per_prompt": 1,
            "seed": 0,
            "max_sequence_length": 2048,
        },
        "user": "test_user",
        "request_id": f"req-{int(time.time())}",
    }

    response = requests.post(
        f"http://{host}:{port}/v1/image/generation",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=300,
    )
    response.raise_for_status()
    result = response.json()
    image_base64 = result["output"]["results"][0]["image"]
    Path("result.png").write_bytes(base64.b64decode(image_base64))


source = image_to_base64("./source.png")
condition = image_to_base64("./condition.jpg")
pos = request_embedding("127.0.0.1", 18007, "text_encoder", "将第一张图中的冲锋衣穿到第二张图片中的模特身上。", [source, condition])
neg = request_embedding("127.0.0.1", 18007, "text_encoder", "hat", [source, condition])
request_image_generation("127.0.0.1", 18002, "Qwen_Image_Edit_2509", pos, neg, source, condition)
```

## 注意事项

- Text Encoder 与 DiT 需要使用不同端口，NPU 设备也不要重叠。
- DiT 请求中的 `model` 必须与 DiT 服务启动时的 `--model_id` 一致；Text Encoder 请求中的 `model` 必须与 Text Encoder 服务启动时的 `--model_id` 一致。
- Qwen Image Edit Plus 当前以 `input.images` 接收多张输入图片，第一张通常为源图，第二张为参考/条件图。
- 当前 Qwen Image Edit Plus Pipeline 不支持 batch image inference，建议 `num_images_per_prompt=1`。
- 若生成尺寸不传，Pipeline 会根据第一张输入图的宽高比自动计算；显式传入时使用 `宽*高` 格式，例如 `896*1184`。
