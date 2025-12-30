/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "npu_qwen3_omni_code2wav_transformer_layer_impl.h"

#include <atb/atb_infer.h>
#include <atb/types.h>
#include <glog/logging.h>
#include <mstx/ms_tools_ext.h>

#include <iostream>
#include <map>

#include "npu_base_layer.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "xllm_kernels/models/qwen3_omni/code2Wav/layer/code2wav_transformer.h"

namespace xllm {
namespace layer {

enum TransformerLayerTensorId : int {
  IN_INPUT_NORM_WEIGHT = 0,
  IN_QKV_WEIGHT_0 = 1,
  IN_QKV_WEIGHT_1 = 2,
  IN_QKV_WEIGHT_2 = 3,
  IN_QKV_WEIGHT = 4,
  IN_ATTENTION_OUT_WEIGHT = 5,
  IN_SELF_ATTN_LAYER_SCALE_SCALE = 6,
  IN_SELFATTENTION_OUT_NORM_WEIGHT = 7,
  IN_MLP_GATEUP_WEIGHT_EXPERT = 8,
  IN_MLP_DOWN_WEIGHT_EXPERT = 9,
  IN_MLP_LAYER_SCALE_SCALE = 10
};

static const uint64_t WEIGHT_COUNT_PER_LAYER = 11;

static const std::unordered_map<std::string, int> WEIGHT_MAPPING = {
    {"input_layernorm.weight", IN_INPUT_NORM_WEIGHT},

    {"self_attn.q_proj.weight", IN_QKV_WEIGHT_0},

    {"self_attn.k_proj.weight", IN_QKV_WEIGHT_1},

    {"self_attn.v_proj.weight", IN_QKV_WEIGHT_2},

    {"self_attn.o_proj.weight", IN_ATTENTION_OUT_WEIGHT},

    {"post_attention_layernorm.weight", IN_SELFATTENTION_OUT_NORM_WEIGHT},

    // Expert MLP - Gate/Up projections
    {"mlp.gate_proj.weight", IN_MLP_GATEUP_WEIGHT_EXPERT},

    {"mlp.up_proj.weight", IN_MLP_GATEUP_WEIGHT_EXPERT},

    // Expert MLP - Down projection
    {"mlp.down_proj.weight", IN_MLP_DOWN_WEIGHT_EXPERT},

    // add for qwen3-omni
    {"mlp_layer_scale.scale", IN_MLP_LAYER_SCALE_SCALE},

    {"self_attn_layer_scale.scale", IN_SELF_ATTN_LAYER_SCALE_SCALE},
};

static const std::map<int, int> WEIGHT_SHARD = {
    {IN_QKV_WEIGHT_0, 0},
    {IN_QKV_WEIGHT_1, 0},
    {IN_QKV_WEIGHT_2, 0},
    {IN_ATTENTION_OUT_WEIGHT, 1},
    {IN_MLP_GATEUP_WEIGHT_EXPERT, 0},
    {IN_MLP_DOWN_WEIGHT_EXPERT, 1},
};

void Qwen3OmniCode2WavTransformerLayerImpl::param_from_args(
    atb_speed::qwen3_omni::Code2WavTransformerLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args) {
  param.isBF16 = args.dtype() == "bfloat16";
  param.rmsNormEps = args.rms_norm_eps();
  param.worldSize = parallel_args.world_size();
  param.numAttentionHeadsPerRank =
      args.mm_num_attention_heads() / param.worldSize;
  param.hiddenSizePerAttentionHead =
      args.mm_hidden_size() / args.mm_num_attention_heads();
  std::optional<long int> optionalValue = args.mm_num_attention_heads();
  param.numKeyValueHeadsPerRank =
      static_cast<int>(optionalValue.value()) / param.worldSize;
  param.rank = parallel_args.rank();
  param.backend = "lccl";
  param.enableLogN = false;
}

Qwen3OmniCode2WavTransformerLayerImpl::Qwen3OmniCode2WavTransformerLayerImpl(
    const ModelContext& context,
    const int32_t layer_id)
    : BaseLayer(context) {
  auto model_args = context.get_model_args();
  auto parallel_args = context.get_parallel_args();
  auto options = context.get_tensor_options();
  param_from_args(encode_param_, model_args, parallel_args);
  at_weight_tensors_.resize(WEIGHT_COUNT_PER_LAYER);
  atb_weight_tensors_.resize(WEIGHT_COUNT_PER_LAYER);
  dtype_ = c10::typeMetaToScalarType(options.dtype());
  device_id_ = options.device().index();
  placeholder_ = atb_speed::Utils::AtTensor2Tensor(
      torch::zeros({1}).to(device_).to(dtype_));
  at_placeholder_ = torch::zeros({1}).to(device_).to(dtype_);
  for (int i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    at_weight_tensors_[i] = torch::zeros({1}).to(options);
  }
}

void Qwen3OmniCode2WavTransformerLayerImpl::verify_loaded_weights(
    const std::string& prefix) const {
  for (const auto& [name, index] : WEIGHT_MAPPING) {
    CHECK(at_weight_tensors_[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

void Qwen3OmniCode2WavTransformerLayerImpl::merge_loaded_weights() {
  // spilt pack qkv weight when enable tp
  get_weights_col_packed_qkv();
  if (encode_param_.worldSize > 1) {
    // merge qkv weight
    auto new_qkv_weight = torch::cat({at_weight_tensors_[IN_QKV_WEIGHT_0],
                                      at_weight_tensors_[IN_QKV_WEIGHT_1],
                                      at_weight_tensors_[IN_QKV_WEIGHT_2]},
                                     0);
    at_weight_tensors_[IN_QKV_WEIGHT] = new_qkv_weight;
    at_weight_tensors_[IN_QKV_WEIGHT_0] = torch::zeros({1}).to(device_);
    at_weight_tensors_[IN_QKV_WEIGHT_1] = torch::zeros({1}).to(device_);
    at_weight_tensors_[IN_QKV_WEIGHT_2] = torch::zeros({1}).to(device_);
  }
  c10_npu::NPUCachingAllocator::emptyCache();
  for (int i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    atb_weight_tensors_[i] =
        atb_speed::Utils::AtTensor2Tensor(at_weight_tensors_[i]);
  }

  init_layer();
}
// tp spilt weight
void Qwen3OmniCode2WavTransformerLayerImpl::get_weights_col_packed_qkv() {
  int rank = encode_param_.rank;
  int worldSize = encode_param_.worldSize;
  // split qkv weight
  qkv_weight = torch::chunk(at_weight_tensors_[IN_QKV_WEIGHT], 3, 0);
  // weight
  at_weight_tensors_[IN_QKV_WEIGHT_0] =
      (qkv_weight[0].chunk(worldSize, 0))[rank];
  at_weight_tensors_[IN_QKV_WEIGHT_1] =
      (qkv_weight[1].chunk(worldSize, 0))[rank];
  at_weight_tensors_[IN_QKV_WEIGHT_2] =
      (qkv_weight[2].chunk(worldSize, 0))[rank];
}

void Qwen3OmniCode2WavTransformerLayerImpl::load_state_dict(
    const StateDict& state_dict) {
  for (const auto& [name, index] : WEIGHT_MAPPING) {
    if (WEIGHT_SHARD.find(index) != WEIGHT_SHARD.end()) {
      set_weight(state_dict, name, index, WEIGHT_SHARD.at(index));
    } else {
      set_weight(state_dict, name, index);
    }
  }
}

int64_t Qwen3OmniCode2WavTransformerLayerImpl::init_layer() {
  name_ = "qwen3_omni_code2wav_transformer_layer";
  model_name_ = "qwen3_omni_code2wav";
  CHECK_OPERATION_STATUS_RETURN(init_node(encode_node_, encode_param_));
  return atb::NO_ERROR;
}

int64_t Qwen3OmniCode2WavTransformerLayerImpl::init_node(
    atb_speed::Model::Node& node,
    atb_speed::qwen3_omni::Code2WavTransformerLayerParam& param) {
  atb::Operation* operation = nullptr;
  atb_speed::qwen3_omni::Qwen3Omni_Code2wav_TransformerLayer(param, &operation);
  node.operation.reset(operation);
  if (node.operation == nullptr) {
    LOG(ERROR) << "node.operation is null";
    return -1;
  }
  if (node.operation->GetInputNum() < 1) {
    LOG(ERROR) << "Can not resize number which is smaller than 1";
    return -1;
  }
  node.inTensors.resize(node.operation->GetInputNum());
  node.outTensors.resize(1);
  size_t inTensorId = 1;

  for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER;
       ++weightTensorId) {
    node.inTensors.at(weightTensorId) = &atb_weight_tensors_[weightTensorId];
  }

  node.variantPack.inTensors.reserve(node.inTensors.size());
  node.variantPack.inTensors.resize(node.inTensors.size());
  node.variantPack.outTensors.reserve(1);
  node.variantPack.outTensors.resize(1);
  return atb::NO_ERROR;
}

torch::Tensor Qwen3OmniCode2WavTransformerLayerImpl::forward(
    torch::Tensor& x,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
    torch::Tensor& attn_mask,
    torch::Tensor& cu_seqlen,
    std::vector<int>& cu_seqlen_vec,
    ModelInputParams& input_params,
    int node_id,
    aclrtEvent* event,
    std::atomic<bool>* event_flag) {
  atb::Status st;

  build_node_variant_pack(encode_node_,
                          x,
                          cos_pos,
                          sin_pos,
                          attn_mask,
                          cu_seqlen,
                          cu_seqlen_vec,
                          input_params,
                          true);
  // mstxRangeEnd(id);
  st = execute_node(encode_node_, node_id);
  LOG_IF(FATAL, st != 0) << model_name_
                         << "excute transformer layer fail, error code: " << st;
  return x;
}

void Qwen3OmniCode2WavTransformerLayerImpl::build_node_variant_pack(
    atb_speed::Model::Node& node,
    torch::Tensor& x,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
    torch::Tensor& attn_mask,
    torch::Tensor& cu_seqlen,
    std::vector<int>& cu_seqlen_vec,
    ModelInputParams& input_params,
    bool is_prefill) {
  internal_tensors_ = atb_speed::Utils::AtTensor2Tensor(x);

  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER) = internal_tensors_;
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 1) =
      atb_speed::Utils::AtTensor2Tensor(cos_pos);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 2) =
      atb_speed::Utils::AtTensor2Tensor(sin_pos);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 3) =
      atb_speed::Utils::AtTensor2Tensor(attn_mask);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 4) =
      atb_speed::Utils::AtTensor2Tensor(cu_seqlen);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 5).hostData =
      cu_seqlen_vec.data();

  for (size_t i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    CHECK_THROW(node.inTensors.at(i) == nullptr,
                model_name_ << "inTensor " << i << "is NULL");
    node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
    // LOG(INFO) << model_name_ << "inTensors[" << i << "]:"
    //               << atb_speed::TensorUtil::TensorToString(
    //                      node.variantPack.inTensors.at(i));
  }

  node.variantPack.outTensors.at(0) = internal_tensors_;
}

}  // namespace layer
}  // namespace xllm
