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

#include "npu_qwen3_audio_encoder_layer_impl.h"

#include <glog/logging.h>
#include <mstx/ms_tools_ext.h>

#include <iostream>
#include <map>

#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "xllm_kernels/models/qwen3_audio/qwen3_audio.h"

namespace xllm {
namespace layer {

enum AudioEncoderLayerTensorId : int {
  AUDIO_IN_INPUT_NORM_WEIGHT = 0,
  AUDIO_IN_INPUT_NORM_BIAS,
  AUDIO_IN_POST_NORM_WEIGHT,
  AUDIO_IN_POST_NORM_BIAS,
  AUDIO_IN_QKV_WEIGHT,
  AUDIO_IN_QKV_BIAS,
  AUDIO_IN_WATTENTION_OUT_WEIGHT,
  AUDIO_IN_WATTENTION_OUT_BIAS,
  AUDIO_IN_LINEAR_FC1_WEIGHT,
  AUDIO_IN_LINEAR_FC1_BIAS,
  AUDIO_IN_LINEAR_FC2_WEIGHT,
  AUDIO_IN_LINEAR_FC2_BIAS,
  AUDIO_IN_Q_WEIGHT,
  AUDIO_IN_Q_BIAS,
  AUDIO_IN_K_WEIGHT,
  AUDIO_IN_K_BIAS,
  AUDIO_IN_V_WEIGHT,
  AUDIO_IN_V_BIAS
};

const uint64_t WEIGHT_COUNT_PER_LAYER = 18;

static std::vector<std::pair<int, std::string>> WEIGHT_MAPPING = {
    {AUDIO_IN_INPUT_NORM_WEIGHT, "self_attn_layer_norm.weight"},
    {AUDIO_IN_INPUT_NORM_BIAS, "self_attn_layer_norm.bias"},
    {AUDIO_IN_POST_NORM_WEIGHT, "final_layer_norm.weight"},
    {AUDIO_IN_POST_NORM_BIAS, "final_layer_norm.bias"},
    {AUDIO_IN_WATTENTION_OUT_WEIGHT, "self_attn.out_proj.weight"},
    {AUDIO_IN_WATTENTION_OUT_BIAS, "self_attn.out_proj.bias"},
    {AUDIO_IN_LINEAR_FC1_WEIGHT, "fc1.weight"},
    {AUDIO_IN_LINEAR_FC1_BIAS, "fc1.bias"},
    {AUDIO_IN_LINEAR_FC2_WEIGHT, "fc2.weight"},
    {AUDIO_IN_LINEAR_FC2_BIAS, "fc2.bias"},
    {AUDIO_IN_Q_WEIGHT, "self_attn.q_proj.weight"},
    {AUDIO_IN_Q_BIAS, "self_attn.q_proj.bias"},
    {AUDIO_IN_K_WEIGHT, "self_attn.k_proj.weight"},
    {AUDIO_IN_K_BIAS, "self_attn.k_proj.bias"},
    {AUDIO_IN_V_WEIGHT, "self_attn.v_proj.weight"},
    {AUDIO_IN_V_BIAS, "self_attn.v_proj.bias"}};

// {weight,dim}
static std::map<int, int> WEIGHT_SHARD = {
    {AUDIO_IN_WATTENTION_OUT_WEIGHT, 1},
    {AUDIO_IN_LINEAR_FC1_WEIGHT, 0},
    {AUDIO_IN_LINEAR_FC1_BIAS, 0},
    {AUDIO_IN_LINEAR_FC2_WEIGHT, 1},
};

void NpuQwen3AudioEncoderLayerImpl::param_from_args(
    atb_speed::qwen::AudioEncoderLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args) {
  param.isBF16 = args.dtype() == "bfloat16";
  param.rmsNormEps = args.rms_norm_eps();
  param.worldSize = parallel_args.world_size();
  bool padding_heads = args.mm_audio_num_attention_heads() % param.worldSize > 0 ? true : false;
  param.numAttentionHeadsPerRank =
      args.mm_audio_num_attention_heads() / param.worldSize;
  /*
  if (padding_heads) {
      param.numAttentionHeadsPerRank = param.numAttentionHeadsPerRank + 1;
  }
  */
  param.hiddenSizePerAttentionHead =
      args.mm_audio_hidden_size() / args.mm_audio_num_attention_heads();
  std::optional<long int> optionalValue = args.mm_audio_num_attention_heads();
  param.numKeyValueHeadsPerRank =
      static_cast<int>(optionalValue.value()) / param.worldSize;
  /*
  if (padding_heads) {
      param.numAttentionHeadsPerRank = param.numAttentionHeadsPerRank + 1;
  }
  */
  param.rank = parallel_args.rank();
  param.backend = "lccl";
  param.enableLogN = false;
}

NpuQwen3AudioEncoderLayerImpl::NpuQwen3AudioEncoderLayerImpl(
    const ModelContext& context)
    : BaseLayer(context),
      model_args_(context.get_model_args()) {
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

void NpuQwen3AudioEncoderLayerImpl::verify_loaded_weights() const {
  LOG(INFO) << "start verify";
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    LOG(INFO) << index;
    LOG(INFO) << name;
    at_weight_tensors_[index].print();
    CHECK(at_weight_tensors_[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
  LOG(INFO) << "stop verify";
}

void NpuQwen3AudioEncoderLayerImpl::merge_loaded_weights() {
  // spilt pack qkv weight when enable tp
  get_weights_col_packed_qkv();
  
  LOG(INFO) << "device is " << device_;
  LOG(INFO) << encode_param_.worldSize;
  // merge qkv weight
  auto new_qkv_weight = torch::cat({at_weight_tensors_[AUDIO_IN_Q_WEIGHT],
                                    at_weight_tensors_[AUDIO_IN_K_WEIGHT],
                                    at_weight_tensors_[AUDIO_IN_V_WEIGHT]},
                                    0).to(device_);
  at_weight_tensors_[AUDIO_IN_QKV_WEIGHT] = new_qkv_weight;
  at_weight_tensors_[AUDIO_IN_Q_WEIGHT] = torch::zeros({1}).to(device_);
  at_weight_tensors_[AUDIO_IN_K_WEIGHT] = torch::zeros({1}).to(device_);
  at_weight_tensors_[AUDIO_IN_V_WEIGHT] = torch::zeros({1}).to(device_);

  // merge qkv bias
  auto new_qkv_bias = torch::cat({at_weight_tensors_[AUDIO_IN_Q_BIAS],
                                  at_weight_tensors_[AUDIO_IN_K_BIAS],
                                  at_weight_tensors_[AUDIO_IN_V_BIAS]},
                                  0).to(device_);
  
  at_weight_tensors_[AUDIO_IN_QKV_BIAS] = new_qkv_bias;
  at_weight_tensors_[AUDIO_IN_Q_BIAS] = torch::zeros({1}).to(device_);
  at_weight_tensors_[AUDIO_IN_K_BIAS] = torch::zeros({1}).to(device_);
  at_weight_tensors_[AUDIO_IN_V_BIAS] = torch::zeros({1}).to(device_);
  for(auto tensor : at_weight_tensors_)
  {
     LOG(INFO) << tensor.device();
  }
  c10_npu::NPUCachingAllocator::emptyCache();
  for (int i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    atb_weight_tensors_[i] =
        atb_speed::Utils::AtTensor2Tensor(at_weight_tensors_[i]);
  }

  init_layer();
}

void NpuQwen3AudioEncoderLayerImpl::get_weights_col_packed_qkv() {
  int rank = encode_param_.rank;
  int worldSize = encode_param_.worldSize;
  /*
  int64_t padding_heads_num = model_args_.mm_audio_num_attention_heads() % encode_param_.worldSize;
  if (padding_heads_num > 0)
  {
     int64_t padding_out = padding_heads_num * encode_param_.hiddenSizePerAttentionHead;
     padded_weight = torch::zeros(
                {padded_out_features, padded_in_features},
                weight.options());
            padded_weight.slice(0, 0, out_features)
                         .slice(1, 0, in_features) = weight;
  }
  if (padding_heads) {
     auto hidden_dim = encode
  }
  */
  // weight
  at_weight_tensors_[AUDIO_IN_Q_WEIGHT] =
      (at_weight_tensors_[AUDIO_IN_Q_WEIGHT].chunk(worldSize, 0))[rank].to(device_);
  at_weight_tensors_[AUDIO_IN_K_WEIGHT] =
      (at_weight_tensors_[AUDIO_IN_K_WEIGHT].chunk(worldSize, 0))[rank].to(device_);;
  at_weight_tensors_[AUDIO_IN_V_WEIGHT] =
      (at_weight_tensors_[AUDIO_IN_V_WEIGHT].chunk(worldSize, 0))[rank].to(device_);;
  // bias
  at_weight_tensors_[AUDIO_IN_Q_BIAS] =
      (at_weight_tensors_[AUDIO_IN_Q_BIAS].chunk(worldSize, 0))[rank].to(device_);;
  at_weight_tensors_[AUDIO_IN_K_BIAS] =
      (at_weight_tensors_[AUDIO_IN_K_BIAS].chunk(worldSize, 0))[rank].to(device_);;
  at_weight_tensors_[AUDIO_IN_V_BIAS] =
      (at_weight_tensors_[AUDIO_IN_V_BIAS].chunk(worldSize, 0))[rank].to(device_);;
}

void NpuQwen3AudioEncoderLayerImpl::load_state_dict(
    const StateDict& state_dict) {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    LOG(INFO) << index;
    LOG(INFO) << name;
    if (WEIGHT_SHARD.find(index) != WEIGHT_SHARD.end()) {
      set_weight(state_dict, name, index, WEIGHT_SHARD[index]);
    } else {
      set_weight(state_dict, name, index);
    }
  }
}

int64_t NpuQwen3AudioEncoderLayerImpl::init_layer() {
  name_ = "qwen3_audio_encoder_layer";
  model_name_ = "qwen3_audio";
  CHECK_OPERATION_STATUS_RETURN(init_node(encode_node_, encode_param_));
  return atb::NO_ERROR;
}

int64_t NpuQwen3AudioEncoderLayerImpl::init_node(
    atb_speed::Model::Node& node,
    atb_speed::qwen::AudioEncoderLayerParam& param) {
  atb::Operation* operation = nullptr;
  atb_speed::qwen::Qwen3_Audio_EncoderLayer(param, &operation);
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

torch::Tensor NpuQwen3AudioEncoderLayerImpl::forward(
    torch::Tensor& x,
    torch::Tensor& cu_seqlen,
    std::vector<int>& cu_seqlen_vec,
    ModelInputParams& input_params,
    int node_id,
    aclrtEvent* event,
    std::atomic<bool>* event_flag) {
  atb::Status st;

  build_node_variant_pack(encode_node_,
                          x,
                          cu_seqlen,
                          cu_seqlen_vec,
                          input_params,
                          true);
  // mstxRangeEnd(id);
  st = execute_node(encode_node_, node_id);
  LOG_IF(FATAL, st != 0) << model_name_
                         << "excute encode layer fail, error code: " << st;
  return x;
}

void NpuQwen3AudioEncoderLayerImpl::build_node_variant_pack(
    atb_speed::Model::Node& node,
    torch::Tensor& x,
    torch::Tensor& cu_seqlen,
    std::vector<int>& cu_seqlen_vec,
    ModelInputParams& input_params,
    bool is_prefill) {
  internal_tensors_ = atb_speed::Utils::AtTensor2Tensor(x);

  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER) = internal_tensors_;
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 1) =
      atb_speed::Utils::AtTensor2Tensor(cu_seqlen);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 1).hostData =
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
