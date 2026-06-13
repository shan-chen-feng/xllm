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

#include "qwen2_decoder_layer.h"

#if defined(USE_NPU)
#include <acl/acl.h>
#include <c10/core/Event.h>
#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#endif

#include <cstdlib>
#include <functional>

namespace xllm {
namespace layer {

#if defined(USE_NPU)
namespace {

double get_prefetch_coefficient(double default_coefficient) {
  const char* coefficient = std::getenv("PREFETCH_COEFFOCIENT");
  return coefficient ? std::atof(coefficient) : default_coefficient;
}

}  // namespace
#endif

Qwen2DecoderLayerImpl::Qwen2DecoderLayerImpl(const ModelContext& context,
                                             int32_t layer_id)
    : parallel_args_(context.get_parallel_args()) {
  const auto& model_args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& options = context.get_tensor_options();
  const std::string mlp_module_prefix =
      layer_id >= 0 ? "model.layers." + std::to_string(layer_id) + ".mlp" : "";
#if defined(USE_NPU)
  prefetch_weight_stream_ = context.get_prefetch_weight_npu_stream();
  device_index_ = options.device().index();
  enable_libtorch_weight_prefetch_ =
      model_args.model_type() == "qwen3" && prefetch_weight_stream_.has_value();
#endif

  // Initialize attention layers
  attention_ = register_module("self_attn", Qwen2Attention(context));

  // Initialize norm layers
  input_norm_ = register_module(
      "input_layernorm",
      RMSNorm(model_args.hidden_size(), model_args.rms_norm_eps(), options));

  post_norm_ = register_module(
      "post_attention_layernorm",
      RMSNorm(model_args.hidden_size(), model_args.rms_norm_eps(), options));

  // Initialize mlp
  mlp_ = register_module("mlp",
                         DenseMLP(model_args.hidden_size(),
                                  model_args.intermediate_size(),
                                  true,
                                  false,
                                  model_args.hidden_act(),
                                  /*enable_result_reduction=*/true,
                                  quant_args,
                                  parallel_args_.tp_group_,
                                  options,
                                  mlp_module_prefix));
}

void Qwen2DecoderLayerImpl::load_state_dict(const StateDict& state_dict) {
  attention_->load_state_dict(state_dict.get_dict_with_prefix("self_attn."));
  input_norm_->load_state_dict(
      state_dict.get_dict_with_prefix("input_layernorm."));
  post_norm_->load_state_dict(
      state_dict.get_dict_with_prefix("post_attention_layernorm."));
  mlp_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>>
Qwen2DecoderLayerImpl::apply_norm(
    RMSNorm& norm,
    torch::Tensor& input,
    std::optional<torch::Tensor>& residual,
    const std::optional<torch::Tensor>& fp8_scale) {
  const bool use_fp8_fusion = fp8_scale.has_value();

  if (!residual.has_value()) {
    // First layer: initialize residual from input
    auto new_residual = input;
    auto output = use_fp8_fusion
                      ? std::get<0>(norm->forward_fp8(input, fp8_scale.value()))
                      : std::get<0>(norm->forward(input));
    return {output, new_residual};
  }

  // Subsequent layers: fused add + norm
  return use_fp8_fusion ? norm->forward_fp8(input, fp8_scale.value(), residual)
                        : norm->forward(input, residual);
}

torch::Tensor Qwen2DecoderLayerImpl::forward(
    torch::Tensor& x,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    const std::optional<torch::Tensor>& next_qkv_weight) {
  auto pre_fp8_scale = attention_->get_fp8_input_scale();
  auto post_fp8_scale = mlp_->get_fp8_input_scale();
  (void)input_params;

  std::function<void()> prefetch_mlp_weight;
  std::function<void()> prefetch_next_qkv_weight;
#if defined(USE_NPU)
  const bool enable_prefetch_this_forward =
      enable_libtorch_weight_prefetch_ && !attn_metadata.is_prefill &&
      !attn_metadata.is_chunked_prefill;
  prefetch_mlp_weight = [this, enable_prefetch_this_forward]() {
    if (enable_prefetch_this_forward) {
      prefetch_weight(mlp_->gate_up_weight(), get_prefetch_coefficient(0.4));
    }
  };
  prefetch_next_qkv_weight =
      [this, enable_prefetch_this_forward, &next_qkv_weight]() {
        if (enable_prefetch_this_forward && next_qkv_weight.has_value()) {
          prefetch_weight(next_qkv_weight.value(),
                          get_prefetch_coefficient(1.0));
        }
      };
#else
  (void)next_qkv_weight;
#endif

  // Pre-attention norm
  std::tie(x, residual) = apply_norm(input_norm_, x, residual, pre_fp8_scale);

  // Attention
  x = attention_->forward(
      positions, x, attn_metadata, kv_cache, prefetch_mlp_weight);

  // Post-attention norm
  std::tie(x, residual) = apply_norm(post_norm_, x, residual, post_fp8_scale);

  // MLP
  x = mlp_->forward(x, prefetch_next_qkv_weight);

  return x;
}

#if defined(USE_NPU)
void Qwen2DecoderLayerImpl::prefetch_weight(const torch::Tensor& weight,
                                            double coefficient) const {
  if (!enable_libtorch_weight_prefetch_ || !weight.defined() ||
      coefficient <= 0.0) {
    return;
  }

  const auto prefetch_size =
      static_cast<size_t>(static_cast<double>(weight.nbytes()) * coefficient);
  if (prefetch_size == 0) {
    return;
  }

  auto main_stream = c10_npu::getCurrentNPUStream(device_index_).unwrap();
  auto side_stream = prefetch_weight_stream_.value().unwrap();

  c10::Event main_ready(main_stream.device_type());
  main_ready.record(main_stream);
  main_ready.block(side_stream);

  const aclError ret = aclrtCmoAsync(weight.data_ptr(),
                                    prefetch_size,
                                    ACL_RT_CMO_TYPE_PREFETCH,
                                    prefetch_weight_stream_.value().stream());
  CHECK_EQ(ret, ACL_SUCCESS) << "aclrtCmoAsync failed, ret=" << ret;
}
#endif

}  // namespace layer
}  // namespace xllm
