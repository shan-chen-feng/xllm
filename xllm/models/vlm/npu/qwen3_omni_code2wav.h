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

#pragma once

#include <boost/algorithm/string.hpp>
#include <string>
#include <vector>

#include "core/framework/model/npu_dp_ep_padding.h"
#include "core/framework/model_context.h"
#include "core/layers/qwen3_omni_code2wav_transformer_layer.h"

namespace xllm {

using torch::indexing::None;
using ISlice = torch::indexing::Slice;

// submodule: upsample
class Qwen3OmniCausalTransConvNextImpl : public torch::nn::Module {
 public:
  Qwen3OmniCausalTransConvNextImpl(const ModelContext& context,
                                   int in_channels,
                                   int out_channels,
                                   int kernel_size,
                                   int stride = 1) {
    conv_ =
        torch::nn::ConvTranspose1d(torch::nn::ConvTranspose1dOptions(
                                       in_channels, out_channels, kernel_size)
                                       .stride(stride));
    register_module("conv", conv_);
    auto pad = kernel_size - stride;
    left_pad_ = static_cast<int64_t>(std::ceil(static_cast<double>(pad)));
    pad = left_pad_;
    right_pad_ = left_pad_;
  }
  void load_state_dict(const StateDict& state_dict) {
    auto conv_weight = state_dict.get_tensor("weight");
    auto conv_bias = state_dict.get_tensor("bias");
    if (conv_weight.defined()) {
      conv_->weight.data().copy_(conv_weight);
      conv_weight_loaded_ = true;
    }

    if (conv_bias.defined()) {
      conv_->bias.data().copy_(conv_bias);
      conv_bias_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(conv_weight_loaded_)
        << "weight is not loaded for " << prefix + "weight";
  }

  void verify_loaded_bias(const std::string& prefix) const {
    CHECK(conv_bias_loaded_) << "bias is not loaded for " << prefix + "bias";
  }

  torch::Tensor forward(torch::Tensor x) {
    x = conv_->forward(x);
    int64_t T = x.size(-1);
    int64_t valid_length = T - left_pad_ - right_pad_;
    return x.narrow(-1, left_pad_, valid_length).contiguous();
  }

 private:
  int64_t left_pad_;
  int64_t right_pad_;
  bool conv_weight_loaded_ = false;
  bool conv_bias_loaded_ = false;
  torch::nn::ConvTranspose1d conv_{nullptr};
};
TORCH_MODULE(Qwen3OmniCausalTransConvNext);

class Qwen3OmniCausalConvNextImpl : public torch::nn::Module {
 public:
  Qwen3OmniCausalConvNextImpl(const ModelContext& context,
                              int in_channels,
                              int out_channels,
                              int kernel_size,
                              int dilations = 1,
                              int stride = 1,
                              int groups = 1) {
    dwconv_ = torch::nn::Conv1d(
        torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(0)
            .dilation(dilations)
            .groups(groups));
    stride_ = stride;
    kernel_size_ = (kernel_size - 1) * dilations + 1;
    padding_ = kernel_size - stride;
    register_module("dwconv", dwconv_);
  }
  void load_state_dict(const StateDict& state_dict) {
    auto dwconv_weight = state_dict.get_tensor("conv.weight");
    if (dwconv_weight.defined()) {
      dwconv_->weight.data().copy_(dwconv_weight);
      dwconv_weight_loaded_ = true;
    }

    auto dwconv_bias = state_dict.get_tensor("conv.bias");
    if (dwconv_bias.defined()) {
      dwconv_->bias.data().copy_(dwconv_bias);
      dwconv_bias_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(dwconv_weight_loaded_)
        << "weight is not loaded for " << prefix + "conv.weight";
    CHECK(dwconv_bias_loaded_)
        << "bias is not loaded for " << prefix + "conv.bias";
  }

  int64_t get_extra_padding_for_conv1d(torch::Tensor x) {
    int64_t length = x.size(-1);
    double n_frames =
        static_cast<double>(length - kernel_size_ + padding_) / stride_ + 1.0;
    int64_t ideal_length =
        (static_cast<int64_t>(std::ceil(n_frames)) - 1) * stride_ +
        (kernel_size_ - padding_);
    return ideal_length - length;
  }

  torch::Tensor forward(torch::Tensor x) {
    auto extra_padding = get_extra_padding_for_conv1d(x);
    x = F::pad(x,
               F::PadFuncOptions({padding_, extra_padding})
                   .mode(torch::kConstant)
                   .value(0));
    auto out = dwconv_->forward(x);
    return out.contiguous();
  }

 private:
  int kernel_size_;
  int stride_;
  int padding_;
  torch::nn::Conv1d dwconv_{nullptr};
  bool dwconv_weight_loaded_;
  bool dwconv_bias_loaded_;
};
TORCH_MODULE(Qwen3OmniCausalConvNext);

// submodule: upsample
class Qwen3OmniConvNextBlockImpl : public torch::nn::Module {
 public:
  Qwen3OmniConvNextBlockImpl(const ModelContext& context, int dim) {
    dim_ = dim;
    dwconv_ = Qwen3OmniCausalConvNext(context, dim_, dim_, 7, 1, dim_);
    norm_ = torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim}).eps(1e-6));
    pwconv1_ = torch::nn::Linear(dim_, 4 * dim_);
    pwconv2_ = torch::nn::Linear(4 * dim_, dim_);
    act_ = torch::nn::GELU();
    register_module("dwconv", dwconv_);
    register_module("norm", norm_);
    register_module("pwconv1", pwconv1_);
    register_module("pwconv2", pwconv2_);
    register_module("act", act_);
    gamma_ = register_parameter(
        "gamma", torch::full({dim}, 1e-6, torch::requires_grad(true)));
  }
  void load_state_dict(const StateDict& state_dict) {
    dwconv_->load_state_dict(state_dict.get_dict_with_prefix("dwconv."));

    auto norm_weight = state_dict.get_tensor("norm.weight");
    if (norm_weight.defined()) {
      norm_->weight.data().copy_(norm_weight);
      norm_weight_loaded_ = true;
    }
    auto norm_bias = state_dict.get_tensor("norm.bias");
    if (norm_bias.defined()) {
      norm_->bias.data().copy_(norm_bias);
      norm_bias_loaded_ = true;
    }

    auto pwconv1_weight = state_dict.get_tensor("pwconv1.weight");
    if (pwconv1_weight.defined()) {
      pwconv1_->weight.data().copy_(pwconv1_weight);
      pwconv1_weight_loaded_ = true;
    }
    auto pwconv1_bias = state_dict.get_tensor("pwconv1.bias");
    if (pwconv1_bias.defined()) {
      pwconv1_->bias.data().copy_(pwconv1_bias);
      pwconv1_bias_loaded_ = true;
    }

    auto pwconv2_weight = state_dict.get_tensor("pwconv2.weight");
    if (pwconv2_weight.defined()) {
      pwconv2_->weight.data().copy_(pwconv2_weight);
      pwconv2_weight_loaded_ = true;
    }
    auto pwconv2_bias = state_dict.get_tensor("pwconv2.bias");
    if (pwconv2_bias.defined()) {
      pwconv2_->bias.data().copy_(pwconv2_bias);
      pwconv2_bias_loaded_ = true;
    }

    auto gamma_weight = state_dict.get_tensor("gamma");
    if (gamma_weight.defined()) {
      gamma_.data().copy_(gamma_weight);
      gamma_weight_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(norm_weight_loaded_)
        << "weight is not loaded for " << prefix + "norm.weight";
    CHECK(norm_bias_loaded_)
        << "bias is not loaded for " << prefix + "norm.bias";
    CHECK(pwconv1_weight_loaded_)
        << "weight is not loaded for " << prefix + "pwconv1.weight";
    CHECK(pwconv1_bias_loaded_)
        << "bias is not loaded for " << prefix + "pwconv1.bias";
    CHECK(pwconv2_weight_loaded_)
        << "weight is not loaded for " << prefix + "pwconv2.weight";
    CHECK(pwconv2_bias_loaded_)
        << "bias is not loaded for " << prefix + "pwconv2.bias";
    CHECK(gamma_weight_loaded_)
        << "weight is not loaded for " << prefix + "gamma";
  }

  torch::Tensor forward(torch::Tensor x) {
    auto input = x;
    x = dwconv_->forward(x);
    x = x.permute({0, 2, 1});
    x = norm_->forward(x);
    x = pwconv1_->forward(x);
    x = act_->forward(x);
    x = pwconv2_->forward(x);
    x = gamma_ * x;
    x = x.permute({0, 2, 1});
    x = x + input;
    return x;
  }

 private:
  int dim_;
  Qwen3OmniCausalConvNext dwconv_{nullptr};
  torch::nn::LayerNorm norm_{nullptr};
  torch::nn::Linear pwconv1_{nullptr};
  torch::nn::Linear pwconv2_{nullptr};
  torch::nn::GELU act_{nullptr};
  torch::Tensor gamma_;
  bool norm_weight_loaded_ = false;
  bool norm_bias_loaded_ = false;
  bool pwconv1_weight_loaded_ = false;
  bool pwconv1_bias_loaded_ = false;
  bool pwconv2_weight_loaded_ = false;
  bool pwconv2_bias_loaded_ = false;
  bool gamma_weight_loaded_ = false;
};
TORCH_MODULE(Qwen3OmniConvNextBlock);

class Qwen3OmniCode2WavSnakeBetaImpl : public torch::nn::Module {
 public:
  Qwen3OmniCode2WavSnakeBetaImpl(const ModelContext& context,
                                 int64_t in_features,
                                 float alpha = 1.0) {
    // register submodules
    in_features_ = in_features;
    alpha_ = register_parameter(
        "alpha",
        torch::full(
            {in_features_}, std::log(alpha), torch::requires_grad(true)));
    beta_ = register_parameter(
        "beta", torch::full({in_features_}, 0.0, torch::requires_grad(true)));
  }
  void load_state_dict(const StateDict& state_dict) {
    auto alpha_weight = state_dict.get_tensor("alpha");
    if (alpha_weight.defined()) {
      alpha_.data().copy_(alpha_weight);
      alpha_weight_loaded_ = true;
    }

    auto beta_weight = state_dict.get_tensor("beta");
    if (beta_weight.defined()) {
      beta_.data().copy_(beta_weight);
      beta_weight_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(alpha_weight_loaded_)
        << "weight is not loaded for " << prefix + "alpha";
    CHECK(beta_weight_loaded_)
        << "weight is not loaded for " << prefix + "beta";
  }

  torch::Tensor forward(torch::Tensor x) {
    auto alpha_expanded = alpha_.view({1, -1, 1});
    auto beta_expanded = beta_.view({1, -1, 1});
    auto alpha_pos = torch::exp(alpha_expanded);
    auto beta_pos = torch::exp(beta_expanded);
    auto sin_sq = torch::pow(torch::sin(x * alpha_pos), 2);
    auto inv_beta = 1.0 / (beta_pos + no_div_by_zero);
    auto output = x + inv_beta * sin_sq;
    return output;
  }

 private:
  double no_div_by_zero = 1e-9;
  torch::Tensor alpha_;
  torch::Tensor beta_;
  int64_t in_features_;
  bool alpha_weight_loaded_ = false;
  bool beta_weight_loaded_ = false;
};
TORCH_MODULE(Qwen3OmniCode2WavSnakeBeta);

class Qwen3OmniCode2WavRmsNormImpl : public torch::nn::Module {
 public:
  Qwen3OmniCode2WavRmsNormImpl(const ModelContext& context) {
    // register submodules
    auto model_args = context.get_model_args();
    auto hidden_size = model_args.code2wav_config_hidden_size();
    variance_epsilon_ = 1e-6;
    norm_ = register_parameter("norm", torch::ones(hidden_size));
  }
  void load_state_dict(const StateDict& state_dict) {
    auto norm_weight = state_dict.get_tensor("weight");
    if (norm_weight.defined()) {
      norm_.data().copy_(norm_weight);
      norm_weight_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(norm_weight_loaded_)
        << "weight is not loaded for " << prefix + "norm";
  }

  torch::Tensor forward(torch::Tensor x) {
    auto input_dtype = x.dtype();
    x = x.to(torch::kFloat32);
    auto variance = torch::mean(torch::pow(x, 2), -1, true);
    x = x * torch::rsqrt(variance + variance_epsilon_);
    return norm_ * x.to(input_dtype);
  }

 private:
  bool norm_weight_loaded_ = false;
  torch::Tensor norm_;
  double variance_epsilon_;
};
TORCH_MODULE(Qwen3OmniCode2WavRmsNorm);

class Qwen3OmniCode2WavRoteryEmbeddingImpl : public torch::nn::Module {
 public:
  Qwen3OmniCode2WavRoteryEmbeddingImpl(const ModelContext& context)
      : device_(context.get_tensor_options().device()) {
    // register submodules
    auto model_args = context.get_model_args();
    auto base = model_args.code2wav_config_rope_theta();
    auto partial_rotary_factory = 1.0;
    auto hidden_size = model_args.code2wav_config_hidden_size();
    auto num_attention_heads = model_args.code2wav_config_num_attention_heads();
    auto head_dim = hidden_size / num_attention_heads;
    auto dim = int(head_dim * partial_rotary_factory);
    attention_scaling_ = 1.0;
    dtype_ = context.get_tensor_options().dtype().toScalarType();
    auto indices =
        torch::arange(0, dim, 2, torch::TensorOptions().dtype(torch::kInt64));
    indices = indices.to(context.get_tensor_options().device());
    auto indices_f = indices.to(torch::kFloat32);
    auto freqs = indices_f / static_cast<float>(dim);
    auto base_tensor = torch::full_like(freqs, static_cast<float>(base));
    auto base_pow = torch::pow(base_tensor, freqs);
    inv_freq_ = base_pow.reciprocal();
    register_buffer("inv_freq", inv_freq_);
  }
  void load_state_dict(const StateDict& state_dict) {}

  void verify_loaded_weights(const std::string& prefix) const {}

  std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x,
                                                  torch::Tensor position_ids) {
    auto inv_freq_expanded = inv_freq_.view({1, -1, -1})
                                 .expand({position_ids.size(0), -1, 1})
                                 .to(device_);
    auto position_ids_expanded = position_ids.unsqueeze(1).to(torch::kFloat32);
    auto freqs =
        torch::matmul(inv_freq_expanded, position_ids_expanded).transpose(1, 2);
    auto emb = torch::cat({freqs, freqs}, -1);
    auto cos = torch::cos(emb) * attention_scaling_;
    auto sin = torch::sin(emb) * attention_scaling_;
    return std::make_pair(cos.to(dtype_), sin.to(dtype_));
  }

 private:
  torch::Tensor inv_freq_;
  torch::Device device_;
  torch::ScalarType dtype_;
  float attention_scaling_;
};
TORCH_MODULE(Qwen3OmniCode2WavRoteryEmbedding);

class Qwen3OmniCode2WavPretransformerImpl : public torch::nn::Module {
 public:
  Qwen3OmniCode2WavPretransformerImpl(const ModelContext& context)
      : device_(context.get_tensor_options().device()) {
    // register submodules
    auto model_args = context.get_model_args();
    auto num_hidden_layers = model_args.code2wav_config_num_hidden_layers();
    auto sliding_window = model_args.code2wav_config_sliding_window();
    blocks_ = register_module("transformer_layer", torch::nn::ModuleList());
    layers_.reserve(model_args.code2wav_config_num_hidden_layers());
    has_sliding_layers_ = true;
    rms_norm_ = Qwen3OmniCode2WavRmsNorm(context);
    rotery_embedding_ = Qwen3OmniCode2WavRoteryEmbedding(context);
    register_module("norm", rms_norm_);
    register_module("rotery_embedding", rotery_embedding_);
    for (int32_t i = 0; i < model_args.code2wav_config_num_hidden_layers();
         i++) {
      auto block = layer::Qwen3OmniCode2WavTransformerLayer(context, i);
      layers_.push_back(block);
      blocks_->push_back(block);
    }
  }
  void load_state_dict(const StateDict& state_dict) {
    rms_norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
    rms_norm_->verify_loaded_weights(prefix + "norm.");
  }

  torch::Tensor forward(torch::Tensor x, ModelInputParams& input_params) {
    // todo x is null , use embed_tokens()
    auto pasr_seen_tokens = 0;
    auto seq_len = x.size(1);
    auto cache_position = torch::arange(
        pasr_seen_tokens,
        pasr_seen_tokens + seq_len,
        torch::TensorOptions().dtype(torch::kLong).device(device_));
    auto position_ids = cache_position.unsqueeze(0);
    auto attention_mask =
        get_silding_attention_mask(x, cache_position, position_ids);

    auto position_embeddings = rotery_embedding_->forward(x, position_ids);

    for (size_t i = 0; i < layers_.size(); i++) {
      aclrtEvent* event = nullptr;
      std::atomic<bool>* event_flag = nullptr;
      // to check
      int32_t seq_len_int = static_cast<int32_t>(seq_len);
      std::vector<int32_t> cu_seqlens_vec;
      cu_seqlens_vec.push_back(seq_len_int);
      // torch::Tensor cu_seqlens = torch::Tensor(seq_len_int).to(x.device());
      torch::Tensor cu_seqlens =
          torch::from_blob(cu_seqlens_vec.data(),
                           {static_cast<int64_t>(cu_seqlens_vec.size())},
                           torch::kInt32)
              .clone()
              .to(x.device());
      auto& layer = layers_[i];
      layer(x,
            position_embeddings.first,
            position_embeddings.second,
            attention_mask,
            cu_seqlens,
            cu_seqlens_vec,
            input_params,
            i,
            event,
            event_flag);
    }

    x = rms_norm_->forward(x);
    return x;
  }

  torch::Tensor get_silding_attention_mask(torch::Tensor x,
                                           torch::Tensor cache_position,
                                           torch::Tensor position_ids) {
    // to be implement
    return x;
  }

 private:
  Qwen3OmniCode2WavRmsNorm rms_norm_{nullptr};
  Qwen3OmniCode2WavRoteryEmbedding rotery_embedding_{nullptr};
  torch::nn::ModuleList blocks_ = nullptr;
  std::vector<layer::Qwen3OmniCode2WavTransformerLayer> layers_;
  bool has_sliding_layers_;
  torch::Device device_;
};
TORCH_MODULE(Qwen3OmniCode2WavPretransformer);

class Qwen3OmniCode2WavUnsampleImpl : public torch::nn::Module {
 public:
  Qwen3OmniCode2WavUnsampleImpl(const ModelContext& context) {
    // register submodules
    auto options = context.get_tensor_options();
    auto model_args = context.get_model_args();
    auto hidden_size = model_args.code2wav_config_hidden_size();
    auto upsampling_ratios_vec =
        model_args.code2wav_config_upsampling_ratios_vec();
    upsample_layers_ =
        register_module("upsample_layers", torch::nn::ModuleList());
    causalTransConvNext_layers_.reserve(upsampling_ratios_vec.size());
    convNextBlock_layers_.reserve(upsampling_ratios_vec.size());
    for (size_t i = 0; i < upsampling_ratios_vec.size(); i++) {
      int64_t factor = upsampling_ratios_vec[i];
      auto block_1 = Qwen3OmniCausalTransConvNext(
          context, hidden_size, hidden_size, factor, factor);
      auto block_2 = Qwen3OmniConvNextBlock(context, hidden_size);
      upsample_layers_->push_back(block_1);
      causalTransConvNext_layers_.push_back(block_1);
      upsample_layers_->push_back(block_2);
      convNextBlock_layers_.push_back(block_2);
    }
  }

  torch::Tensor forward(torch::Tensor x) {
    for (size_t i = 0; i < causalTransConvNext_layers_.size(); ++i) {
      x = causalTransConvNext_layers_[i]->forward(x);
      x = convNextBlock_layers_[i]->forward(x);
    }
    return x;
  }

  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < causalTransConvNext_layers_.size(); ++i) {
      causalTransConvNext_layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix(std::to_string(i) + ".0."));
      convNextBlock_layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix(std::to_string(i) + ".1."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    for (size_t i = 0; i < causalTransConvNext_layers_.size(); ++i) {
      causalTransConvNext_layers_[i]->verify_loaded_weights(
          prefix + std::to_string(i) + ".0.");
      convNextBlock_layers_[i]->verify_loaded_weights(
          prefix + std::to_string(i) + ".1.");
    }
  }

 private:
  torch::nn::ModuleList upsample_layers_{nullptr};
  std::vector<Qwen3OmniCausalTransConvNext> causalTransConvNext_layers_{
      nullptr};
  std::vector<Qwen3OmniConvNextBlock> convNextBlock_layers_{nullptr};
};
TORCH_MODULE(Qwen3OmniCode2WavUnsample);

class Qwen3OmniCode2WavDecoderResidualUnitImpl : public torch::nn::Module {
 public:
  Qwen3OmniCode2WavDecoderResidualUnitImpl(const ModelContext& context,
                                           int dim = 16,
                                           int dilation = 1) {
    // register submodules
    dim_ = dim;
    dilation_ = dilation;
    act1_ = register_module("act1", Qwen3OmniCode2WavSnakeBeta(context, dim_));
    conv1_ = register_module(
        "conv1", Qwen3OmniCausalConvNext(context, dim_, dim_, 7, dilation_));
    act2_ = register_module("act2", Qwen3OmniCode2WavSnakeBeta(context, dim_));
    conv2_ = register_module("conv2",
                             Qwen3OmniCausalConvNext(context, dim_, dim_, 1));
  }

  void load_state_dict(const StateDict& state_dict) {
    act1_->load_state_dict(state_dict.get_dict_with_prefix("act1."));
    conv1_->load_state_dict(state_dict.get_dict_with_prefix("conv1."));
    act2_->load_state_dict(state_dict.get_dict_with_prefix("act2."));
    conv2_->load_state_dict(state_dict.get_dict_with_prefix("conv2."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    act1_->verify_loaded_weights(prefix + "act1.");
    conv1_->verify_loaded_weights(prefix + "conv1.");
    act2_->verify_loaded_weights(prefix + "act2.");
    conv2_->verify_loaded_weights(prefix + "conv2.");
  }

  torch::Tensor forward(torch::Tensor x) {
    auto residual = x;
    x = act1_->forward(x);
    x = conv1_->forward(x);
    x = act2_->forward(x);
    x = conv2_->forward(x);
    return x + residual;
  }

 private:
  Qwen3OmniCode2WavSnakeBeta act1_{nullptr}, act2_{nullptr};
  Qwen3OmniCausalConvNext conv1_{nullptr}, conv2_{nullptr};
  int dim_;
  int dilation_;
};
TORCH_MODULE(Qwen3OmniCode2WavDecoderResidualUnit);

class Qwen3OmniCode2WavDecoderBlockImpl : public torch::nn::Module {
 public:
  Qwen3OmniCode2WavDecoderBlockImpl(const ModelContext& context,
                                    int layer_idx) {
    // register submodules
    auto model_args = context.get_model_args();
    auto decoder_dim = model_args.code2wav_config_decoder_dim();
    auto upsample_rates_vec = model_args.code2wav_config_upsample_rates_vec();
    upsample_rate_ = upsample_rates_vec[layer_idx];
    in_dim_ = decoder_dim >> layer_idx;
    out_dim_ = decoder_dim >> (layer_idx + 1);
    snk_ = Qwen3OmniCode2WavSnakeBeta(context, in_dim_);
    trans_conv_ = Qwen3OmniCausalTransConvNext(
        context, in_dim_, out_dim_, 2 * upsample_rate_, upsample_rate_);

    decoder_block_->push_back(snk_);
    decoder_block_->push_back(trans_conv_);

    decoder_residual_1_ =
        Qwen3OmniCode2WavDecoderResidualUnit(context, out_dim_, 1);
    decoder_residual_2_ =
        Qwen3OmniCode2WavDecoderResidualUnit(context, out_dim_, 3);
    decoder_residual_3_ =
        Qwen3OmniCode2WavDecoderResidualUnit(context, out_dim_, 9);
    decoder_block_->push_back(decoder_residual_1_);
    decoder_block_->push_back(decoder_residual_2_);
    decoder_block_->push_back(decoder_residual_3_);
    register_module("decoder_block", decoder_block_);
  }
  void load_state_dict(const StateDict& state_dict) {
    snk_->load_state_dict(state_dict.get_dict_with_prefix("0."));
    trans_conv_->load_state_dict(state_dict.get_dict_with_prefix("1."));
    decoder_residual_1_->load_state_dict(state_dict.get_dict_with_prefix("2."));
    decoder_residual_2_->load_state_dict(state_dict.get_dict_with_prefix("3."));
    decoder_residual_3_->load_state_dict(state_dict.get_dict_with_prefix("4."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    snk_->verify_loaded_weights(prefix + "0.");
    trans_conv_->verify_loaded_weights(prefix + "1.");
    decoder_residual_1_->verify_loaded_weights(prefix + "2.");
    decoder_residual_2_->verify_loaded_weights(prefix + "3.");
    decoder_residual_3_->verify_loaded_weights(prefix + "4.");
  }

  torch::Tensor forward(torch::Tensor x) {
    x = snk_->forward(x);
    x = trans_conv_->forward(x);
    x = decoder_residual_1_->forward(x);
    x = decoder_residual_2_->forward(x);
    x = decoder_residual_3_->forward(x);
    return x;
  }

 private:
  int in_dim_;
  int out_dim_;
  int upsample_rate_;
  Qwen3OmniCode2WavSnakeBeta snk_{nullptr};
  Qwen3OmniCausalTransConvNext trans_conv_{nullptr};
  Qwen3OmniCode2WavDecoderResidualUnit decoder_residual_1_{nullptr};
  Qwen3OmniCode2WavDecoderResidualUnit decoder_residual_2_{nullptr};
  Qwen3OmniCode2WavDecoderResidualUnit decoder_residual_3_{nullptr};
  torch::nn::ModuleList decoder_block_{nullptr};
};
TORCH_MODULE(Qwen3OmniCode2WavDecoderBlock);

class Qwen3OmniCode2WavDecoderImpl : public torch::nn::Module {
 public:
  Qwen3OmniCode2WavDecoderImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto hidden_size = model_args.code2wav_config_hidden_size();
    auto decoder_dim = model_args.code2wav_config_decoder_dim();
    auto upsample_rates_vec = model_args.code2wav_config_upsample_rates_vec();

    casual_conv_1_ =
        Qwen3OmniCausalConvNext(context, hidden_size, decoder_dim, 7);
    decoder_block_1_ =
        Qwen3OmniCode2WavDecoderBlock(context, upsample_rates_vec[0]);
    decoder_block_2_ =
        Qwen3OmniCode2WavDecoderBlock(context, upsample_rates_vec[1]);
    decoder_block_3_ =
        Qwen3OmniCode2WavDecoderBlock(context, upsample_rates_vec[2]);
    decoder_block_4_ =
        Qwen3OmniCode2WavDecoderBlock(context, upsample_rates_vec[3]);

    int output_dim = decoder_dim >> upsample_rates_vec.size();  // or / (1 << N)
    snk_beta_ = Qwen3OmniCode2WavSnakeBeta(context, output_dim);
    casual_conv_2_ = Qwen3OmniCausalConvNext(context, output_dim, 1, 7);
    decoder_layer_->push_back(casual_conv_1_);
    decoder_layer_->push_back(decoder_block_1_);
    decoder_layer_->push_back(decoder_block_2_);
    decoder_layer_->push_back(decoder_block_3_);
    decoder_layer_->push_back(decoder_block_4_);
    decoder_layer_->push_back(snk_beta_);
    decoder_layer_->push_back(casual_conv_2_);

    register_module("decoder_layer", decoder_layer_);
  }

  void load_state_dict(const StateDict& state_dict) {
    casual_conv_1_->load_state_dict(state_dict.get_dict_with_prefix("0."));
    decoder_block_1_->load_state_dict(
        state_dict.get_dict_with_prefix("1.block"));
    decoder_block_2_->load_state_dict(
        state_dict.get_dict_with_prefix("2.block"));
    decoder_block_3_->load_state_dict(
        state_dict.get_dict_with_prefix("3.block"));
    decoder_block_4_->load_state_dict(
        state_dict.get_dict_with_prefix("4.block"));
    snk_beta_->load_state_dict(state_dict.get_dict_with_prefix("5."));
    casual_conv_2_->load_state_dict(state_dict.get_dict_with_prefix("6."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    casual_conv_1_->verify_loaded_weights(prefix + "0.");
    decoder_block_1_->verify_loaded_weights(prefix + "1.block");
    decoder_block_2_->verify_loaded_weights(prefix + "2.block");
    decoder_block_3_->verify_loaded_weights(prefix + "3.block");
    decoder_block_4_->verify_loaded_weights(prefix + "4.block");
    snk_beta_->verify_loaded_weights(prefix + "5.");
    casual_conv_2_->verify_loaded_weights(prefix + "6.");
  }

  torch::Tensor forward(torch::Tensor x) {
    x = casual_conv_1_->forward(x);
    x = decoder_block_1_->forward(x);
    x = decoder_block_2_->forward(x);
    x = decoder_block_3_->forward(x);
    x = decoder_block_4_->forward(x);
    x = snk_beta_->forward(x);
    x = casual_conv_2_->forward(x);
    return x;
  }

 private:
  torch::nn::ModuleList decoder_layer_{nullptr};
  Qwen3OmniCausalConvNext casual_conv_1_{nullptr}, casual_conv_2_{nullptr};
  Qwen3OmniCode2WavSnakeBeta snk_beta_{nullptr};
  Qwen3OmniCode2WavDecoderBlock decoder_block_1_{nullptr},
      decoder_block_2_{nullptr}, decoder_block_3_{nullptr},
      decoder_block_4_{nullptr};
};

TORCH_MODULE(Qwen3OmniCode2WavDecoder);

class Qwen3OmniCode2WavImpl : public torch::nn::Module {
 public:
  Qwen3OmniCode2WavImpl(const ModelContext& context) {
    // register submodules
    options_ = context.get_tensor_options();
    auto model_args = context.get_model_args();
    auto codebook_size = model_args.code2wav_config_codebook_size();
    auto num_quantizers = model_args.code2wav_config_num_quantizers();
    auto hidden_size = model_args.code2wav_config_hidden_size();
    auto upsampling_ratios_vec =
        model_args.code2wav_config_upsampling_ratios_vec();
    auto upsample_rates_vec = model_args.code2wav_config_upsample_rates_vec();
    total_upsample_ = 1;
    for (int64_t ratio : upsampling_ratios_vec) {
      total_upsample_ *= ratio;
    }
    for (int64_t rate : upsample_rates_vec) {
      total_upsample_ *= rate;
    }

    code_embedding_ = register_module(
        "code_embedding",
        torch::nn::Embedding(codebook_size * num_quantizers, hidden_size));
    code_embedding_->weight.set_data(code_embedding_->weight.to(options_));

    code_offset_ = (torch::arange(num_quantizers, torch::kLong) * codebook_size)
                       .view({1, -1, 1});

    pretransformer_ = register_module("pre_transformer",
                                      Qwen3OmniCode2WavPretransformer(context));
    unsample_layer_ =
        register_module("unsample", Qwen3OmniCode2WavUnsample(context));
    decoder_layer_ =
        register_module("decoder", Qwen3OmniCode2WavDecoder(context));
  }
  torch::Tensor forward(const torch::Tensor& input,
                        ModelInputParams& input_params) {
    auto x = input;
    auto state_dict = StateDictFromSafeTensor::load(
        "/export/home/shanchenfeng/xllm_build/xllm_qwen_embed/qwen_omni_code/"
        "input_featss.pt");
    auto input_featss = torch::ones({16, 156});
    bool is_conv_out_weight_loaded_ = false;
    weight::load_weight(
        *state_dict, "input_feat", input_featss, is_conv_out_weight_loaded_);
    input_featss = input_featss.to(options_);
    x = input_featss;

    auto hidden = code_embedding_->forward(x + code_offset_);
    hidden = hidden.mean(1);
    hidden = pretransformer_->forward(hidden, input_params);
    hidden = hidden.permute({0, 2, 1});
    hidden = unsample_layer_->forward(hidden);
    hidden = decoder_layer_->forward(hidden);
    return torch::clamp(hidden, -1.0, 1.0);
  }
  void load_state_dict(const StateDict& state_dict) {
    auto emb_weight = state_dict.get_tensor("code_embedding.weight");
    if (emb_weight.defined()) {
      code_embedding_->weight.data().copy_(emb_weight);
      code_embedding_weight_loaded_ = true;
    }
    pretransformer_->load_state_dict(
        state_dict.get_dict_with_prefix("pre_transformer."));
    unsample_layer_->load_state_dict(
        state_dict.get_dict_with_prefix("upsample."));
    decoder_layer_->load_state_dict(
        state_dict.get_dict_with_prefix("decoder."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(code_embedding_weight_loaded_)
        << "weight is not loaded for " << prefix + "code_embedding.weight";
    pretransformer_->verify_loaded_weights(prefix + "pre_transformer.");
    unsample_layer_->verify_loaded_weights(prefix + "upsample.");
    decoder_layer_->verify_loaded_weights(prefix + "decoder.");
  }

  torch::Tensor chunked_decode(const torch::Tensor& codes,
                               ModelInputParams& input_params,
                               int64_t chunk_size = 300,
                               int64_t left_context_size = 25) {
    TORCH_CHECK(codes.dim() >= 2, "codes must have at least 2 dim");
    int64_t total_length = codes.size(-1);
    std::vector<torch::Tensor> wavs;
    int64_t start_index = 0;

    while (start_index < total_length) {
      int64_t end_index = std::min(start_index + chunk_size, total_length);
      int64_t context_start = (start_index >= left_context_size)
                                  ? start_index - left_context_size
                                  : 0;
      int64_t context_size = start_index - context_start;

      std::vector<torch::indexing::TensorIndex> slice_indices;
      for (int64_t i = 0; i < codes.dim() - 1; ++i) {
        slice_indices.push_back(torch::indexing::Slice());
      }
      slice_indices.push_back(torch::indexing::Slice(context_start, end_index));
      auto codes_chunk = codes.index(slice_indices);
      auto wav_chunk = forward(codes_chunk, input_params);

      int64_t output_context_samples = context_size * total_upsample_;
      std::vector<torch::indexing::TensorIndex> output_slice;
      for (int64_t i = 0; i < wav_chunk.dim() - 1; ++i) {
        output_slice.push_back(torch::indexing::Slice());
      }
      output_slice.push_back(
          torch::indexing::Slice(output_context_samples, None));
      auto wav_cropped = wav_chunk.index(output_slice);

      wavs.push_back(wav_cropped);
      start_index = end_index;
    }
    return torch::cat(wavs, -1);
  }

 private:
  Qwen3OmniCode2WavPretransformer pretransformer_{nullptr};
  Qwen3OmniCode2WavUnsample unsample_layer_{nullptr};
  Qwen3OmniCode2WavDecoder decoder_layer_{nullptr};
  torch::nn::Embedding code_embedding_{nullptr};
  torch::Tensor code_offset_;
  bool code_embedding_weight_loaded_ = false;
  int64_t total_upsample_;
  torch::TensorOptions options_;
};
TORCH_MODULE(Qwen3OmniCode2Wav);

}  // namespace xllm
