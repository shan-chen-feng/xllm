/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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
#include <glog/logging.h>
#include <torch/nn/functional/linear.h>
#include <torch/torch.h>
#include <torch_npu/csrc/aten/CustomFunctions.h>
#include <torch_npu/csrc/libs/init_npu.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/framework/dit_cache/dit_cache.h"
#include "core/framework/dit_model_loader.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/parallel_state/parallel_state.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/state_dict/utils.h"
#include "dit_linear.h"
#include "framework/model_context.h"
#include "models/model_registry.h"

#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

namespace xllm {

inline torch::Tensor gather_sequence(const torch::Tensor& input_,
                                     int world_size,
                                     int64_t dim,
                                     int64_t pad,
                                     ParallelArgs pg) {
  auto input = input_.contiguous();
  if (world_size == 1) {
    return input;
  }

  // all gather
  auto tensor_list = parallel_state::all_gather(input, pg.process_group_);

  // concat
  auto output = torch::cat(tensor_list, dim);

  if (pad > 0) {
    output = output.narrow(dim, 0, output.size(dim) - pad);
  }

  return output;
}

inline torch::Tensor split_sequence(const torch::Tensor& input_,
                                    int64_t world_size,
                                    int64_t rank,
                                    int64_t dim,
                                    int64_t pad) {
  if (world_size == 1) {
    return input_;
  }

  torch::Tensor input = input_;
  torch::save(input.cpu(), "sp/input_rank_" + std::to_string(rank) + ".pt");
  if (pad > 0) {
    LOG(INFO) << "split_sequence: dim=" << dim << ", pad=" << pad;
    std::vector<int64_t> pad_size(input.sizes().begin(), input.sizes().end());
    pad_size[dim] = pad;
    input = torch::cat(
        {input,
         torch::zeros(pad_size,
                      torch::dtype(input.dtype()).device(input.device()))},
        dim);
  }

  auto tensor_list = torch::chunk(input, world_size, dim);
  auto output = tensor_list[rank].contiguous();
  torch::save(output.cpu(), "sp/output_rank_" + std::to_string(rank) + ".pt");

  return output;
}

inline torch::Tensor pad_sequence(const torch::Tensor& input_,
                                  int64_t dim,
                                  int64_t pad) {
  torch::Tensor input = input_;
  if (pad > 0) {
    std::vector<int64_t> pad_size(input.sizes().begin(), input.sizes().end());
    pad_size[dim] = pad;
    input = torch::cat(
        {input,
         torch::zeros(pad_size,
                      torch::dtype(input.dtype()).device(input.device()))},
        dim);
  }

  auto output = input.contiguous();
  return output;
}

inline torch::Tensor unpad_sequence(const torch::Tensor& input_,
                                    int64_t dim,
                                    int64_t pad) {
  if (pad > 0) {
    auto output = input_.narrow(dim, 0, input_.size(dim) - pad);
    return output;
  }
  return input_;
}
namespace qwenimage {

// TODO: This class should be extracted from dit class and integrated into a
// common class.
class RMSNormImpl : public torch::nn::Module {
 public:
  // Constructor: dim (normalization dimension), eps (stabilization term)
  // elementwise_affine (enable affine transform), bias (enable bias term)
  RMSNormImpl(int64_t dim, double eps, bool elementwise_affine, bool bias)
      : eps_(eps), elementwise_affine_(elementwise_affine), is_bias_(bias) {
    if (elementwise_affine_) {
      weight_ = register_parameter("weight", torch::ones({dim}));
      if (is_bias_) {
        bias_ = register_parameter("bias", torch::zeros({dim}));
      }
    }
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    auto [output, rstd] =
        at_npu::native::custom_ops::npu_rms_norm(hidden_states, weight_, eps_);
    if (is_bias_ && bias_.defined()) {
      output = output + bias_;
    }
    return output;
  }

  void load_state_dict(const StateDict& state_dict) {
    if (elementwise_affine_) {
      weight::load_weight(state_dict, "weight", weight_, weight_is_loaded_);
      if (is_bias_) {
        weight::load_weight(state_dict, "bias", bias_, bias_is_loaded_);
      }
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(weight_is_loaded_)
        << "weight is not loaded for " << prefix + "weight";
    CHECK(!is_bias_ || bias_is_loaded_)
        << "bias is not loaded for " << prefix + "bias";
  }

 private:
  double eps_;               // Small epsilon to avoid division by zero
  bool elementwise_affine_;  // Whether to apply learnable affine parameters
  torch::Tensor weight_;     // Learnable scale parameter
  torch::Tensor bias_;       // Learnable bias parameter (optional)
  bool is_bias_;
  bool weight_is_loaded_{false};
  bool bias_is_loaded_{false};
};
TORCH_MODULE(RMSNorm);

// TODO: This class should be extracted from dit class and integrated into a
// common class.
class AdaLayerNormContinuousImpl : public torch::nn::Module {
 public:
  explicit AdaLayerNormContinuousImpl(const ModelContext& context,
                                      int64_t embedding_dim,
                                      int64_t conditioning_embedding_dim,
                                      bool elementwise_affine = true,
                                      double eps = 1e-5,
                                      bool bias = true)
      : options_(context.get_tensor_options()) {
    ModelArgs model_args = context.get_model_args();
    silu_ = register_module("silu", torch::nn::SiLU());
    linear_ = register_module(
        "linear",
        DiTLinear(conditioning_embedding_dim, 2 * embedding_dim, bias));
    norm_ = register_module(
        "norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({embedding_dim})
                                 .elementwise_affine(false)
                                 .eps(eps)));
  }

  torch::Tensor forward(const torch::Tensor& x,
                        const torch::Tensor& conditioning_embedding) {
    auto cond_emb = silu_->forward(conditioning_embedding);
    cond_emb = cond_emb.to(x.dtype());

    auto emb = linear_->forward(cond_emb);
    auto chunks = torch::chunk(emb, 2, 1);
    torch::Tensor scale, shift;

    scale = chunks[0];
    shift = chunks[1];
    auto x_norm = norm_->forward(x);
    return x_norm * (1 + scale).unsqueeze(1) + shift.unsqueeze(1);
  }

  void load_state_dict(const StateDict& state_dict) {
    //  linear
    linear_->load_state_dict(state_dict.get_dict_with_prefix("linear."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    linear_->verify_loaded_weights(prefix + "linear.");
  }

 private:
  DiTLinear linear_{nullptr};
  torch::nn::SiLU silu_{nullptr};
  torch::nn::LayerNorm norm_{nullptr};
  double eps_;
  std::string norm_type_;
  bool elementwise_affine_;
  torch::Tensor rms_scale_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(AdaLayerNormContinuous);

// TODO: This class should be extracted from dit class and integrated into a
// common class.
class AdaLayerNormImpl : public torch::nn::Module {
 public:
  AdaLayerNormImpl(const ModelContext& contex,
                   int64_t hidden_size,
                   double eps = 1e-6)
      : hidden_size_(hidden_size), eps_(eps) {}

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& x,
      const torch::Tensor& mod_params,
      const torch::Tensor& index = torch::Tensor()) {
    auto chunks = mod_params.chunk(3, -1);
    auto shift = chunks[0];
    auto scale = chunks[1];
    auto gate = chunks[2];

    torch::Tensor shift_result, scale_result, gate_result;

    if (index.defined()) {
      // Assuming mod_params batch dim is 2*actual_batch (chunked into 2 parts)
      // So shift, scale, gate have shape [2*actual_batch, d]
      int64_t actual_batch = shift.size(0) / 2;

      // Split into two parts
      auto shift_0 = shift.slice(0, 0, actual_batch);
      auto shift_1 = shift.slice(0, actual_batch, shift.size(0));

      auto scale_0 = scale.slice(0, 0, actual_batch);
      auto scale_1 = scale.slice(0, actual_batch, scale.size(0));

      auto gate_0 = gate.slice(0, 0, actual_batch);
      auto gate_1 = gate.slice(0, actual_batch, gate.size(0));

      // index: [b, l] where b is actual batch size
      // Expand to [b, l, 1] to match feature dimension
      auto index_expanded = index.unsqueeze(-1);  // [b, l, 1]

      // Expand chunks to [b, 1, d] then broadcast to [b, l, d]
      auto shift_0_exp = shift_0.unsqueeze(1);  // [b, 1, d]
      auto shift_1_exp = shift_1.unsqueeze(1);  // [b, 1, d]
      auto scale_0_exp = scale_0.unsqueeze(1);
      auto scale_1_exp = scale_1.unsqueeze(1);
      auto gate_0_exp = gate_0.unsqueeze(1);
      auto gate_1_exp = gate_1.unsqueeze(1);

      // Use torch::where to select based on index
      shift_result =
          torch::where(index_expanded == 0, shift_0_exp, shift_1_exp);
      scale_result =
          torch::where(index_expanded == 0, scale_0_exp, scale_1_exp);
      gate_result = torch::where(index_expanded == 0, gate_0_exp, gate_1_exp);
    } else {
      shift_result = shift.unsqueeze(1);
      scale_result = scale.unsqueeze(1);
      gate_result = gate.unsqueeze(1);
    }

    scale_result = 1 + scale_result;

    auto result = at_npu::native::custom_ops::npu_layer_norm_eval(
        x, {hidden_size_}, scale_result, shift_result, eps_);

    return std::make_tuple(result, gate_result);
  }

 private:
  double eps_;
  int64_t hidden_size_;
};
TORCH_MODULE(AdaLayerNorm);

torch::Tensor apply_rotary_emb_qwen(const torch::Tensor& x,
                                    const torch::Tensor& freqs_cis,
                                    bool use_real = true,
                                    int64_t use_real_unbind_dim = -1) {
  auto cos = torch::real(freqs_cis);
  auto sin = torch::imag(freqs_cis);

  int64_t seqlen = cos.size(0);

  auto cos_expanded = cos.unsqueeze(0)
                          .unsqueeze(2)
                          .unsqueeze(-1)
                          .expand({-1, -1, -1, -1, 2})
                          .reshape({1, seqlen, 1, -1});
  auto sin_expanded = sin.unsqueeze(0)
                          .unsqueeze(2)
                          .unsqueeze(-1)
                          .expand({-1, -1, -1, -1, 2})
                          .reshape({1, seqlen, 1, -1});

  auto x_out = at_npu::native::custom_ops::npu_rotary_mul(
      x, cos_expanded, sin_expanded, "interleave");
  return x_out.to(x.dtype());
}

class TimestepsImpl : public torch::nn::Module {
 public:
  TimestepsImpl(const ModelContext& context,
                int64_t num_channels,
                bool flip_sin_to_cos,
                double downscale_freq_shift,
                double scale,
                int64_t max_period = 10000)
      : embedding_dim_(num_channels),
        flip_sin_to_cos_(flip_sin_to_cos),
        downscale_freq_shift_(downscale_freq_shift),
        scale_(scale),
        max_period_(max_period) {}

  torch::Tensor forward(const torch::Tensor& timesteps) {
    CHECK(timesteps.dim() == 1) << "Timesteps should be a 1d-array";

    int64_t half_dim = embedding_dim_ / 2;

    auto exponent =
        -std::log(max_period_) * torch::arange(0,
                                               half_dim,
                                               torch::TensorOptions()
                                                   .dtype(torch::kFloat32)
                                                   .device(timesteps.device()));

    exponent = exponent / (half_dim - downscale_freq_shift_);
    auto emb = torch::exp(exponent);
    emb = timesteps.unsqueeze(1).to(torch::kFloat) * emb.unsqueeze(0);

    emb = scale_ * emb;

    // concat sine and cosine embeddings
    auto sin_emb = torch::sin(emb);
    auto cos_emb = torch::cos(emb);
    emb = torch::cat({sin_emb, cos_emb}, /*dim=*/-1);
    // flip sine and cosine embeddings
    if (flip_sin_to_cos_) {
      emb = torch::cat({cos_emb, sin_emb}, /*dim=*/-1);
    }
    // zero pad
    if (embedding_dim_ % 2 == 1) {
      emb = torch::nn::functional::pad(
          emb, torch::nn::functional::PadFuncOptions({0, 1}));
    }
    return emb;
  }

 private:
  int64_t embedding_dim_;
  int64_t max_period_;
  bool flip_sin_to_cos_;
  double scale_;
  double downscale_freq_shift_;
};
TORCH_MODULE(Timesteps);

// TODO: a factory function that provides activation functions based on string
// input
std::function<torch::Tensor(const torch::Tensor&)> get_activation(
    const std::string& act_fn) {
  if (act_fn == "silu") {
    return [](const torch::Tensor& x) { return torch::silu(x); };
  } else if (act_fn == "relu") {
    return [](const torch::Tensor& x) { return torch::relu(x); };
  } else if (act_fn == "gelu") {
    return [](const torch::Tensor& x) { return torch::gelu(x); };
  } else if (act_fn == "tanh") {
    return [](const torch::Tensor& x) { return torch::tanh(x); };
  } else if (act_fn == "sigmoid") {
    return [](const torch::Tensor& x) { return torch::sigmoid(x); };
  } else if (act_fn == "none" || act_fn.empty()) {
    return [](const torch::Tensor& x) { return x; };
  } else {
    LOG(ERROR) << "Unsupported activation function: " << act_fn;
    throw std::out_of_range(
        "activation function out of range, given activation function:  " +
        act_fn);
  }
}

class TimestepEmbeddingImpl : public torch::nn::Module {
 public:
  TimestepEmbeddingImpl(const ModelContext& context,
                        int64_t in_channels,
                        int64_t time_embed_dim,
                        const std::string& act_fn = "silu",
                        int64_t out_dim = -1,
                        const std::string& post_act_fn = "",
                        int64_t cond_proj_dim = -1,
                        bool sample_proj_bias = true) {
    linear_1_ = register_module(
        "linear_1", DiTLinear(in_channels, time_embed_dim, sample_proj_bias));

    if (cond_proj_dim > 0) {
      cond_proj_ = register_module(
          "cond_proj", DiTLinear(cond_proj_dim, in_channels, false));
    }

    act_fn_ = register_module("act_fn", torch::nn::SiLU());

    int64_t time_embed_dim_out = (out_dim > 0) ? out_dim : time_embed_dim;

    linear_2_ = register_module(
        "linear_2",
        DiTLinear(time_embed_dim, time_embed_dim_out, sample_proj_bias));
  }

  torch::Tensor forward(const torch::Tensor& sample,
                        const torch::Tensor& condition = torch::Tensor()) {
    torch::Tensor x = sample;

    if (cond_proj_) {
      x = x + cond_proj_->forward(condition);
    }
    x = linear_1_->forward(x);
    x = act_fn_(x);
    x = linear_2_->forward(x);

    return x;
  }

  void load_state_dict(const StateDict& state_dict) {
    // linear1
    linear_1_->load_state_dict(state_dict.get_dict_with_prefix("linear_1."));
    // linear2
    linear_2_->load_state_dict(state_dict.get_dict_with_prefix("linear_2."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    linear_1_->verify_loaded_weights(prefix + "linear_1.");
    linear_2_->verify_loaded_weights(prefix + "linear_2.");
  }

 private:
  torch::nn::SiLU act_fn_{nullptr};

  DiTLinear linear_1_{nullptr};
  DiTLinear linear_2_{nullptr};
  DiTLinear cond_proj_{nullptr};
};
TORCH_MODULE(TimestepEmbedding);

std::tuple<int64_t, std::optional<torch::Tensor>, std::optional<torch::Tensor>>
compute_text_seq_len_from_mask(
    const torch::Tensor& encoder_hidden_states,
    const std::optional<torch::Tensor>& encoder_hidden_states_mask) {
  auto batch_size = encoder_hidden_states.size(0);
  auto text_seq_len = encoder_hidden_states.size(1);

  if (!encoder_hidden_states_mask.has_value()) {
    return std::make_tuple(text_seq_len, std::nullopt, std::nullopt);
  }

  auto mask =
      encoder_hidden_states_mask.value().to(encoder_hidden_states.device());

  if (mask.size(0) != batch_size || mask.size(1) != text_seq_len) {
    LOG(ERROR) << "`encoder_hidden_states_mask` shape " << mask.sizes()
               << " must match (batch_size, text_seq_len)=(" << batch_size
               << ", " << text_seq_len << ").";
  }

  if (mask.dtype() != torch::kBool) {
    mask = mask.to(torch::kBool);
  }

  auto device = encoder_hidden_states.device();
  auto position_ids = torch::arange(
      text_seq_len, torch::TensorOptions().device(device).dtype(torch::kLong));

  // Compute active positions (use position ID where mask is True, else 0)
  auto zero_tensor = torch::zeros(
      {}, torch::TensorOptions().device(device).dtype(torch::kLong));

  auto active_positions = torch::where(mask, position_ids, zero_tensor);

  // Check which samples have active positions
  auto has_active = mask.any(/*dim=*/1);

  // Compute per-sample length: max position + 1 if active, else use full length
  auto max_positions = std::get<0>(active_positions.max(/*dim=*/1));
  auto per_sample_len = torch::where(
      has_active,
      max_positions + 1,
      torch::tensor(text_seq_len,
                    torch::TensorOptions().device(device).dtype(torch::kLong)));

  return std::make_tuple(text_seq_len, per_sample_len, mask);
}

class QwenTimestepProjEmbeddingsImpl : public torch::nn::Module {
 public:
  QwenTimestepProjEmbeddingsImpl(const ModelContext& context,
                                 int64_t embedding_dim,
                                 bool use_additional_t_cond = false)
      : use_additional_t_cond_(use_additional_t_cond) {
    time_proj_ = register_module("time_proj",
                                 Timesteps(context,
                                           /*num_channels=*/256,
                                           /*flip_sin_to_cos=*/true,
                                           /*downscale_freq_shift=*/0.0,
                                           /*scale=*/1000));
    timestep_embedder_ =
        register_module("timestep_embedder",
                        TimestepEmbedding(context,
                                          /*in_channels=*/256,
                                          /*time_embed_dim*/ embedding_dim));
    if (use_additional_t_cond) {
      addition_t_embedding_ =
          register_module("addition_t_embedding",
                          torch::nn::Embedding(torch::nn::EmbeddingOptions(
                              /*num=*/2, embedding_dim)));
    }
  }

  torch::Tensor forward(
      const torch::Tensor& timestep,
      const torch::Tensor& hidden_states,
      const torch::Tensor& addition_t_cond = torch::Tensor()) {
    auto timesteps_proj = time_proj_->forward(timestep);
    auto timesteps_emb =
        timestep_embedder_->forward(timesteps_proj.to(hidden_states.dtype()));

    torch::Tensor conditioning = timesteps_emb;
    if (use_additional_t_cond_) {
      CHECK(addition_t_cond.defined())
          << "expected to pass addition_t_cond when"
          << " use_additional_t_cond_ is setup to true";
      auto addition_t_emb = addition_t_embedding_->forward(addition_t_cond);
      addition_t_emb = addition_t_emb.to(hidden_states.dtype());
      conditioning = conditioning + addition_t_emb;
    }

    return conditioning;
  }
  void load_state_dict(const StateDict& state_dict) {
    timestep_embedder_->load_state_dict(
        state_dict.get_dict_with_prefix("timestep_embedder."));
    if (use_additional_t_cond_) {
      weight::load_weight(state_dict,
                          "addition_t_embedding.weight",
                          addition_t_embedding_->weight,
                          weight_is_loaded_);
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    timestep_embedder_->verify_loaded_weights(prefix + "timestep_embedder.");
    if (use_additional_t_cond_) {
      CHECK(weight_is_loaded_)
          << "weight is not loaded for " << prefix + "weight";
    }
  }

 private:
  Timesteps time_proj_{nullptr};
  TimestepEmbedding timestep_embedder_{nullptr};
  torch::nn::Embedding addition_t_embedding_{nullptr};
  bool use_additional_t_cond_;
  bool weight_is_loaded_{false};
};
TORCH_MODULE(QwenTimestepProjEmbeddings);

class QwenEmbedRopeImpl : public torch::nn::Module {
 public:
  QwenEmbedRopeImpl(const ModelContext& context,
                    int64_t theta,
                    std::vector<int64_t> axes_dim,
                    bool scale_rope = false)
      : theta_(theta), axes_dim_(axes_dim), scale_rope_(scale_rope) {
    auto pos_index = torch::arange(4096);
    auto neg_index = torch::arange(4096).flip(0) * -1 - 1;

    pos_freqs_ = torch::cat({rope_params(pos_index, axes_dim[0], theta),
                             rope_params(pos_index, axes_dim[1], theta),
                             rope_params(pos_index, axes_dim[2], theta)},
                            1);

    neg_freqs_ = torch::cat({rope_params(neg_index, axes_dim[0], theta),
                             rope_params(neg_index, axes_dim[1], theta),
                             rope_params(neg_index, axes_dim[2], theta)},
                            1);
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const std::vector<std::vector<int64_t>>& video_fhw,
      const std::optional<int64_t>& txt_seq_lens,
      torch::Device device,
      const std::optional<int64_t>& max_txt_seq_len) {
    if (pos_freqs_.device() != device) {
      pos_freqs_ = pos_freqs_.to(device);
      neg_freqs_ = neg_freqs_.to(device);
    }

    std::vector<torch::Tensor> vid_freqs;
    int64_t max_vid_index = 0;

    for (size_t idx = 0; idx < video_fhw.size(); idx++) {
      const auto& fhw = video_fhw[idx];
      int64_t frame = fhw[0], height = fhw[1], width = fhw[2];

      std::string rope_key = std::to_string(idx) + "_" +
                             std::to_string(height) + "_" +
                             std::to_string(width);

      auto video_freq = _compute_video_freqs(frame, height, width, idx, device);
      vid_freqs.push_back(video_freq);

      if (scale_rope_) {
        max_vid_index = std::max({height / 2, width / 2, max_vid_index});
      } else {
        max_vid_index = std::max({height, width, max_vid_index});
      }
    }

    int64_t max_len;
    if (txt_seq_lens.has_value() && !max_txt_seq_len.has_value()) {
      max_len = txt_seq_lens.value();
    } else if (max_txt_seq_len.has_value()) {
      max_len = max_txt_seq_len.value();
    } else {
      LOG(ERROR) << "need to pass txt_seq_lens or max_txt_seq_len "
                 << "to calculate the mrope";
    }

    auto txt_freqs =
        pos_freqs_.slice(0, max_vid_index, max_vid_index + max_len);
    auto vid_freqs_cat = torch::cat(vid_freqs, 0);
    return std::make_tuple(vid_freqs_cat, txt_freqs);
  }

 protected:
  torch::Tensor rope_params(const torch::Tensor& index,
                            int64_t dim,
                            int64_t theta) {
    CHECK(dim % 2 == 0) << "dim must be even";

    auto exponents =
        torch::arange(
            0, dim, 2, torch::TensorOptions().dtype(torch::kFloat32)) /
        static_cast<float>(dim);
    auto freqs = 1.0 / torch::pow(theta, exponents);

    auto outer_result = torch::outer(index.to(torch::kFloat32), freqs);

    auto complex_freqs =
        torch::polar(torch::ones_like(outer_result), outer_result);

    return complex_freqs;
  }

  torch::Tensor _compute_video_freqs(int64_t frame,
                                     int64_t height,
                                     int64_t width,
                                     int64_t idx,
                                     torch::Device device) {
    int64_t seq_lens = frame * height * width;

    auto pos_freqs = pos_freqs_.to(device);
    auto neg_freqs = neg_freqs_.to(device);

    std::vector<int64_t> split_sizes;
    for (auto dim : axes_dim_) {
      split_sizes.push_back(dim / 2);
    }

    auto freqs_pos_chunks = pos_freqs_.split_with_sizes(split_sizes, 1);
    auto freqs_neg_chunks = neg_freqs_.split_with_sizes(split_sizes, 1);

    auto freqs_frame = freqs_pos_chunks[0]
                           .slice(0, idx, idx + frame)
                           .view({frame, 1, 1, -1})
                           .expand({frame, height, width, -1});

    torch::Tensor freqs_height, freqs_width;
    if (scale_rope_) {
      auto height_neg_part = freqs_neg_chunks[1].slice(
          0, -(height - height / 2), torch::indexing::None);
      auto height_pos_part = freqs_pos_chunks[1].slice(0, 0, height / 2);
      freqs_height = torch::cat({height_neg_part, height_pos_part}, 0)
                         .view({1, height, 1, -1})
                         .expand({frame, height, width, -1});

      auto width_neg_part = freqs_neg_chunks[2].slice(
          0, -(width - width / 2), torch::indexing::None);
      auto width_pos_part = freqs_pos_chunks[2].slice(0, 0, width / 2);
      freqs_width = torch::cat({width_neg_part, width_pos_part}, 0)
                        .view({1, 1, width, -1})
                        .expand({frame, height, width, -1});
    } else {
      freqs_height = freqs_pos_chunks[1]
                         .slice(0, 0, height)
                         .view({1, height, 1, -1})
                         .expand({frame, height, width, -1});

      freqs_width = freqs_pos_chunks[2]
                        .slice(0, 0, width)
                        .view({1, 1, width, -1})
                        .expand({frame, height, width, -1});
    }
    auto freqs = torch::cat({freqs_frame, freqs_height, freqs_width}, -1)
                     .reshape({seq_lens, -1});
    return freqs.contiguous();
  }

  int64_t theta_;
  std::vector<int64_t> axes_dim_;
  bool scale_rope_;
  torch::Tensor pos_freqs_;
  torch::Tensor neg_freqs_;
  std::unordered_map<std::string, torch::Tensor> rope_cache_;
};

TORCH_MODULE(QwenEmbedRope);

class QwenEmbedRopeWithCacheImpl : public QwenEmbedRopeImpl {
 public:
  QwenEmbedRopeWithCacheImpl(const ModelContext& context,
                             int64_t theta,
                             std::vector<int64_t> axes_dim,
                             bool scale_rope = false)
      : QwenEmbedRopeImpl(context, theta, axes_dim, scale_rope) {}

 private:
  torch::Tensor _compute_video_freqs_cached(int64_t frame,
                                            int64_t height,
                                            int64_t width,
                                            int64_t idx,
                                            torch::Device device) {
    std::string key = std::to_string(idx) + "_" + std::to_string(height) + "_" +
                      std::to_string(width);

    auto it = rope_cache_.find(key);
    if (it != rope_cache_.end()) {
      return it->second;
    } else {
      auto result = _compute_video_freqs(frame, height, width, idx, device);
      rope_cache_[key] = result;
      return result;
    }
  }

  std::unordered_map<std::string, torch::Tensor> rope_cache_;
};
TORCH_MODULE(QwenEmbedRopeWithCache);

class QwenEmbedLayer3DRopeImpl : public torch::nn::Module {
 public:
  QwenEmbedLayer3DRopeImpl(const ModelContext& context,
                           int64_t theta,
                           std::vector<int64_t>& axes_dim,
                           bool scale_rope = false)
      : theta_(theta), axes_dim_(axes_dim), scale_rope_(scale_rope) {
    auto pos_index = torch::arange(4096);
    auto neg_index = torch::arange(4096).flip(0) * -1 - 1;

    std::vector<torch::Tensor> pos_freqs_parts;
    pos_freqs_ = torch::cat({rope_params(pos_index, axes_dim[0], theta),
                             rope_params(pos_index, axes_dim[1], theta),
                             rope_params(pos_index, axes_dim[2], theta)},
                            1);

    neg_freqs_ = torch::cat({rope_params(neg_index, axes_dim[0], theta),
                             rope_params(neg_index, axes_dim[1], theta),
                             rope_params(neg_index, axes_dim[2], theta)},
                            1);
  }

  virtual std::pair<torch::Tensor, torch::Tensor> forward(
      const std::vector<std::vector<int64_t>>& video_fhw,
      int64_t max_txt_seq_len,
      torch::Device device = torch::Device(torch::kCPU)) {
    std::vector<torch::Tensor> vid_freqs_list;
    int64_t max_vid_index = 0;
    int64_t layer_num = video_fhw.size() - 1;

    for (size_t idx = 0; idx < video_fhw.size(); idx++) {
      const std::vector<int64_t>& fhw = video_fhw[idx];

      int64_t frame = fhw[0];
      int64_t height = fhw[1];
      int64_t width = fhw[2];

      torch::Tensor video_freq;

      if (idx != layer_num) {
        video_freq = _compute_video_freqs(frame, height, width, idx, device);
      } else {
        video_freq = _compute_condition_freqs(frame, height, width, device);
      }
      vid_freqs_list.push_back(video_freq);

      if (scale_rope_) {
        max_vid_index = std::max({height / 2, width / 2, max_vid_index});
      } else {
        max_vid_index = std::max({height, width, max_vid_index});
      }
    }

    int64_t max_txt_seq_len_int = std::max(max_vid_index, layer_num);

    torch::Tensor txt_freqs = pos_freqs_.to(device).slice(
        0, max_vid_index, max_vid_index + max_txt_seq_len_int);

    torch::Tensor vid_freqs = torch::cat(vid_freqs_list, 0);

    return {vid_freqs, txt_freqs};
  }

 protected:
  torch::Tensor rope_params(torch::Tensor index, int64_t dim, int64_t theta) {
    CHECK(dim % 2 == 0) << "dim must be even";

    auto exponents =
        torch::arange(
            0, dim, 2, torch::TensorOptions().dtype(torch::kFloat32)) /
        static_cast<float>(dim);
    auto freqs = 1.0 / torch::pow(theta, exponents);

    auto outer_result = torch::outer(index.to(torch::kFloat32), freqs);

    auto complex_freqs =
        torch::polar(torch::ones_like(outer_result), outer_result);

    return complex_freqs;
  }

  torch::Tensor _compute_video_freqs(int64_t frame,
                                     int64_t height,
                                     int64_t width,
                                     int64_t idx,
                                     torch::Device device) {
    int64_t seq_lens = frame * height * width;

    torch::Tensor pos_freqs = pos_freqs_.to(device);
    torch::Tensor neg_freqs = neg_freqs_.to(device);

    std::vector<int64_t> split_sizes;
    for (int64_t dim : axes_dim_) {
      split_sizes.push_back(dim / 2);
    }

    auto freqs_pos = pos_freqs.split_with_sizes(split_sizes, 1);
    auto freqs_neg = neg_freqs.split_with_sizes(split_sizes, 1);

    auto freqs_frame = freqs_pos[0]
                           .slice(0, idx, idx + frame)
                           .view({frame, 1, 1, -1})
                           .expand({frame, height, width, -1});

    torch::Tensor freqs_height;
    if (scale_rope_) {
      auto height_neg_part =
          freqs_neg[1].slice(0, -(height / 2), freqs_neg[1].size(0));
      auto height_pos_part = freqs_pos[1].slice(0, 0, height / 2);
      freqs_height = torch::cat({height_neg_part, height_pos_part}, 0)
                         .view({1, height, 1, -1})
                         .expand({frame, height, width, -1});
    } else {
      freqs_height = freqs_pos[1]
                         .slice(0, 0, height)
                         .view({1, height, 1, -1})
                         .expand({frame, height, width, -1});
    }

    torch::Tensor freqs_width;
    if (scale_rope_) {
      auto neg_part = freqs_neg[2].slice(0, -(width / 2), freqs_neg[2].size(0));
      auto pos_part = freqs_pos[2].slice(0, 0, width / 2);
      freqs_width = torch::cat({neg_part, pos_part}, 0)
                        .view({1, 1, width, -1})
                        .expand({frame, height, width, -1});
    } else {
      freqs_width = freqs_pos[2]
                        .slice(0, 0, width)
                        .view({1, 1, width, -1})
                        .expand({frame, height, width, -1});
    }
    auto freqs =
        torch::cat({freqs_frame, freqs_height, freqs_width}, /*dim=*/-1)
            .reshape({seq_lens, -1})
            .clone()
            .contiguous();

    return freqs;
  }

  torch::Tensor _compute_condition_freqs(int64_t frame,
                                         int64_t height,
                                         int64_t width,
                                         torch::Device device) {
    int64_t seq_lens = frame * height * width;

    torch::Tensor pos_freqs = pos_freqs_.to(device);
    torch::Tensor neg_freqs = neg_freqs_.to(device);

    std::vector<int64_t> split_sizes;
    for (int64_t dim : axes_dim_) {
      split_sizes.push_back(dim / 2);
    }

    auto freqs_pos = pos_freqs.split_with_sizes(split_sizes, 1);
    auto freqs_neg = neg_freqs.split_with_sizes(split_sizes, 1);

    auto freqs_frame = freqs_neg[0]
                           .slice(0, -1, freqs_neg[0].size(0))
                           .view({frame, 1, 1, -1})
                           .expand({frame, height, width, -1});

    torch::Tensor freqs_height;
    if (scale_rope_) {
      auto neg_part =
          freqs_neg[1].slice(0, -(height / 2), freqs_neg[1].size(0));
      auto pos_part = freqs_pos[1].slice(0, 0, height / 2);
      freqs_height = torch::cat({neg_part, pos_part}, 0)
                         .view({1, height, 1, -1})
                         .expand({frame, height, width, -1});
    } else {
      freqs_height = freqs_pos[1]
                         .slice(0, 0, height)
                         .view({1, height, 1, -1})
                         .expand({frame, height, width, -1});
    }
    torch::Tensor freqs_width;
    if (scale_rope_) {
      auto neg_part = freqs_neg[2].slice(0, -(width / 2), freqs_neg[2].size(0));
      auto pos_part = freqs_pos[2].slice(0, 0, width / 2);
      freqs_width = torch::cat({neg_part, pos_part}, 0)
                        .view({1, 1, width, -1})
                        .expand({frame, height, width, -1});
    } else {
      freqs_width = freqs_pos[2]
                        .slice(0, 0, width)
                        .view({1, 1, width, -1})
                        .expand({frame, height, width, -1});
    }
    auto freqs = torch::cat({freqs_frame, freqs_height, freqs_width}, -1)
                     .reshape({seq_lens, -1})
                     .clone()
                     .contiguous();

    return freqs;
  }

  int64_t theta_;
  std::vector<int64_t>& axes_dim_;
  bool scale_rope_;
  torch::Tensor pos_freqs_;
  torch::Tensor neg_freqs_;
};

TORCH_MODULE(QwenEmbedLayer3DRope);

class QwenEmbedLayer3DRopeWithCacheImpl : public QwenEmbedLayer3DRopeImpl {
 public:
  QwenEmbedLayer3DRopeWithCacheImpl(const ModelContext& context,
                                    int64_t theta,
                                    std::vector<int64_t>& axes_dim,
                                    bool scale_rope = false)
      : QwenEmbedLayer3DRopeImpl(context, theta, axes_dim, scale_rope) {}

  std::pair<torch::Tensor, torch::Tensor> forward(
      const std::vector<std::vector<int64_t>>& video_fhw,
      int64_t max_txt_seq_len,
      torch::Device device = torch::Device(torch::kCPU)) override {
    std::vector<torch::Tensor> vid_freqs_list;
    int64_t max_vid_index = 0;
    int64_t layer_num = video_fhw.size() - 1;

    for (size_t idx = 0; idx < video_fhw.size(); idx++) {
      const std::vector<int64_t>& fhw = video_fhw[idx];

      int64_t frame = fhw[0];
      int64_t height = fhw[1];
      int64_t width = fhw[2];

      torch::Tensor video_freq;

      if (idx != layer_num) {
        video_freq =
            _compute_video_freqs_with_cache(frame, height, width, idx, device);
      } else {
        video_freq =
            _compute_condition_freqs_with_cache(frame, height, width, device);
      }
      vid_freqs_list.push_back(video_freq);

      if (scale_rope_) {
        max_vid_index = std::max({height / 2, width / 2, max_vid_index});
      } else {
        max_vid_index = std::max({height, width, max_vid_index});
      }
    }

    int64_t max_txt_seq_len_int = std::max(max_vid_index, layer_num);

    torch::Tensor txt_freqs = pos_freqs_.to(device).slice(
        0, max_vid_index, max_vid_index + max_txt_seq_len_int);

    torch::Tensor vid_freqs = torch::cat(vid_freqs_list, 0);

    return {vid_freqs, txt_freqs};
  }

 private:
  torch::Tensor _compute_video_freqs_with_cache(int64_t frame,
                                                int64_t height,
                                                int64_t width,
                                                int64_t idx,
                                                torch::Device device) {
    std::string key = std::to_string(frame) + "_" + std::to_string(idx) + "_" +
                      std::to_string(height) + "_" + std::to_string(width);

    // TODO: currently the freqs tensors are cached on device
    // need to check whether to swap them to cpu to save device memory
    auto it = video_freqs_cache_.find(key);
    if (it != video_freqs_cache_.end()) {
      return it->second.clone().contiguous();
    } else {
      auto result = _compute_video_freqs(frame, height, width, idx, device);
      video_freqs_cache_[key] = result.clone();
      return result;
    }
  }

  torch::Tensor _compute_condition_freqs_with_cache(int64_t frame,
                                                    int64_t height,
                                                    int64_t width,
                                                    torch::Device device) {
    std::string key = std::to_string(frame) + "_" + std::to_string(height) +
                      "_" + std::to_string(width);

    // TODO: currently the freqs tensors are cached on device
    // need to check whether to swap them to cpu to save device memory
    auto it = condition_cache_.find(key);
    if (it != condition_cache_.end()) {
      return it->second.clone().contiguous();
    } else {
      auto result = _compute_condition_freqs(frame, height, width, device);
      condition_cache_[key] = result.clone();
      return result;
    }
  }

  std::unordered_map<std::string, torch::Tensor> video_freqs_cache_;
  std::unordered_map<std::string, torch::Tensor> condition_cache_;
};

TORCH_MODULE(QwenEmbedLayer3DRopeWithCache);

// A internel class that only register necessary modules for attention
// implementation The attention forward shouldn't be implemented here, but in
// processor classes
// TODO: This class should be extracted from dit class and integrated into a
// common class.
class AttentionImpl : public torch::nn::Module {
 public:
  AttentionImpl(const ModelContext context,
                int64_t query_dim,
                std::optional<int64_t> cross_attention_dim = std::nullopt,
                int64_t heads = 8,
                std::optional<int64_t> kv_heads = std::nullopt,
                int64_t dim_head = 64,
                double dropout = 0.0,
                bool bias = false,
                const std::string& qk_norm = "",
                const std::string& cross_attention_norm = "",
                std::optional<int64_t> added_kv_proj_dim = std::nullopt,
                bool added_proj_bias = true,
                bool out_bias = true,
                bool scale_qk = true,
                bool only_cross_attention = false,
                double eps = 1e-5,
                double rescale_output_factor = 1.0,
                bool residual_connection = false,
                std::optional<int64_t> out_dim = std::nullopt,
                std::optional<int64_t> out_context_dim = std::nullopt,
                std::optional<bool> context_pre_only = std::nullopt,
                int img_pad = 0,
                int text_pad = 0,
                bool pre_only = false,
                bool elementwise_affine = true,
                bool is_causal = false)
      : heads_(heads),
        bias_(bias),
        out_bias_(out_bias),
        added_proj_bias_(added_proj_bias),
        pg_(context.get_parallel_args()),
        img_pad_(img_pad),
        text_pad_(text_pad) {
    world_size_ = pg_.world_size();
    rank_ = pg_.rank();
    use_sp_ = world_size_ > 1;
    if (qk_norm == "layer_norm") {
      layer_norm_q_ = register_module(
          "norm_q",
          torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim_head})
                                   .eps(eps)
                                   .elementwise_affine(elementwise_affine)));
      layer_norm_k_ = register_module(
          "norm_k",
          torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim_head})
                                   .eps(eps)
                                   .elementwise_affine(elementwise_affine)));
    } else if (qk_norm == "layer_norm_across_heads") {
      // Lumina applies qk norm across all heads
      CHECK(kv_heads.has_value())
          << "qk_norm is set to: " + qk_norm + ", but get no kv_heads ";
      layer_norm_q_ = register_module(
          "norm_q",
          torch::nn::LayerNorm(
              torch::nn::LayerNormOptions({dim_head * heads}).eps(eps)));
      layer_norm_k_ = register_module(
          "norm_k",
          torch::nn::LayerNorm(
              torch::nn::LayerNormOptions({dim_head * kv_heads.value()})
                  .eps(eps)));
    } else if (qk_norm == "rms_norm") {
      // Assuming you have an RMSNorm implementation
      norm_q_ = register_module("norm_q", RMSNorm(dim_head, eps, true, false));
      norm_k_ = register_module("norm_k", RMSNorm(dim_head, eps, true, false));
    } else if (qk_norm == "rms_norm_across_heads") {
      // LTX applies qk norm across all heads
      CHECK(kv_heads.has_value())
          << "qk_norm is set to: " + qk_norm + ", but get no kv_heads ";

      norm_q_ = register_module("norm_q", RMSNorm(dim_head, eps, true, false));
      norm_k_ = register_module(
          "norm_k", RMSNorm(dim_head * kv_heads.value(), eps, true, false));
    } else {
      CHECK(qk_norm.empty()) << "unknown qk_norm: " + qk_norm +
                                    ". Should be "
                                    "'','layer_norm','rms_norm','layer_norm_"
                                    "across_heads', 'rms_norm_across_heads'";
    }

    if (cross_attention_norm == "layer_norm") {
      norm_cross_ = register_module(
          "norm_cross",
          torch::nn::LayerNorm(
              torch::nn::LayerNormOptions({cross_attention_dim.value()})));
    } else {
      CHECK(cross_attention_norm.empty())
          << "unknown cross_attention_norm: " + cross_attention_norm +
                 ". Should be '', 'layer_norm'";
    }

    int64_t q_dim = out_dim.has_value() ? out_dim.value() : dim_head * heads;
    int64_t kv_dim =
        !kv_heads.has_value() ? q_dim : dim_head * kv_heads.value();
    cross_attention_dim = cross_attention_dim.has_value()
                              ? cross_attention_dim.value()
                              : query_dim;
    out_context_dim =
        out_context_dim.has_value() ? out_context_dim.value() : query_dim;
    to_q_ = register_module("to_q", DiTLinear(query_dim, q_dim, bias));

    // Key-Value projections (if not only cross attention)
    if (!only_cross_attention) {
      to_k_ = register_module(
          "to_k", DiTLinear(cross_attention_dim.value(), kv_dim, bias));
      to_v_ = register_module(
          "to_v", DiTLinear(cross_attention_dim.value(), kv_dim, bias));
    }

    if (added_kv_proj_dim.has_value()) {
      add_k_proj_ = register_module(
          "add_k_proj",
          DiTLinear(added_kv_proj_dim.value(), kv_dim, added_proj_bias));
      add_v_proj_ = register_module(
          "add_v_proj",
          DiTLinear(added_kv_proj_dim.value(), kv_dim, added_proj_bias));

      if (context_pre_only.has_value()) {
        add_q_proj_ = register_module(
            "add_q_proj",
            DiTLinear(added_kv_proj_dim.value(), q_dim, added_proj_bias));
      }
    }

    // Output projections
    if (!pre_only) {
      to_out_ = register_module("to_out", torch::nn::Sequential());
      to_out_->push_back(DiTLinear(q_dim, out_dim.value(), out_bias));
      to_out_->push_back(
          torch::nn::Dropout(torch::nn::DropoutOptions(dropout)));
    }

    // Additional output for context
    if (context_pre_only.has_value() && context_pre_only) {
      to_add_out_ = register_module(
          "to_add_out", DiTLinear(q_dim, out_context_dim.value(), out_bias));
    }

    // Added QK normalization for added KV projections
    if (!qk_norm.empty() && added_kv_proj_dim.has_value()) {
      if (qk_norm == "rms_norm") {
        norm_added_q_ = register_module("norm_added_q",
                                        RMSNorm(dim_head, eps, true, false));
        norm_added_k_ = register_module("norm_added_k",
                                        RMSNorm(dim_head, eps, true, false));
      } else {
        CHECK(qk_norm.empty()) << "unknown qk_norm: " + qk_norm +
                                      ". Should be one of '','rms_norm'";
        // For layer_norm, we would register similar layers here
      }
    }
  }

  void load_state_dict(const StateDict& state_dict) {
    // to_out
    to_out_[0]->as<DiTLinear>()->load_state_dict(
        state_dict.get_dict_with_prefix("to_out.0."));
    // to_add_out
    to_add_out_->load_state_dict(
        state_dict.get_dict_with_prefix("to_add_out."));
    // norm_q
    norm_q_->load_state_dict(state_dict.get_dict_with_prefix("norm_q."));
    // norm_k
    norm_k_->load_state_dict(state_dict.get_dict_with_prefix("norm_k."));
    // norm_added_q
    norm_added_q_->load_state_dict(
        state_dict.get_dict_with_prefix("norm_added_q."));
    // norm_added_k
    norm_added_k_->load_state_dict(
        state_dict.get_dict_with_prefix("norm_added_k."));

    to_q_->load_state_dict(state_dict.get_dict_with_prefix("to_q."));
    to_k_->load_state_dict(state_dict.get_dict_with_prefix("to_k."));
    to_v_->load_state_dict(state_dict.get_dict_with_prefix("to_v."));

    add_q_proj_->load_state_dict(
        state_dict.get_dict_with_prefix("add_q_proj."));
    add_k_proj_->load_state_dict(
        state_dict.get_dict_with_prefix("add_k_proj."));
    add_v_proj_->load_state_dict(
        state_dict.get_dict_with_prefix("add_v_proj."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    // to_out
    to_out_[0]->as<DiTLinear>()->verify_loaded_weights(prefix + "to_out.0.");
    // to_add_out
    to_add_out_->verify_loaded_weights(prefix + "to_add_out.");
    // norm_q
    norm_q_->verify_loaded_weights(prefix + "norm_q.");
    // norm_k
    norm_k_->verify_loaded_weights(prefix + "norm_k.");
    // norm_added_q
    norm_added_q_->verify_loaded_weights(prefix + "norm_added_q.");
    // norm_added_k
    norm_added_k_->verify_loaded_weights(prefix + "norm_added_k.");

    to_q_->verify_loaded_weights(prefix + "to_q.");
    to_k_->verify_loaded_weights(prefix + "to_k.");
    to_v_->verify_loaded_weights(prefix + "to_v.");

    add_q_proj_->verify_loaded_weights(prefix + "add_q_proj.");
    add_k_proj_->verify_loaded_weights(prefix + "add_k_proj.");
    add_v_proj_->verify_loaded_weights(prefix + "add_v_proj.");
  }

 public:
  int64_t heads_;
  bool bias_;
  bool out_bias_;
  bool added_proj_bias_;

  torch::nn::LayerNorm layer_norm_q_{nullptr}, layer_norm_k_{nullptr},
      norm_cross_{nullptr};
  DiTLinear to_q_{nullptr}, to_k_{nullptr}, to_v_{nullptr};
  DiTLinear add_k_proj_{nullptr}, add_v_proj_{nullptr}, add_q_proj_{nullptr};
  torch::nn::Sequential to_out_{nullptr};
  DiTLinear to_add_out_{nullptr};

  // Assuming you have RMSNorm implemented
  RMSNorm norm_q_{nullptr}, norm_k_{nullptr}, norm_added_q_{nullptr},
      norm_added_k_{nullptr};

  // For sequence parallel
  int world_size_{1};
  int rank_{0};
  bool use_sp_{false};
  int text_pad_{0};
  int img_pad_{0};
  ParallelArgs pg_;
};
TORCH_MODULE(Attention);

// Implementation of attention forward
class QwenDoubleStreamAttnProcessor2_0Impl : public torch::nn::Module {
 public:
  QwenDoubleStreamAttnProcessor2_0Impl(Attention&& attn_module) {
    attn_ = register_module("attn", std::move(attn_module));
  }

  std::tuple<torch::Tensor,
             torch::Tensor,
             torch::Tensor,
             torch::Tensor,
             torch::Tensor,
             torch::Tensor>
  sp_qkv_matmul(const torch::Tensor& hidden_states,
                const torch::Tensor& encoder_hidden_states) {
    // std::cout << "[DEBUG sp_qkv_matmul] Input hidden_states shape: " <<
    // hidden_states.sizes() << std::endl; std::cout << "[DEBUG sp_qkv_matmul]
    // Input encoder_hidden_states shape: " << encoder_hidden_states.sizes() <<
    // std::endl;

    int64_t seq_txt = encoder_hidden_states.size(1);
    int64_t seq_img = hidden_states.size(1);
    auto pg_ = attn_->pg_.process_group_;
    auto rank_ = attn_->rank_;
    auto world_size_ = attn_->world_size_;

    // int64_t bs_txt = encoder_hidden_states.size(0);
    int64_t bs_img = hidden_states.size(0);
    int64_t inner_dim = attn_->to_k_->weight.size(0);
    int64_t heads = attn_->heads_;
    int64_t head_dim = inner_dim / heads;

    // Compute QKV for image stream (sample projections)
    auto img_query = attn_->to_q_->forward(hidden_states);
    // std::cout << "[DEBUG sp_qkv_matmul] img_query after to_q forward shape: "
    // << img_query.sizes() << std::endl;
    auto save_tensor = [this](const torch::Tensor& tensor,
                              const std::string& name) {
      if (tensor.defined()) {
        torch::Tensor cpu_tensor = tensor.cpu();
        std::string filename =
            name + "_rank_" + std::to_string(attn_->rank_) + ".pt";
        torch::save(cpu_tensor, filename);
      }
    };

    save_tensor(encoder_hidden_states, "sp/encoder_hidden_states");

    auto reshape_dims = std::vector<int64_t>{heads, -1};
    // img_query = img_query.unflatten(-1, reshape_dims);
    img_query = img_query.view({bs_img, -1, heads, head_dim});

    // std::cout << "[DEBUG sp_qkv_matmul] img_query after view shape: " <<
    // img_query.sizes() << std::endl;

    auto handle_iq = parallel_state::all_to_all_4D(
        img_query, rank_, world_size_, 2, 1, false, pg_);
    // std::cout << "[DEBUG sp_qkv_matmul] img_query before all_to_all_4D shape:
    // " << img_query.sizes() << std::endl;

    auto img_key = attn_->to_k_->forward(hidden_states);
    // std::cout << "[DEBUG sp_qkv_matmul] img_key after to_k forward shape: "
    // << img_key.sizes() << std::endl;

    // img_key = img_key.unflatten(-1, reshape_dims);
    img_key = img_key.view({bs_img, -1, heads, head_dim});
    // std::cout << "[DEBUG sp_qkv_matmul] img_key after view shape: " <<
    // img_key.sizes() << std::endl;

    img_query = parallel_state::all_to_all_4D_post2(handle_iq);
    // std::cout << "[DEBUG sp_qkv_matmul] img_query after all_to_all_4D_post2
    // shape: " << img_query.sizes() << std::endl;
    save_tensor(img_query, "sp/img_query_a2a");

    auto handle_ik = parallel_state::all_to_all_4D(
        img_key, rank_, world_size_, 2, 1, false, pg_);
    // std::cout << "[DEBUG sp_qkv_matmul] img_key before all_to_all_4D shape: "
    // << img_key.sizes() << std::endl;

    auto img_value = attn_->to_v_->forward(hidden_states);
    // std::cout << "[DEBUG sp_qkv_matmul] img_value after to_v forward shape: "
    // << img_value.sizes() << std::endl;

    // img_value = img_value.unflatten(-1, reshape_dims);
    img_value = img_value.view({bs_img, -1, heads, head_dim});
    // std::cout << "[DEBUG sp_qkv_matmul] img_value after view shape: " <<
    // img_value.sizes() << std::endl;

    img_key = parallel_state::all_to_all_4D_post2(handle_ik);
    // std::cout << "[DEBUG sp_qkv_matmul] img_key after all_to_all_4D_post2
    // shape: " << img_key.sizes() << std::endl;

    auto handle_iv = parallel_state::all_to_all_4D(
        img_value, rank_, world_size_, 2, 1, false, pg_);
    // std::cout << "[DEBUG sp_qkv_matmul] img_value before all_to_all_4D shape:
    // " << img_value.sizes() << std::endl;

    // Compute QKV for text stream (context projections)
    auto txt_query = attn_->add_q_proj_->forward(encoder_hidden_states);
    // std::cout << "[DEBUG sp_qkv_matmul] txt_query after add_q_proj forward
    // shape: " << txt_query.sizes() << std::endl;
    
    // txt_query = txt_query.unflatten(-1, reshape_dims);
    txt_query = txt_query.view({bs_img, -1, heads, head_dim});
    save_tensor(txt_query, "sp/txt_querymm");
    // std::cout << "[DEBUG sp_qkv_matmul] txt_query after view shape: " <<
    // txt_query.sizes() << std::endl;

    img_value = parallel_state::all_to_all_4D_post2(handle_iv);
    // std::cout << "[DEBUG sp_qkv_matmul] img_value after all_to_all_4D_post2
    // shape: " << img_value.sizes() << std::endl;

    auto handle_tq = parallel_state::all_to_all_4D(
        txt_query, rank_, world_size_, 2, 1, false, pg_);
    // std::cout << "[DEBUG sp_qkv_matmul] txt_query before all_to_all_4D shape:
    // " << txt_query.sizes() << std::endl;

    auto txt_key = attn_->add_k_proj_->forward(encoder_hidden_states);
    // std::cout << "[DEBUG sp_qkv_matmul] txt_key after add_k_proj forward
    // shape: " << txt_key.sizes() << std::endl;
    save_tensor(txt_key, "sp/txt_keymm");

    // txt_key = txt_key.unflatten(-1, reshape_dims);
    txt_key = txt_key.view({bs_img, -1, heads, head_dim});
    // std::cout << "[DEBUG sp_qkv_matmul] txt_key after view shape: " <<
    // txt_key.sizes() << std::endl;

    txt_query = parallel_state::all_to_all_4D_post2(handle_tq);
    // std::cout << "[DEBUG sp_qkv_matmul] txt_query after all_to_all_4D_post2
    // shape: " << txt_query.sizes() << std::endl;

    auto handle_tk = parallel_state::all_to_all_4D(
        txt_key, rank_, world_size_, 2, 1, false, pg_);
    // std::cout << "[DEBUG sp_qkv_matmul] txt_key before all_to_all_4D shape: "
    // << txt_key.sizes() << std::endl;

    auto txt_value = attn_->add_v_proj_->forward(encoder_hidden_states);
    // std::cout << "[DEBUG sp_qkv_matmul] txt_value after add_v_proj forward
    // shape: " << txt_value.sizes() << std::endl;
    save_tensor(txt_value, "sp/txt_valuemm");

    // txt_value = txt_value.unflatten(-1, reshape_dims);
    txt_value = txt_value.view({bs_img, -1, heads, head_dim});
    // std::cout << "[DEBUG sp_qkv_matmul] txt_value after view shape: " <<
    // txt_value.sizes() << std::endl;

    txt_key = parallel_state::all_to_all_4D_post2(handle_tk);
    // std::cout << "[DEBUG sp_qkv_matmul] txt_key after all_to_all_4D_post2
    // shape: " << txt_key.sizes() << std::endl;

    auto handle_tv = parallel_state::all_to_all_4D(
        txt_value, rank_, world_size_, 2, 1, false, pg_);
    // std::cout << "[DEBUG sp_qkv_matmul] txt_value before all_to_all_4D shape:
    // " << txt_value.sizes() << std::endl;

    // sp unpad
    img_query = unpad_sequence(img_query, 1, attn_->img_pad_);
    // std::cout << "[DEBUG sp_qkv_matmul] img_query after unpad_sequence shape:
    // " << img_query.sizes() << std::endl;

    img_key = unpad_sequence(img_key, 1, attn_->img_pad_);
    // std::cout << "[DEBUG sp_qkv_matmul] img_key after unpad_sequence shape: "
    // << img_key.sizes() << std::endl;

    img_value = unpad_sequence(img_value, 1, attn_->img_pad_);
    // std::cout << "[DEBUG sp_qkv_matmul] img_value after unpad_sequence shape:
    // " << img_value.sizes() << std::endl;

    txt_query = unpad_sequence(txt_query, 1, attn_->text_pad_);
    // std::cout << "[DEBUG sp_qkv_matmul] txt_query after unpad_sequence shape:
    // " << txt_query.sizes() << std::endl;

    txt_key = unpad_sequence(txt_key, 1, attn_->text_pad_);
    // std::cout << "[DEBUG sp_qkv_matmul] txt_key after unpad_sequence shape: "
    // << txt_key.sizes() << std::endl;

    txt_value = parallel_state::all_to_all_4D_post2(handle_tv);
    // std::cout << "[DEBUG sp_qkv_matmul] txt_value after all_to_all_4D_post2
    // shape: " << txt_value.sizes() << std::endl;

    txt_value = unpad_sequence(txt_value, 1, attn_->text_pad_);
    // std::cout << "[DEBUG sp_qkv_matmul] txt_value after unpad_sequence shape:
    // " << txt_value.sizes() << std::endl;

    // std::cout << "[DEBUG sp_qkv_matmul] Final img_query shape: " <<
    // img_query.sizes() << std::endl; std::cout << "[DEBUG sp_qkv_matmul] Final
    // img_key shape: " << img_key.sizes() << std::endl; std::cout << "[DEBUG
    // sp_qkv_matmul] Final img_value shape: " << img_value.sizes() <<
    // std::endl; std::cout << "[DEBUG sp_qkv_matmul] Final txt_query shape: "
    // << txt_query.sizes() << std::endl; std::cout << "[DEBUG sp_qkv_matmul]
    // Final txt_key shape: " << txt_key.sizes() << std::endl; std::cout <<
    // "[DEBUG sp_qkv_matmul] Final txt_value shape: " << txt_value.sizes() <<
    // std::endl;

    return std::make_tuple(
        img_query, img_key, img_value, txt_query, txt_key, txt_value);
  }

  std::tuple<torch::Tensor,
             torch::Tensor,
             torch::Tensor,
             torch::Tensor,
             torch::Tensor,
             torch::Tensor>
  qkv_matmul(const torch::Tensor& hidden_states,
             const torch::Tensor& encoder_hidden_states) {
    // std::cout << "[DEBUG qkv_matmul] Input hidden_states shape: " <<
    // hidden_states.sizes() << std::endl; std::cout << "[DEBUG qkv_matmul]
    // Input encoder_hidden_states shape: " << encoder_hidden_states.sizes() <<
    // std::endl;

    int64_t seq_txt = encoder_hidden_states.size(1);
    int64_t seq_img = hidden_states.size(1);
    // Compute QKV for image stream (sample projections)
    auto img_query = attn_->to_q_->forward(hidden_states);
    // std::cout << "[DEBUG qkv_matmul] img_query after to_q forward shape: " <<
    // img_query.sizes() << std::endl;

    torch::save(encoder_hidden_states.cpu(), "tp1/encoder_hidden_states.pt");
    // Reshape for multi-head attention
    int64_t heads = attn_->heads_;
    auto reshape_dims = std::vector<int64_t>{heads, -1};
    img_query = img_query.unflatten(-1, reshape_dims);
    // std::cout << "[DEBUG qkv_matmul] img_query after unflatten shape: " <<
    // img_query.sizes() << std::endl;
    // torch::save(img_query.cpu(), "tp1/img_querymm.pt");
    auto img_key = attn_->to_k_->forward(hidden_states);
    // std::cout << "[DEBUG qkv_matmul] img_key after to_k forward shape: " <<
    // img_key.sizes() << std::endl;

    img_key = img_key.unflatten(-1, reshape_dims);
    // std::cout << "[DEBUG qkv_matmul] img_key after unflatten shape: " <<
    // img_key.sizes() << std::endl;

    auto img_value = attn_->to_v_->forward(hidden_states);
    // std::cout << "[DEBUG qkv_matmul] img_value after to_v forward shape: " <<
    // img_value.sizes() << std::endl;

    img_value = img_value.unflatten(-1, reshape_dims);
    // std::cout << "[DEBUG qkv_matmul] img_value after unflatten shape: " <<
    // img_value.sizes() << std::endl;

    // Compute QKV for text stream (context projections)
    auto txt_query = attn_->add_q_proj_->forward(encoder_hidden_states);
    // std::cout << "[DEBUG qkv_matmul] txt_query after add_q_proj forward
    // shape: " << txt_query.sizes() << std::endl;

    txt_query = txt_query.unflatten(-1, reshape_dims);
    // std::cout << "[DEBUG qkv_matmul] txt_query after unflatten shape: " <<
    // txt_query.sizes() << std::endl;
    torch::save(txt_query.cpu(), "tp1/txt_querymm.pt");

    auto txt_key = attn_->add_k_proj_->forward(encoder_hidden_states);
    // std::cout << "[DEBUG qkv_matmul] txt_key after add_k_proj forward shape:
    // " << txt_key.sizes() << std::endl;
    torch::save(txt_key.cpu(), "tp1/txt_keymm.pt");

    txt_key = txt_key.unflatten(-1, reshape_dims);
    // std::cout << "[DEBUG qkv_matmul] txt_key after unflatten shape: " <<
    // txt_key.sizes() << std::endl;

    auto txt_value = attn_->add_v_proj_->forward(encoder_hidden_states);
    // std::cout << "[DEBUG qkv_matmul] txt_value after add_v_proj forward
    // shape: " << txt_value.sizes() << std::endl;
    torch::save(txt_value.cpu(), "tp1/txt_valuemm.pt");

    txt_value = txt_value.unflatten(-1, reshape_dims);
    // std::cout << "[DEBUG qkv_matmul] txt_value after unflatten shape: " <<
    // txt_value.sizes() << std::endl;

    // std::cout << "[DEBUG qkv_matmul] Final img_query shape: " <<
    // img_query.sizes() << std::endl; std::cout << "[DEBUG qkv_matmul] Final
    // img_key shape: " << img_key.sizes() << std::endl; std::cout << "[DEBUG
    // qkv_matmul] Final img_value shape: " << img_value.sizes() << std::endl;
    // std::cout << "[DEBUG qkv_matmul] Final txt_query shape: " <<
    // txt_query.sizes() << std::endl; std::cout << "[DEBUG qkv_matmul] Final
    // txt_key shape: " << txt_key.sizes() << std::endl; std::cout << "[DEBUG
    // qkv_matmul] Final txt_value shape: " << txt_value.sizes() << std::endl;

    return std::make_tuple(
        img_query, img_key, img_value, txt_query, txt_key, txt_value);
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& hidden_states,          // Image stream
      const torch::Tensor& encoder_hidden_states,  // Text stream
      const torch::Tensor& encoder_hidden_states_mask = torch::Tensor(),
      const torch::Tensor& attention_mask = torch::Tensor(),
      const std::tuple<at::Tensor, at::Tensor>& image_rotary_emb = {}) {
    // std::cout << "[DEBUG forward] Input hidden_states shape: " <<
    // hidden_states.sizes() << std::endl; std::cout << "[DEBUG forward] Input
    // encoder_hidden_states shape: " << encoder_hidden_states.sizes() <<
    // std::endl;

    torch::Tensor img_query, img_key, img_value;
    torch::Tensor txt_query, txt_key, txt_value;
    // 将注意力张量转移到CPU并保存到本地文件，文件后缀为当前rank
    auto save_tensor = [this](const torch::Tensor& tensor,
                              const std::string& name) {
      if (tensor.defined()) {
        torch::Tensor cpu_tensor = tensor.cpu();
        std::string filename =
            name + "_rank_" + std::to_string(attn_->rank_) + ".pt";
        torch::save(cpu_tensor, filename);
      }
    };
    // Compute QKV projections
    if (attn_->use_sp_) {
      std::tie(img_query, img_key, img_value, txt_query, txt_key, txt_value) =
          sp_qkv_matmul(hidden_states, encoder_hidden_states);

      // save_tensor(img_query, "sp/img_query");
      // save_tensor(img_key, "sp/img_key");
      // save_tensor(img_value, "sp/img_value");
      save_tensor(txt_query, "sp/txt_query");
      save_tensor(txt_key, "sp/txt_key");
      save_tensor(txt_value, "sp/txt_value");
    } else {
      std::tie(img_query, img_key, img_value, txt_query, txt_key, txt_value) =
          qkv_matmul(hidden_states, encoder_hidden_states);
      // 将张量转移到CPU并保存到本地
      // torch::save(img_query.cpu(), "tp1/img_query.pt");
      // torch::save(img_key.cpu(), "tp1/img_key.pt");
      // torch::save(img_value.cpu(), "tp1/img_value.pt");
      torch::save(txt_query.cpu(), "tp1/txt_query.pt");
      torch::save(txt_key.cpu(), "tp1/txt_key.pt");
      torch::save(txt_value.cpu(), "tp1/txt_value.pt");
    }

    // std::cout << "[DEBUG forward] img_query shape after QKV matmul: " <<
    // img_query.sizes() << std::endl; std::cout << "[DEBUG forward] img_key
    // shape after QKV matmul: " << img_key.sizes() << std::endl; std::cout <<
    // "[DEBUG forward] img_value shape after QKV matmul: " << img_value.sizes()
    // << std::endl; std::cout << "[DEBUG forward] txt_query shape after QKV
    // matmul: " << txt_query.sizes() << std::endl; std::cout << "[DEBUG
    // forward] txt_key shape after QKV matmul: " << txt_key.sizes() <<
    // std::endl; std::cout << "[DEBUG forward] txt_value shape after QKV
    // matmul: " << txt_value.sizes() << std::endl;

    // Apply QK normalization
    if (attn_->norm_q_) {
      img_query = attn_->norm_q_->forward(img_query);
    }
    if (attn_->norm_k_) {
      img_key = attn_->norm_k_->forward(img_key);
    }
    if (attn_->norm_added_q_) {
      txt_query = attn_->norm_added_q_->forward(txt_query);
    }
    if (attn_->norm_added_k_) {
      txt_key = attn_->norm_added_k_->forward(txt_key);
    }

    if (attn_->use_sp_) {
      save_tensor(txt_query, "sp/txt_query_norm");
      save_tensor(txt_key, "sp/txt_key_norm");  
    } else {
      torch::save(txt_query.cpu(), "tp1/txt_query_norm.pt");
      torch::save(txt_key.cpu(), "tp1/txt_key_norm.pt");
    }


    // std::cout << "[DEBUG forward] img_query shape after normalization: " <<
    // img_query.sizes() << std::endl; std::cout << "[DEBUG forward] img_key
    // shape after normalization: " << img_key.sizes() << std::endl; std::cout
    // << "[DEBUG forward] txt_query shape after normalization: " <<
    // txt_query.sizes() << std::endl; std::cout << "[DEBUG forward] txt_key
    // shape after normalization: " << txt_key.sizes() << std::endl;

    // Apply RoPE if provided
    auto img_freqs = std::get<0>(image_rotary_emb);
    auto txt_freqs = std::get<1>(image_rotary_emb);

    img_query = apply_rotary_emb_qwen(img_query, img_freqs, false);
    img_key = apply_rotary_emb_qwen(img_key, img_freqs, false);
    txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, false);
    txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, false);

    if (attn_->use_sp_) {
      save_tensor(txt_query, "sp/txt_query_rope");
      save_tensor(txt_key, "sp/txt_key_rope");  
    } else {
      torch::save(txt_query.cpu(), "tp1/txt_query_rope.pt");
      torch::save(txt_key.cpu(), "tp1/txt_key_rope.pt");
    }

    // std::cout << "[DEBUG forward] img_query shape after RoPE: " <<
    // img_query.sizes() << std::endl; std::cout << "[DEBUG forward] img_key
    // shape after RoPE: " << img_key.sizes() << std::endl; std::cout << "[DEBUG
    // forward] txt_query shape after RoPE: " << txt_query.sizes() << std::endl;
    // std::cout << "[DEBUG forward] txt_key shape after RoPE: " <<
    // txt_key.sizes() << std::endl;

    // Concatenate for joint attention - Order: [text, image]
    auto joint_query = torch::cat({txt_query, img_query}, 1);
    auto joint_key = torch::cat({txt_key, img_key}, 1);
    auto joint_value = torch::cat({txt_value, img_value}, 1);

    // std::cout << "[DEBUG forward] joint_query shape after concatenation: " <<
    // joint_query.sizes() << std::endl; std::cout << "[DEBUG forward] joint_key
    // shape after concatenation: " << joint_key.sizes() << std::endl; std::cout
    // << "[DEBUG forward] joint_value shape after concatenation: " <<
    // joint_value.sizes() << std::endl;

    int head_num = img_query.size(2);
    // int64_t head_dim_ = img_query.size(-1);

    auto results = at_npu::native::custom_ops::npu_fusion_attention(
        joint_query,
        joint_key,
        joint_value,
        head_num,
        /*input_layout=*/"BSND",
        /*pse=*/torch::nullopt,
        /*padding_mask=*/torch::nullopt,
        /*atten_mask*/ torch::nullopt,
        /*scale=*/pow(joint_query.size(3), -0.5),
        /*keep_prob=*/1.0,
        /*pre_tockens=*/65535,
        /*next_tockens=*/65535);

    auto joint_hidden_states = std::get<0>(results);
    // std::cout << "[DEBUG forward] joint_hidden_states shape after attention:
    // " << joint_hidden_states.sizes() << std::endl;

    // Reshape back
    joint_hidden_states = joint_hidden_states.flatten(2, 3);
    // std::cout << "[DEBUG forward] joint_hidden_states shape after flatten: "
    // << joint_hidden_states.sizes() << std::endl;

    joint_hidden_states = joint_hidden_states.to(joint_query.dtype());

    int64_t seq_txt = txt_query.size(1);
    int64_t seq_img = img_query.size(1);
    auto pg_ = attn_->pg_.process_group_;
    auto rank_ = attn_->rank_;
    auto world_size_ = attn_->world_size_;
    // Split attention outputs back
    auto chunks = torch::split(joint_hidden_states, {seq_txt, seq_img}, 1);
    auto txt_attn_output = chunks[0];
    auto img_attn_output = chunks[1];

    if (attn_->use_sp_) {
      save_tensor(txt_attn_output, "sp/txt_attn_output");
    } else {
      torch::save(txt_attn_output.cpu(), "tp1/txt_attn_output.pt");
    }

    // std::cout << "[DEBUG forward] txt_attn_output shape after split: " <<
    // txt_attn_output.sizes() << std::endl; std::cout << "[DEBUG forward]
    // img_attn_output shape after split: " << img_attn_output.sizes() <<
    // std::endl; all tp all 前需要sp pad
    parallel_state::AllToAll4DHandle handle_io, handle_t_o;
    int64_t batch_size = hidden_states.size(0);
    int64_t inner_dim = attn_->to_k_->weight.size(0);
    int64_t attn_heads = attn_->heads_;
    int64_t head_dim = inner_dim / attn_heads;
    if (attn_->use_sp_) {
      img_attn_output = pad_sequence(img_attn_output, 1, attn_->img_pad_);
      // std::cout << "[DEBUG forward] img_attn_output shape after pad_sequence:
      // " << img_attn_output.sizes() << std::endl;

      handle_io = parallel_state::all_to_all_4D(
          img_attn_output.view({batch_size, -1, head_num, head_dim}),
          rank_,
          world_size_,
          1,
          2,
          true,
          pg_);
      img_attn_output = parallel_state::all_to_all_4D_post(handle_io);
      // std::cout << "[DEBUG forward] img_attn_output shape after
      // all_to_all_4D_post: " << img_attn_output.sizes() << std::endl;

      img_attn_output = img_attn_output.view({batch_size, -1, inner_dim});
      // std::cout << "[DEBUG forward] img_attn_output shape after view: " <<
      // img_attn_output.sizes() << std::endl;

      txt_attn_output = pad_sequence(txt_attn_output, 1, attn_->text_pad_);
      // std::cout << "[DEBUG forward] txt_attn_output shape after pad_sequence:
      // " << txt_attn_output.sizes() << std::endl;

      handle_t_o = parallel_state::all_to_all_4D(
          txt_attn_output.view({batch_size, -1, head_num, head_dim}),
          rank_,
          world_size_,
          1,
          2,
          false,
          pg_);
    }
    // Apply output projections
    img_attn_output = attn_->to_out_->forward(img_attn_output);
    // std::cout << "[DEBUG forward] img_attn_output shape after to_out forward:
    // " << img_attn_output.sizes() << std::endl;

    if (attn_->use_sp_) {
      txt_attn_output = parallel_state::all_to_all_4D_post(handle_t_o);
      // std::cout << "[DEBUG forward] txt_attn_output shape after
      // all_to_all_4D_post: " << txt_attn_output.sizes() << std::endl;

      txt_attn_output = txt_attn_output.view({batch_size, -1, inner_dim});
      // std::cout << "[DEBUG forward] txt_attn_output shape after view: " <<
      // txt_attn_output.sizes() << std::endl;
    }
    if (attn_->use_sp_) {
      save_tensor(txt_attn_output, "sp/txt_attn_output_a2a");
    }
    txt_attn_output = attn_->to_add_out_->forward(txt_attn_output);
    // std::cout << "[DEBUG forward] txt_attn_output shape after to_add_out
    // forward: " << txt_attn_output.sizes() << std::endl;

    // std::cout << "[DEBUG forward] Final img_attn_output shape: " <<
    // img_attn_output.sizes() << std::endl; std::cout << "[DEBUG forward] Final
    // txt_attn_output shape: " << txt_attn_output.sizes() << std::endl;
    if (attn_->use_sp_) {
      save_tensor(txt_attn_output, "sp/txt_attn_output_o");
    } else {
      torch::save(txt_attn_output.cpu(), "tp1/txt_attn_output_o.pt");
    }
    return std::make_tuple(img_attn_output, txt_attn_output);
  }

  void load_state_dict(const StateDict& state_dict) {
    attn_->load_state_dict(state_dict);
  }

  void verify_loaded_weights(const std::string& prefix) {
    attn_->verify_loaded_weights(prefix);
  }

  void set_text_pad_(const int pad) { attn_->text_pad_ = pad; }

  void set_img_pad_(const int pad) { attn_->img_pad_ = pad; }

  int get_img_pad_() { return attn_->img_pad_; }

  int get_text_pad_() { return attn_->text_pad_; }

  //  private:
  Attention attn_{nullptr};
};
TORCH_MODULE(QwenDoubleStreamAttnProcessor2_0);

class FeedForwardImpl : public torch::nn::Module {
 public:
  explicit FeedForwardImpl(const ModelContext& context,
                           int64_t dim,
                           int64_t dim_out,
                           int64_t mult = 4,
                           double dropout = 0.0)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    auto inner_dim = dim * 4;

    // linear1
    linear1_ = register_module("linear1", DiTLinear(dim, inner_dim, true));

    // activation
    activation_ = register_module(
        "activation",
        torch::nn::Functional(std::function<at::Tensor(const at::Tensor&)>(
            [](const at::Tensor& x) { return torch::gelu(x, "tanh"); })));

    // linear2
    linear2_ = register_module("linear2", DiTLinear(inner_dim, dim_out, true));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    torch::Tensor out = linear1_->forward(hidden_states);
    out = activation_(out);
    out = linear2_->forward(out);
    return out;
  }

  void load_state_dict(const StateDict& state_dict) {
    // linear1
    linear1_->load_state_dict(state_dict.get_dict_with_prefix("net.0.proj."));
    // linear2
    linear2_->load_state_dict(state_dict.get_dict_with_prefix("net.2."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    linear1_->verify_loaded_weights(prefix + "net.0.proj.");
    linear2_->verify_loaded_weights(prefix + "net.2.");
  }

 private:
  DiTLinear linear1_{nullptr};
  DiTLinear linear2_{nullptr};
  torch::nn::Functional activation_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(FeedForward);

bool ADALN_FUSE = true;

class QwenImageTransformerBlockImpl : public torch::nn::Module {
 public:
  QwenImageTransformerBlockImpl(const ModelContext& context,
                                int64_t dim,
                                int64_t num_attention_heads,
                                int64_t attention_head_dim,
                                bool zero_cond_t = false,
                                const std::string& qk_norm = "rms_norm",
                                double eps = 1e-6)
      : zero_cond_t_(zero_cond_t) {
    // Image processing modules
    img_mod_ =
        register_module("img_mod",
                        torch::nn::Sequential(torch::nn::SiLU(),
                                              DiTLinear(dim, 6 * dim, true)));

    // Image normalization
    img_norm1_ = register_module("img_norm1", AdaLayerNorm(context, dim, eps));
    // Attention module
    auto attn_ = Attention(context,
                           /*query_dim=*/dim,
                           /*cross_attention_dim=*/std::nullopt,
                           /*heads=*/num_attention_heads,
                           /*kv_heads=*/std::nullopt,
                           /*dim_head=*/attention_head_dim,
                           /*drop_out=*/0.0,
                           /*bias=*/true,
                           /*qk_norm=*/qk_norm,
                           /*cross_attention_norm=*/"",
                           /*added_kv_proj_dim=*/dim,
                           /*added_proj_bias*/ true,
                           /*out_bias*/ true,
                           /*scale_qk*/ true,
                           /*only_cross_attention=*/false,
                           eps,
                           /*rescale_output_factor=*/1.0,
                           /*residual_connection=*/false,
                           /*out_dim=*/dim,
                           /*out_context_dim=*/std::nullopt,
                           /*context_pre_only=*/true);
    attn_processor_ = register_module(
        "attn_processor_", QwenDoubleStreamAttnProcessor2_0(std::move(attn_)));
    // Image normalization 2
    img_norm2_ = register_module("img_norm2", AdaLayerNorm(context, dim, eps));

    // Image MLP
    img_mlp_ = register_module("img_mlp", FeedForward(context, dim, dim));

    // Text processing modules
    txt_mod_ =
        register_module("txt_mod",
                        torch::nn::Sequential(torch::nn::SiLU(),
                                              DiTLinear(dim, 6 * dim, true)));

    // Text normalization 1
    txt_norm1_ = register_module("txt_norm1", AdaLayerNorm(context, dim, eps));

    // Text normalization 2
    txt_norm2_ = register_module("txt_norm2", AdaLayerNorm(context, dim, eps));

    // Text MLP
    txt_mlp_ = register_module("txt_mlp", FeedForward(context, dim, dim));
  }

  std::pair<torch::Tensor, torch::Tensor> _modulate(
      const torch::Tensor& x,
      const torch::Tensor& mod_params,
      const torch::Tensor& index = torch::Tensor()) {
    // x: b l d, shift: b d, scale: b d, gate: b d
    auto chunks = mod_params.chunk(3, -1);
    auto shift = chunks[0];
    auto scale = chunks[1];
    auto gate = chunks[2];

    torch::Tensor shift_result, scale_result, gate_result;

    if (index.defined()) {
      // Assuming mod_params batch dim is 2*actual_batch (chunked into 2 parts)
      // So shift, scale, gate have shape [2*actual_batch, d]
      int64_t actual_batch = shift.size(0) / 2;

      // Split into two parts
      auto shift_0 = shift.slice(0, 0, actual_batch);
      auto shift_1 = shift.slice(0, actual_batch, shift.size(0));

      auto scale_0 = scale.slice(0, 0, actual_batch);
      auto scale_1 = scale.slice(0, actual_batch, scale.size(0));

      auto gate_0 = gate.slice(0, 0, actual_batch);
      auto gate_1 = gate.slice(0, actual_batch, gate.size(0));

      // index: [b, l] where b is actual batch size
      // Expand to [b, l, 1] to match feature dimension
      auto index_expanded = index.unsqueeze(-1);  // [b, l, 1]

      // Expand chunks to [b, 1, d] then broadcast to [b, l, d]
      auto shift_0_exp = shift_0.unsqueeze(1);  // [b, 1, d]
      auto shift_1_exp = shift_1.unsqueeze(1);  // [b, 1, d]
      auto scale_0_exp = scale_0.unsqueeze(1);
      auto scale_1_exp = scale_1.unsqueeze(1);
      auto gate_0_exp = gate_0.unsqueeze(1);
      auto gate_1_exp = gate_1.unsqueeze(1);

      // Use torch::where to select based on index
      shift_result =
          torch::where(index_expanded == 0, shift_0_exp, shift_1_exp);
      scale_result =
          torch::where(index_expanded == 0, scale_0_exp, scale_1_exp);
      gate_result = torch::where(index_expanded == 0, gate_0_exp, gate_1_exp);
    } else {
      shift_result = shift.unsqueeze(1);
      scale_result = scale.unsqueeze(1);
      gate_result = gate.unsqueeze(1);
    }

    // Apply modulation: x * (1 + scale_result) + shift_result
    auto modulated_x = x * (1 + scale_result) + shift_result;

    return {modulated_x, gate_result};
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& hidden_states,
      const torch::Tensor& encoder_hidden_states,
      const torch::Tensor& encoder_hidden_states_mask,
      const torch::Tensor& temb,
      const std::tuple<torch::Tensor, torch::Tensor>& image_rotary_emb = {},
      const std::unordered_map<std::string, torch::Tensor>&
          joint_attention_kwargs = {},
      const torch::Tensor& modulate_index = torch::Tensor()) {
    torch::Tensor hidden_states_ = hidden_states;
    torch::Tensor encoder_hidden_states_ = encoder_hidden_states;
    torch::Tensor modulate_index_ = modulate_index;
    // Get modulation parameters for both streams
    auto img_mod_params = img_mod_->forward(temb);  // [B, 6*dim]
    torch::Tensor new_temb;
    if (zero_cond_t_) {
      new_temb = temb.chunk(2, 0)[0];
    } else {
      new_temb = temb;
    }
    auto txt_mod_params = txt_mod_->forward(new_temb);  // [B, 6*dim]
    //  Split modulation parameters for norm1 and norm2
    auto img_mod_chunks = img_mod_params.chunk(2, -1);
    auto img_mod1 = img_mod_chunks[0];  // [B, 3*dim]
    auto img_mod2 = img_mod_chunks[1];  // [B, 3*dim]

    auto txt_mod_chunks = txt_mod_params.chunk(2, -1);
    auto txt_mod1 = txt_mod_chunks[0];  // [B, 3*dim]
    auto txt_mod2 = txt_mod_chunks[1];  // [B, 3*dim]

    // Process image stream - norm1 + modulation
    torch::Tensor img_modulated, img_gate1;
    std::tie(img_modulated, img_gate1) =
        img_norm1_->forward(hidden_states_, img_mod1, modulate_index_);

    //  Process text stream - norm1 + modulation
    torch::Tensor txt_modulated, txt_gate1;
    std::tie(txt_modulated, txt_gate1) =
        txt_norm1_->forward(encoder_hidden_states_, txt_mod1);

    if (attn_processor_->attn_->use_sp_) {
      auto save_tensor = [this](const torch::Tensor& tensor,
                                const std::string& name) {
        if (tensor.defined()) {
          torch::Tensor cpu_tensor = tensor.cpu();
          std::string filename = name + "_rank_" +
                                 std::to_string(attn_processor_->attn_->rank_) +
                                 ".pt";
          torch::save(cpu_tensor, filename);
        }
      };

      save_tensor(modulate_index_, "sp/modulate_index");
      save_tensor(hidden_states_,
                  "sp/hidden_states_block");  // step2  有误差， 比较大
      save_tensor(encoder_hidden_states_, "sp/encoder_hidden_states_block");
      save_tensor(img_modulated, "sp/img_modulated");  // 有误差但能接受
      save_tensor(txt_modulated, "sp/txt_modulated");
    } else {
      torch::save(modulate_index_.cpu(), "tp1/modulate_index.pt");
      torch::save(hidden_states_.cpu(), "tp1/hidden_states_block.pt");
      torch::save(encoder_hidden_states_.cpu(),
                  "tp1/encoder_hidden_states_block.pt");
      torch::save(img_modulated.cpu(), "tp1/img_modulated.pt");
      torch::save(txt_modulated.cpu(), "tp1/txt_modulated.pt");
    }

    // Use QwenAttnProcessor2_0 for joint attention computation
    auto attn_output = attn_processor_->forward(img_modulated,  // Image stream
                                                txt_modulated,  // Text stream
                                                encoder_hidden_states_mask,
                                                torch::Tensor(),  // timestep
                                                image_rotary_emb);

    // QwenAttnProcessor2_0 returns (img_output, txt_output)
    auto img_attn_output = std::get<0>(attn_output);
    auto txt_attn_output = std::get<1>(attn_output);

    //  Apply attention gates and add residual
    auto new_hidden_states = hidden_states_ + img_gate1 * img_attn_output;
    auto new_encoder_hidden_states =
        encoder_hidden_states_ + txt_gate1 * txt_attn_output;

    // Process image stream - norm2 + MLP
    torch::Tensor img_modulated2, img_gate2;
    std::tie(img_modulated2, img_gate2) =
        img_norm2_->forward(new_hidden_states, img_mod2, modulate_index_);

    auto img_mlp_output = img_mlp_->forward(img_modulated2);
    new_hidden_states = new_hidden_states + img_gate2 * img_mlp_output;

    // Process text stream - norm2 + MLP
    torch::Tensor txt_modulated2, txt_gate2;
    std::tie(txt_modulated2, txt_gate2) =
        txt_norm2_->forward(new_encoder_hidden_states, txt_mod2);

    auto txt_mlp_output = txt_mlp_->forward(txt_modulated2);
    new_encoder_hidden_states =
        new_encoder_hidden_states + txt_gate2 * txt_mlp_output;

    //  Clip to prevent overflow for fp16
    if (new_encoder_hidden_states.dtype() == torch::kFloat16) {
      new_encoder_hidden_states =
          new_encoder_hidden_states.clamp(-65504, 65504);
    }
    if (new_hidden_states.dtype() == torch::kFloat16) {
      new_hidden_states = new_hidden_states.clamp(-65504, 65504);
    }
    return std::make_tuple(new_hidden_states, new_encoder_hidden_states);
  }

  void load_state_dict(const StateDict& state_dict) {
    img_mod_[1]->as<DiTLinear>()->load_state_dict(
        state_dict.get_dict_with_prefix("img_mod.1."));
    img_mlp_->load_state_dict(state_dict.get_dict_with_prefix("img_mlp."));
    txt_mod_[1]->as<DiTLinear>()->load_state_dict(
        state_dict.get_dict_with_prefix("txt_mod.1."));
    txt_mlp_->load_state_dict(state_dict.get_dict_with_prefix("txt_mlp."));
    attn_processor_->load_state_dict(state_dict.get_dict_with_prefix("attn."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    img_mod_[1]->as<DiTLinear>()->verify_loaded_weights(prefix + "img_mod.1.");
    img_mlp_->verify_loaded_weights(prefix + "img_mlp.");
    txt_mod_[1]->as<DiTLinear>()->verify_loaded_weights(prefix + "txt_mod.1.");
    txt_mlp_->verify_loaded_weights(prefix + "txt_mlp.");
    attn_processor_->verify_loaded_weights(prefix + "attn.");
  }

  void set_text_pad_(const int pad) { attn_processor_->set_text_pad_(pad); }

  void set_img_pad_(const int pad) { attn_processor_->set_img_pad_(pad); }

  int get_img_pad_() { return attn_processor_->get_img_pad_(); }

  int get_text_pad_() { return attn_processor_->get_text_pad_(); }

 private:
  torch::nn::Sequential img_mod_{nullptr};
  AdaLayerNorm img_norm1_{nullptr};
  AdaLayerNorm img_norm2_{nullptr};
  std::shared_ptr<Attention> attn_{nullptr};
  QwenDoubleStreamAttnProcessor2_0 attn_processor_{nullptr};
  FeedForward img_mlp_{nullptr};

  torch::nn::Sequential txt_mod_{nullptr};
  AdaLayerNorm txt_norm1_{nullptr};
  AdaLayerNorm txt_norm2_{nullptr};
  FeedForward txt_mlp_{nullptr};
  bool zero_cond_t_;
};

TORCH_MODULE(QwenImageTransformerBlock);

class QwenImageTransformer2DModelImpl : public torch::nn::Module {
 public:
  QwenImageTransformer2DModelImpl(const ModelContext& context)
      : pg_(context.get_parallel_args()) {
    use_sp_ = context.get_parallel_args().world_size() > 1;
    auto model_args = context.get_model_args();
    int64_t num_attention_heads = model_args.n_heads();
    int64_t attention_head_dim = model_args.head_dim();
    int64_t joint_attention_dim = model_args.joint_attention_dim();
    std::vector<int64_t> axes_dims_rope = model_args.axes_dims_rope();
    int64_t num_layers = model_args.num_layers();
    int64_t patch_size = model_args.mm_patch_size();
    int64_t in_channels = model_args.in_channels();
    int64_t out_channels = model_args.out_channels();
    bool zero_cond_t = model_args.zero_cond_t();
    bool use_additional_t_cond = model_args.use_additional_t_cond();
    use_layer3d_rope_ = model_args.use_layer3d_rope();

    out_channels = (out_channels > 0) ? out_channels : in_channels;
    auto inner_dim = num_attention_heads * attention_head_dim;

    // Positional embedding
    if (use_layer3d_rope_) {
      pos_embed_3d_rope_ = register_module(
          "pos_embed",
          QwenEmbedLayer3DRope(context, /*theta=*/10000, axes_dims_rope, true));
    } else {
      pos_embed_ = register_module(
          "pos_embed",
          QwenEmbedRope(context, /*theta=*/10000, axes_dims_rope, true));
    }

    // Time-text embedding
    time_text_embed_ = register_module(
        "time_text_embed",
        QwenTimestepProjEmbeddings(context, inner_dim, use_additional_t_cond));

    // Text normalization
    txt_norm_ = register_module(
        "txt_norm", RMSNorm(joint_attention_dim, 1e-6, true, false));

    // Input projections
    img_in_ =
        register_module("img_in", DiTLinear(in_channels, inner_dim, true));
    txt_in_ = register_module("txt_in",
                              DiTLinear(joint_attention_dim, inner_dim, true));
    // Transformer blocks
    transformer_blocks_ =
        register_module("transformer_blocks", torch::nn::ModuleList());
    for (int64_t i = 0; i < num_layers; ++i) {
      transformer_blocks_->push_back(
          QwenImageTransformerBlock(context,
                                    inner_dim,
                                    num_attention_heads,
                                    attention_head_dim,
                                    zero_cond_t));
    }

    // Output layers
    norm_out_ = register_module(
        "norm_out",
        AdaLayerNormContinuous(context, inner_dim, inner_dim, false, 1e-6));
    proj_out_ = register_module(
        "proj_out",
        DiTLinear(inner_dim, patch_size * patch_size * out_channels, true));

    // Cache for conditional and unconditional
    cache_cond_ = false;
    cache_uncond_ = false;

    zero_cond_t_ = zero_cond_t;
  }
  torch::Tensor forward(
      const torch::Tensor& hidden_states,
      const torch::Tensor& encoder_hidden_states = torch::Tensor(),
      const torch::Tensor& encoder_hidden_states_mask = torch::Tensor(),
      torch::Tensor timestep = torch::Tensor(),
      std::vector<std::vector<int64_t>> img_shapes = {},
      torch::Tensor txt_seq_lens = torch::Tensor(),
      torch::Tensor addition_t_cond = torch::Tensor(),
      torch::Tensor guidance = torch::Tensor(),
      const std::unordered_map<std::string, torch::Tensor>& attention_kwargs =
          {},
      const std::vector<torch::Tensor>& controlnet_block_samples = {}) {
    auto new_hidden_states = img_in_->forward(hidden_states);
    auto new_timestep = timestep.to(new_hidden_states.dtype());

    torch::Tensor modulate_index;
    if (zero_cond_t_) {
      new_timestep = torch::cat({new_timestep, new_timestep * 0}, /*dim=*/0);
      std::vector<torch::Tensor> modulate_index_list;
      for (size_t sample_index = 0; sample_index < 1; sample_index++) {
        auto zero_prods = torch::zeros({img_shapes[0][1] * img_shapes[0][2]},
                                       torch::TensorOptions()
                                           .device(new_timestep.device())
                                           .dtype(torch::kInt64));
        int64_t one_prods_size = 0;
        for (size_t index = 1; index < img_shapes.size(); index++) {
          one_prods_size += img_shapes[index][1] * img_shapes[index][2];
        }
        auto ones_prods = torch::ones({one_prods_size},
                                      torch::TensorOptions()
                                          .device(new_timestep.device())
                                          .dtype(torch::kInt64));
        modulate_index_list.emplace_back(
            torch::cat({zero_prods, ones_prods}, /*dim=*/0));
      }
      modulate_index = torch::stack(modulate_index_list, /*dim=*/0);
    } else {
      modulate_index = torch::Tensor();
    }

    auto new_encoder_hidden_states = txt_norm_->forward(encoder_hidden_states);
    new_encoder_hidden_states = txt_in_->forward(new_encoder_hidden_states);

    // Use the encoder_hidden_states sequence length for RoPE computation and
    // normalize mask
    auto [text_seq_len, per_sample_len, new_encoder_hidden_states_mask] =
        compute_text_seq_len_from_mask(new_encoder_hidden_states,
                                       encoder_hidden_states_mask);

    auto temb = time_text_embed_->forward(
        new_timestep, new_hidden_states, addition_t_cond);

    std::tuple<torch::Tensor, torch::Tensor> image_rotary_emb;
    if (use_layer3d_rope_) {
      image_rotary_emb = pos_embed_3d_rope_->forward(
          img_shapes, text_seq_len, new_hidden_states.device());
    } else {
      image_rotary_emb = pos_embed_->forward(img_shapes,
                                             text_seq_len,
                                             new_hidden_states.device(),
                                             /*max_txt_seq_len=*/std::nullopt);
    }

    std::unordered_map<std::string, torch::Tensor> block_attention_kwargs;
    if (new_encoder_hidden_states_mask.has_value() &&
        new_encoder_hidden_states_mask.value().defined()) {
      int64_t batch_size = new_hidden_states.size(0);
      int64_t image_seq_len = new_hidden_states.size(1);
      auto image_mask = torch::ones({batch_size, image_seq_len},
                                    torch::TensorOptions()
                                        .device(new_hidden_states.device())
                                        .dtype(torch::kBool));
      auto joint_attention_mask = torch::cat(
          {new_encoder_hidden_states_mask.value(), image_mask}, /*dim=*/1);
      block_attention_kwargs["attention_mask"] = joint_attention_mask;
    }
    auto image_rot = std::get<0>(image_rotary_emb);
    auto txt_rot = std::get<1>(image_rotary_emb);

    int world_size_ = pg_.world_size();
    int rank_ = pg_.rank();

    if (use_sp_) {
      // 保存new_hidden_states和new_encoder_hidden_states到本地
      torch::save(new_hidden_states.cpu(),
                  "sp/new_hidden_states_before.pt");  // step 2 有误差，很大
      torch::save(new_encoder_hidden_states.cpu(),
                  "sp/new_encoder_hidden_states_before.pt");
      // split the sequence for hidden_states and the encoder_hidden_states
      int32_t seq_len, encoder_seq_len;
      seq_len = new_hidden_states.size(1);
      encoder_seq_len = new_encoder_hidden_states.size(1);
      pad_ = (world_size_ - (seq_len % world_size_)) % world_size_;
      encoder_pad_ =
          (world_size_ - (encoder_seq_len % world_size_)) % world_size_;

      new_encoder_hidden_states = split_sequence(
          new_encoder_hidden_states, world_size_, rank_, 1, encoder_pad_);
      modulate_index =
          split_sequence(modulate_index, world_size_, rank_, 1, pad_);
      new_hidden_states =
          split_sequence(new_hidden_states, world_size_, rank_, 1, pad_);
      auto save_tensor = [this](const torch::Tensor& tensor,
                                const std::string& name) {
        if (tensor.defined()) {
          torch::Tensor cpu_tensor = tensor.cpu();
          std::string filename =
              name + "_rank_" + std::to_string(pg_.rank()) + ".pt";
          torch::save(cpu_tensor, filename);
        }
      };

      save_tensor(new_hidden_states, "sp/new_hidden_states_before");
      save_tensor(new_encoder_hidden_states, "sp/new_encoder_hidden_states");
    } else {
      // 保存new_hidden_states和new_encoder_hidden_states到本地
      torch::save(new_hidden_states.cpu(), "tp1/new_hidden_states_before.pt");
      torch::save(new_encoder_hidden_states.cpu(),
                  "tp1/new_encoder_hidden_states_before.pt");
    }
    for (int64_t index_block = 0; index_block < transformer_blocks_->size();
         ++index_block) {
      // for (int64_t index_block = 0; index_block < 1; ++index_block) {
      if (use_sp_) {
        transformer_blocks_[index_block]
            ->as<QwenImageTransformerBlock>()
            ->set_text_pad_(encoder_pad_);
        transformer_blocks_[index_block]
            ->as<QwenImageTransformerBlock>()
            ->set_img_pad_(pad_);
      }
      if (use_sp_) {
        auto save_tensor = [this, index_block](const torch::Tensor& tensor,
                                               const std::string& name) {
          if (tensor.defined()) {
            torch::Tensor cpu_tensor = tensor.cpu();
            std::string filename = name + std::to_string(index_block) +
                                   "_rank_" + std::to_string(pg_.rank()) +
                                   ".pt";
            torch::save(cpu_tensor, filename);
          }
        };

        save_tensor(new_hidden_states, "sp/new_hidden_states_");
        save_tensor(new_encoder_hidden_states, "sp/new_encoder_hidden_states_");
      } else {
        // 保存new_hidden_states和new_encoder_hidden_states到本地
        torch::save(
            new_hidden_states.cpu(),
            "tp1/new_hidden_states_" + std::to_string(index_block) + ".pt");
        torch::save(new_encoder_hidden_states.cpu(),
                    "tp1/new_encoder_hidden_states_" +
                        std::to_string(index_block) + ".pt");
      }
      std::tie(new_hidden_states, new_encoder_hidden_states) =
          transformer_blocks_[index_block]
              ->as<QwenImageTransformerBlock>()
              ->forward(new_hidden_states,
                        new_encoder_hidden_states,
                        /*encoder_hidden_states_mask=*/torch::Tensor(),
                        temb,
                        image_rotary_emb,
                        block_attention_kwargs,
                        modulate_index);
    }

    if (use_sp_) {
      // gather the full sequence for hidden_states and the
      // encoder_hidden_states
      new_hidden_states =
          gather_sequence(new_hidden_states, world_size_, 1, pad_, pg_);
      new_encoder_hidden_states = gather_sequence(
          new_encoder_hidden_states, world_size_, 1, encoder_pad_, pg_);
      torch::save(new_hidden_states.cpu(), "sp/new_hidden_states_after.pt");
      torch::save(new_encoder_hidden_states.cpu(),
                  "sp/new_encoder_hidden_states_after.pt");
    } else {
      // 保存new_hidden_states和new_encoder_hidden_states到本地
      // 将张量转移到CPU后保存
      torch::save(new_hidden_states.cpu(), "tp1/new_hidden_states_after.pt");
      torch::save(new_encoder_hidden_states.cpu(),
                  "tp1/new_encoder_hidden_states_after.pt");
    }

    if (zero_cond_t_) {
      temb = temb.chunk(2, 0)[0];
    }

    new_hidden_states = norm_out_->forward(new_hidden_states, temb);
    new_hidden_states = proj_out_->forward(new_hidden_states);
    return new_hidden_states;
  }

  void verify_loaded_weights(const std::string& prefix) {
    time_text_embed_->verify_loaded_weights(prefix + "time_text_embed.");
    txt_norm_->verify_loaded_weights(prefix + "txt_norm.");
    img_in_->verify_loaded_weights(prefix + "img_in.");
    txt_in_->verify_loaded_weights(prefix + "txt_in.");
    norm_out_->verify_loaded_weights(prefix + "norm_out.");
    proj_out_->verify_loaded_weights(prefix + "proj_out.");
    for (size_t i = 0; i < transformer_blocks_->size(); i++) {
      auto block_prefix = "transformer_blocks." + std::to_string(i) + ".";
      transformer_blocks_[i]
          ->as<QwenImageTransformerBlock>()
          ->verify_loaded_weights(prefix + block_prefix);
    }
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      time_text_embed_->load_state_dict(
          state_dict->get_dict_with_prefix("time_text_embed."));
      txt_norm_->load_state_dict(state_dict->get_dict_with_prefix("txt_norm."));

      img_in_->load_state_dict(state_dict->get_dict_with_prefix("img_in."));
      txt_in_->load_state_dict(state_dict->get_dict_with_prefix("txt_in."));

      norm_out_->load_state_dict(state_dict->get_dict_with_prefix("norm_out."));
      proj_out_->load_state_dict(state_dict->get_dict_with_prefix("proj_out."));

      for (size_t i = 0; i < transformer_blocks_->size(); i++) {
        auto prefix = "transformer_blocks." + std::to_string(i) + ".";
        transformer_blocks_[i]
            ->as<QwenImageTransformerBlock>()
            ->load_state_dict(state_dict->get_dict_with_prefix(prefix));
      }
    }
    verify_loaded_weights("");
    LOG(INFO) << "qwen image vae model loaded successfully.";
  }

 private:
  QwenEmbedRope pos_embed_{nullptr};
  QwenEmbedLayer3DRope pos_embed_3d_rope_{nullptr};
  QwenTimestepProjEmbeddings time_text_embed_{nullptr};
  RMSNorm txt_norm_{nullptr};
  DiTLinear img_in_{nullptr};
  DiTLinear txt_in_{nullptr};
  torch::nn::ModuleList transformer_blocks_{nullptr};
  AdaLayerNormContinuous norm_out_{nullptr};
  DiTLinear proj_out_{nullptr};

  // Cache objects
  bool cache_cond_;
  bool cache_uncond_;

  bool zero_cond_t_;
  bool use_layer3d_rope_;
  // sp
  bool use_sp_{false};
  ParallelArgs pg_;
  int pad_{0};
  int encoder_pad_{0};
};

TORCH_MODULE(QwenImageTransformer2DModel);

REGISTER_MODEL_ARGS(QwenImageTransformer2DModel, [&] {
  // qwen-image 2509 params
  LOAD_ARG_OR(dtype, "dtype", "bfloat16");
  LOAD_ARG_OR(in_channels, "in_channels", 64);
  LOAD_ARG_OR(out_channels, "out_channels", 16);
  LOAD_ARG_OR(num_layers, "num_layers", 60);
  LOAD_ARG_OR(num_single_layers, "num_single_layers", 24);
  LOAD_ARG_OR(head_dim, "attention_head_dim", 128);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 24);
  LOAD_ARG_OR(joint_attention_dim, "joint_attention_dim", 3584);
  LOAD_ARG_OR(mm_patch_size, "patch_size", 2);
  LOAD_ARG_OR(guidance_embeds, "guidance_embeds", false);
  LOAD_ARG_OR(
      axes_dims_rope, "axes_dims_rope", (std::vector<int64_t>{16, 56, 56}));

  // qwen-image 2511 params
  LOAD_ARG_OR(zero_cond_t, "zero_cond_t", false);
  LOAD_ARG_OR(use_additional_t_cond, "use_additional_t_cond", false);
  LOAD_ARG_OR(use_layer3d_rope, "use_layer3d_rope", false);
});

}  // namespace qwenimage
}  // namespace xllm
