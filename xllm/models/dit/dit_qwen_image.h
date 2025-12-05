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
  bool weight_is_loaded_{false};
  bool bias_is_loaded_{false};
  bool is_bias_;
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
  torch::nn::SiLU silu_{nullptr};
  DiTLinear linear_{nullptr};
  torch::nn::LayerNorm norm_{nullptr};
  std::string norm_type_;
  double eps_;
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
      const torch::Tensor& mod_params) {
    auto chunks = mod_params.chunk(3, -1);
    auto shift = chunks[0];
    auto scale = chunks[1];
    auto gate = chunks[2];

    scale = (1 + scale.unsqueeze(1));
    shift = shift.unsqueeze(1);

    auto result = at_npu::native::custom_ops::npu_layer_norm_eval(
        x, {hidden_size_}, scale, shift, eps_);

    return std::make_tuple(result, gate.unsqueeze(1));
  }

 private:
  int64_t hidden_size_;
  double eps_;
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
                double scale)
      : embedding_dim(num_channels),
        flip_sin_to_cos(flip_sin_to_cos),
        downscale_freq_shift(downscale_freq_shift),
        scale(scale),
        max_period(10000) {}

  torch::Tensor forward(const torch::Tensor& timesteps) {
    CHECK(timesteps.dim() == 1) << "Timesteps should be a 1d-array";

    int64_t half_dim = embedding_dim / 2;

    auto exponent =
        -std::log(max_period) * torch::arange(0,
                                              half_dim,
                                              torch::TensorOptions()
                                                  .dtype(torch::kFloat32)
                                                  .device(timesteps.device()));

    exponent = exponent / (half_dim - downscale_freq_shift);
    auto emb = torch::exp(exponent);
    emb = timesteps.unsqueeze(1).to(torch::kFloat) * emb.unsqueeze(0);

    emb = scale * emb;

    // concat sine and cosine embeddings
    auto sin_emb = torch::sin(emb);
    auto cos_emb = torch::cos(emb);
    emb = torch::cat({sin_emb, cos_emb}, -1);
    // flip sine and cosine embeddings
    if (flip_sin_to_cos) {
      emb = torch::cat({cos_emb, sin_emb}, -1);
    }
    // zero pad
    if (embedding_dim % 2 == 1) {
      emb = torch::nn::functional::pad(
          emb, torch::nn::functional::PadFuncOptions({0, 1}));
    }
    return emb;
  }

 private:
  int64_t embedding_dim;
  bool flip_sin_to_cos;
  double downscale_freq_shift;
  double scale;
  int64_t max_period;
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

class QwenTimestepProjEmbeddingsImpl : public torch::nn::Module {
 public:
  QwenTimestepProjEmbeddingsImpl(const ModelContext& context,
                                 int64_t embedding_dim) {
    time_proj_ =
        register_module("time_proj", Timesteps(context, 256, true, 0.0, 1000));
    timestep_embedder_ = register_module(
        "timestep_embedder", TimestepEmbedding(context, 256, embedding_dim));
  }

  torch::Tensor forward(const torch::Tensor& timestep,
                        const torch::Tensor& hidden_states) {
    auto timesteps_proj = time_proj_->forward(timestep);
    auto timesteps_emb =
        timestep_embedder_->forward(timesteps_proj.to(hidden_states.dtype()));

    auto conditioning = timesteps_emb;
    return conditioning;
  }
  void load_state_dict(const StateDict& state_dict) {
    timestep_embedder_->load_state_dict(
        state_dict.get_dict_with_prefix("timestep_embedder."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    timestep_embedder_->verify_loaded_weights(prefix + "timestep_embedder.");
  }

 private:
  Timesteps time_proj_{nullptr};
  TimestepEmbedding timestep_embedder_{nullptr};
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
                                     int64_t idx = 0) {
    int64_t seq_lens = frame * height * width;

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

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const std::vector<std::vector<int64_t>>& video_fhw,
      const torch::Tensor& txt_seq_lens,
      torch::Device device) {
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

      auto video_freq = _compute_video_freqs(frame, height, width, idx);
      video_freq = video_freq.to(device);
      vid_freqs.push_back(video_freq);

      if (scale_rope_) {
        max_vid_index = std::max({height / 2, width / 2, max_vid_index});
      } else {
        max_vid_index = std::max({height, width, max_vid_index});
      }
    }

    int64_t max_len = torch::max(txt_seq_lens).item<int64_t>();
    auto txt_freqs =
        pos_freqs_.slice(0, max_vid_index, max_vid_index + max_len);

    auto vid_freqs_cat = torch::cat(vid_freqs, 0);
    return std::make_tuple(vid_freqs_cat, txt_freqs);
  }

 private:
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

  torch::Tensor _compute_video_freqs_cached(int64_t frame,
                                            int64_t height,
                                            int64_t width,
                                            int64_t idx = 0) {
    std::string key = std::to_string(idx) + "_" + std::to_string(height) + "_" +
                      std::to_string(width);

    auto it = rope_cache_.find(key);
    if (it != rope_cache_.end()) {
      return it->second;
    } else {
      auto result = _compute_video_freqs(frame, height, width, idx);
      rope_cache_[key] = result;
      return result;
    }
  }

 private:
  std::unordered_map<std::string, torch::Tensor> rope_cache_;
};
TORCH_MODULE(QwenEmbedRopeWithCache);

// A internel class that only register necessary modules for attention
// implementation The attention forward shouldn't be implemented here, but in
// processor classes
// TODO: This class should be extracted from dit class and integrated into a
// common class.
class AttentionImpl : public torch::nn::Module {
 public:
  AttentionImpl(const ModelContext context,
                int64_t query_dim,
                int64_t cross_attention_dim = -1,
                int64_t heads = 8,
                int64_t kv_heads = -1,
                int64_t dim_head = 64,
                double dropout = 0.0,
                bool bias = false,
                const std::string& qk_norm = "",
                const std::string& cross_attention_norm = "",
                int64_t added_kv_proj_dim = -1,
                bool added_proj_bias = true,
                bool out_bias = true,
                bool scale_qk = true,
                bool only_cross_attention = false,
                double eps = 1e-5,
                double rescale_output_factor = 1.0,
                bool residual_connection = false,
                int64_t out_dim = -1,
                int64_t out_context_dim = -1,
                int64_t context_pre_only = -1,
                bool pre_only = false,
                bool elementwise_affine = true,
                bool is_causal = false)
      : heads_(heads),
        bias_(bias),
        out_bias_(out_bias),
        added_proj_bias_(added_proj_bias) {
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
      CHECK(kv_heads != -1) << "qk_norm is set to: " + qk_norm +
                                   ", but get kv_heads " +
                                   std::to_string(kv_heads);
      layer_norm_q_ = register_module(
          "norm_q",
          torch::nn::LayerNorm(
              torch::nn::LayerNormOptions({dim_head * heads}).eps(eps)));
      layer_norm_k_ = register_module(
          "norm_k",
          torch::nn::LayerNorm(
              torch::nn::LayerNormOptions({dim_head * kv_heads}).eps(eps)));
    } else if (qk_norm == "rms_norm") {
      // Assuming you have an RMSNorm implementation
      norm_q_ = register_module("norm_q", RMSNorm(dim_head, eps, true, false));
      norm_k_ = register_module("norm_k", RMSNorm(dim_head, eps, true, false));
    } else if (qk_norm == "rms_norm_across_heads") {
      // LTX applies qk norm across all heads
      CHECK(kv_heads != -1) << "qk_norm is set to: " + qk_norm +
                                   ", but get kv_heads " +
                                   std::to_string(kv_heads);

      norm_q_ = register_module("norm_q", RMSNorm(dim_head, eps, true, false));
      norm_k_ = register_module("norm_k",
                                RMSNorm(dim_head * kv_heads, eps, true, false));
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
              torch::nn::LayerNormOptions({cross_attention_dim})));
    } else {
      CHECK(cross_attention_norm.empty())
          << "unknown cross_attention_norm: " + cross_attention_norm +
                 ". Should be '', 'layer_norm'";
    }

    int64_t q_dim = out_dim > -1 ? out_dim : dim_head * heads;
    int64_t kv_dim = kv_heads == -1 ? q_dim : dim_head * kv_heads;
    cross_attention_dim =
        cross_attention_dim > -1 ? cross_attention_dim : query_dim;
    out_context_dim = out_context_dim > -1 ? out_context_dim : query_dim;
    to_q_ = register_module("to_q", DiTLinear(query_dim, q_dim, bias));

    // Key-Value projections (if not only cross attention)
    if (!only_cross_attention) {
      to_k_ =
          register_module("to_k", DiTLinear(cross_attention_dim, kv_dim, bias));
      to_v_ =
          register_module("to_v", DiTLinear(cross_attention_dim, kv_dim, bias));
    }

    if (added_kv_proj_dim > -1) {
      add_k_proj_ = register_module(
          "add_k_proj", DiTLinear(added_kv_proj_dim, kv_dim, added_proj_bias));
      add_v_proj_ = register_module(
          "add_v_proj", DiTLinear(added_kv_proj_dim, kv_dim, added_proj_bias));

      if (context_pre_only != -1) {
        add_q_proj_ = register_module(
            "add_q_proj", DiTLinear(added_kv_proj_dim, q_dim, added_proj_bias));
      }
    }

    // Output projections
    if (!pre_only) {
      to_out_ = register_module("to_out", torch::nn::Sequential());
      to_out_->push_back(DiTLinear(q_dim, out_dim, out_bias));
      to_out_->push_back(
          torch::nn::Dropout(torch::nn::DropoutOptions(dropout)));
    }

    // Additional output for context
    if (context_pre_only == 1) {
      to_add_out_ = register_module(
          "to_add_out", DiTLinear(q_dim, out_context_dim, out_bias));
    }

    // Added QK normalization for added KV projections
    if (!qk_norm.empty() && added_kv_proj_dim > -1) {
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
};
TORCH_MODULE(Attention);

// Implementation of attention forward
class QwenDoubleStreamAttnProcessor2_0Impl : public torch::nn::Module {
 public:
  QwenDoubleStreamAttnProcessor2_0Impl(Attention&& attn_module) {
    attn_ = register_module("attn", std::move(attn_module));
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& hidden_states,          // Image stream
      const torch::Tensor& encoder_hidden_states,  // Text stream
      const torch::Tensor& encoder_hidden_states_mask = torch::Tensor(),
      const torch::Tensor& attention_mask = torch::Tensor(),
      const std::tuple<at::Tensor, at::Tensor>& image_rotary_emb = {}) {
    int64_t seq_txt = encoder_hidden_states.size(1);
    int64_t seq_img = hidden_states.size(1);
    // Compute QKV for image stream (sample projections)
    auto img_query = attn_->to_q_->forward(hidden_states);
    auto img_key = attn_->to_k_->forward(hidden_states);
    auto img_value = attn_->to_v_->forward(hidden_states);

    // Compute QKV for text stream (context projections)
    auto txt_query = attn_->add_q_proj_->forward(encoder_hidden_states);
    auto txt_key = attn_->add_k_proj_->forward(encoder_hidden_states);
    auto txt_value = attn_->add_v_proj_->forward(encoder_hidden_states);

    // Reshape for multi-head attention
    int64_t heads = attn_->heads_;
    auto reshape_dims = std::vector<int64_t>{heads, -1};

    img_query = img_query.unflatten(-1, reshape_dims);
    img_key = img_key.unflatten(-1, reshape_dims);
    img_value = img_value.unflatten(-1, reshape_dims);
    txt_query = txt_query.unflatten(-1, reshape_dims);
    txt_key = txt_key.unflatten(-1, reshape_dims);
    txt_value = txt_value.unflatten(-1, reshape_dims);
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

    // Apply RoPE if provided
    auto img_freqs = std::get<0>(image_rotary_emb);
    auto txt_freqs = std::get<1>(image_rotary_emb);

    img_query = apply_rotary_emb_qwen(img_query, img_freqs, false);
    img_key = apply_rotary_emb_qwen(img_key, img_freqs, false);
    txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, false);
    txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, false);

    // Concatenate for joint attention - Order: [text, image]
    auto joint_query = torch::cat({txt_query, img_query}, 1);
    auto joint_key = torch::cat({txt_key, img_key}, 1);
    auto joint_value = torch::cat({txt_value, img_value}, 1);

    auto results = at_npu::native::custom_ops::npu_fusion_attention(
        joint_query,
        joint_key,
        joint_value,
        heads,
        "BSND",
        torch::nullopt,
        torch::nullopt,
        torch::nullopt,
        pow(joint_query.size(3), -0.5),
        1.0,
        65535,
        65535);

    auto joint_hidden_states = std::get<0>(results);
    // Reshape back
    joint_hidden_states = joint_hidden_states.flatten(2, 3);
    joint_hidden_states = joint_hidden_states.to(joint_query.dtype());

    // Split attention outputs back
    auto chunks = torch::split(joint_hidden_states, {seq_txt, seq_img}, 1);
    auto txt_attn_output = chunks[0];
    auto img_attn_output = chunks[1];

    // Apply output projections
    img_attn_output = attn_->to_out_->forward(img_attn_output);

    txt_attn_output = attn_->to_add_out_->forward(txt_attn_output);
    return std::make_tuple(img_attn_output, txt_attn_output);
  }

  void load_state_dict(const StateDict& state_dict) {
    attn_->load_state_dict(state_dict);
  }

  void verify_loaded_weights(const std::string& prefix) {
    attn_->verify_loaded_weights(prefix);
  }

 private:
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

    // linear1_->to(options_);
    // linear2_->to(options_);
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
  torch::nn::Functional activation_{nullptr};
  DiTLinear linear2_{nullptr};
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
                                const std::string& qk_norm = "rms_norm",
                                double eps = 1e-6) {
    // Image processing modules
    img_mod_ =
        register_module("img_mod",
                        torch::nn::Sequential(torch::nn::SiLU(),
                                              DiTLinear(dim, 6 * dim, true)));

    // Image normalization
    img_norm1_ = register_module("img_norm1", AdaLayerNorm(context, dim, eps));
    // Attention module
    auto attn_ = Attention(context,
                           dim,
                           -1,
                           num_attention_heads,
                           -1,
                           attention_head_dim,
                           0.0,
                           true,
                           qk_norm,
                           "",
                           dim,
                           true,
                           true,
                           true,
                           false,
                           eps,
                           1.0,
                           false,
                           dim,
                           -1,
                           1);
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

  std::tuple<torch::Tensor, torch::Tensor> _modulate(
      const torch::Tensor& x,
      const torch::Tensor& mod_params) {
    auto chunks = mod_params.chunk(3, -1);
    auto shift = chunks[0];
    auto scale = chunks[1];
    auto gate = chunks[2];

    auto modulated_x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1);
    return std::make_tuple(modulated_x, gate.unsqueeze(1));
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& hidden_states,
      const torch::Tensor& encoder_hidden_states,
      const torch::Tensor& encoder_hidden_states_mask,
      const torch::Tensor& temb,
      const std::tuple<torch::Tensor, torch::Tensor>& image_rotary_emb = {},
      const std::unordered_map<std::string, torch::Tensor>&
          joint_attention_kwargs = {}) {
    // Get modulation parameters for both streams
    auto img_mod_params = img_mod_->forward(temb);  // [B, 6*dim]
    auto txt_mod_params = txt_mod_->forward(temb);  // [B, 6*dim]
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
        img_norm1_->forward(hidden_states, img_mod1);

    //  Process text stream - norm1 + modulation
    torch::Tensor txt_modulated, txt_gate1;
    std::tie(txt_modulated, txt_gate1) =
        txt_norm1_->forward(encoder_hidden_states, txt_mod1);

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
    auto new_hidden_states = hidden_states + img_gate1 * img_attn_output;
    auto new_encoder_hidden_states =
        encoder_hidden_states + txt_gate1 * txt_attn_output;

    // Process image stream - norm2 + MLP
    torch::Tensor img_modulated2, img_gate2;
    std::tie(img_modulated2, img_gate2) =
        img_norm2_->forward(new_hidden_states, img_mod2);

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
};

TORCH_MODULE(QwenImageTransformerBlock);

class QwenImageTransformer2DModelImpl : public torch::nn::Module {
 public:
  QwenImageTransformer2DModelImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto num_attention_heads = model_args.n_heads();
    auto attention_head_dim = model_args.head_dim();
    auto joint_attention_dim = model_args.joint_attention_dim();
    auto axes_dims_rope = model_args.axes_dims_rope();
    auto num_layers = model_args.num_layers();
    auto patch_size = model_args.mm_patch_size();
    auto in_channels = model_args.in_channels();
    auto out_channels = model_args.out_channels();

    out_channels = (out_channels > 0) ? out_channels : in_channels;
    auto inner_dim = num_attention_heads * attention_head_dim;

    // Positional embedding
    pos_embed_ = register_module(
        "pos_embed", QwenEmbedRope(context, 10000, axes_dims_rope, true));

    // Time-text embedding
    time_text_embed_ = register_module(
        "time_text_embed", QwenTimestepProjEmbeddings(context, inner_dim));

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
      transformer_blocks_->push_back(QwenImageTransformerBlock(
          context, inner_dim, num_attention_heads, attention_head_dim));
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
  }
  torch::Tensor forward(
      const torch::Tensor& hidden_states,
      const torch::Tensor& encoder_hidden_states = torch::Tensor(),
      const torch::Tensor& encoder_hidden_states_mask = torch::Tensor(),
      const torch::Tensor& timestep = torch::Tensor(),
      const std::vector<std::vector<int64_t>>& img_shapes = {},
      const torch::Tensor& txt_seq_lens = torch::Tensor(),
      const torch::Tensor& guidance = torch::Tensor(),
      const std::unordered_map<std::string, torch::Tensor>& attention_kwargs =
          {},
      const std::vector<torch::Tensor>& controlnet_block_samples = {},
      bool return_dict = true,
      bool use_cache = false,
      bool if_cond = true) {
    auto new_hidden_states = img_in_->forward(hidden_states);
    auto new_timestep = timestep.to(new_hidden_states.dtype());
    auto new_encoder_hidden_states = txt_norm_->forward(encoder_hidden_states);
    new_encoder_hidden_states = txt_in_->forward(new_encoder_hidden_states);
    auto temb = time_text_embed_->forward(new_timestep, new_hidden_states);
    auto image_rotary_emb = pos_embed_->forward(
        img_shapes, txt_seq_lens, new_hidden_states.device());
    auto image_rot = std::get<0>(image_rotary_emb);
    auto txt_rot = std::get<1>(image_rotary_emb);

    for (int64_t index_block = 0; index_block < transformer_blocks_->size();
         ++index_block) {
      std::tie(new_hidden_states, new_encoder_hidden_states) =
          transformer_blocks_[index_block]
              ->as<QwenImageTransformerBlock>()
              ->forward(new_hidden_states,
                        new_encoder_hidden_states,
                        encoder_hidden_states_mask,
                        temb,
                        image_rotary_emb,
                        attention_kwargs);
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
};

TORCH_MODULE(QwenImageTransformer2DModel);

REGISTER_MODEL_ARGS(QwenImageTransformer2DModel, [&] {
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
});

}  // namespace qwenimage
}  // namespace xllm
