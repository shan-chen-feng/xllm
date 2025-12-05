#pragma once
#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/normalization.h>
#include <torch/torch.h>
#include <torch_npu/csrc/aten/CustomFunctions.h>
#include <torch_npu/csrc/libs/init_npu.h>

#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/framework/dit_model_loader.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/state_dict/state_dict.h"
#include "dit_linear.h"
#include "framework/model_context.h"
#include "models/model_registry.h"

#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

// VAE model compatible with huggingface weights
// ref to:
// https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage.py

namespace xllm {
namespace qwenimage {

// TODO: This class should be extracted from autoencoder class and integrated
// into a common class.
torch::Tensor randn_tensor(const std::vector<int64_t>& shape,
                           int64_t seed,
                           torch::TensorOptions& options) {
  if (shape.empty()) {
    LOG(FATAL) << "Shape must not be empty.";
  }
  at::Generator gen = at::detail::createCPUGenerator();
  gen = gen.clone();
  gen.set_current_seed(seed);
  torch::Tensor latents;
  latents = torch::randn(shape,
                         gen,
                         options.device(torch::kCPU)
                             .dtype(torch::kFloat32)
                             .layout(torch::kStrided));
  latents = latents.to(options);
  return latents;
}

// TODO: This class should be extracted from autoencoder class and integrated
// into a common class.
class VAEImageProcessorImpl : public torch::nn::Module {
 public:
  explicit VAEImageProcessorImpl(ModelContext context,
                                 bool do_resize = true,
                                 bool do_normalize = true,
                                 bool do_binarize = false,
                                 bool do_convert_rgb = false,
                                 bool do_convert_grayscale = false,
                                 int64_t latent_channels = 4) {
    const auto& model_args = context.get_model_args();
    dtype_ = context.get_tensor_options().dtype().toScalarType();
    scale_factor_ = 1 << model_args.block_out_channels().size();
    latent_channels_ = latent_channels;
    do_resize_ = do_resize;
    do_normalize_ = do_normalize;
    do_binarize_ = do_binarize;
    do_convert_rgb_ = do_convert_rgb;
    do_convert_grayscale_ = do_convert_grayscale;
  }

  std::pair<int64_t, int64_t> adjust_dimensions(int64_t height,
                                                int64_t width) const {
    height = height - (height % scale_factor_);
    width = width - (width % scale_factor_);
    return {height, width};
  }

  torch::Tensor preprocess(
      const torch::Tensor& image,
      std::optional<int64_t> height = std::nullopt,
      std::optional<int64_t> width = std::nullopt,
      const std::string& resize_mode = "default",
      std::optional<std::tuple<int64_t, int64_t, int64_t, int64_t>>
          crop_coords = std::nullopt) {
    torch::Tensor processed = image.clone();
    if (processed.dtype() != torch::kFloat32) {
      processed = processed.to(torch::kFloat32);
    }
    if (processed.max().item<float>() > 1.1f) {
      processed = processed / 255.0f;
    }
    if (crop_coords.has_value()) {
      auto [x1, y1, x2, y2] = crop_coords.value();
      x1 = std::max(int64_t(0), x1);
      y1 = std::max(int64_t(0), y1);
      x2 = std::min(processed.size(-1), x2);
      y2 = std::min(processed.size(-2), y2);

      if (processed.dim() == 3) {
        processed = processed.index({torch::indexing::Slice(),
                                     torch::indexing::Slice(y1, y2),
                                     torch::indexing::Slice(x1, x2)});
      } else if (processed.dim() == 4) {
        processed = processed.index({torch::indexing::Slice(),
                                     torch::indexing::Slice(),
                                     torch::indexing::Slice(y1, y2),
                                     torch::indexing::Slice(x1, x2)});
      }
    }
    int channel = processed.size(1);
    if (channel == latent_channels_) {
      return image;
    }
    auto [target_h, target_w] =
        get_default_height_width(processed, height, width);
    if (do_resize_) {
      processed = resize(processed, target_h, target_w);
    }

    if (do_normalize_) {
      processed = normalize(processed);
    }
    if (do_binarize_) {
      processed = (processed >= 0.5f).to(torch::kFloat32);
    }
    processed = processed.to(dtype_);
    return processed;
  }

  torch::Tensor postprocess(
      const torch::Tensor& tensor,
      const std::string& output_type = "pt",
      std::optional<std::vector<bool>> do_denormalize = std::nullopt) {
    torch::Tensor processed = tensor.clone();
    if (do_normalize_) {
      if (!do_denormalize.has_value()) {
        processed = denormalize(processed);
      } else {
        for (int64_t i = 0; i < processed.size(0); ++i) {
          if (i < do_denormalize.value().size() && do_denormalize.value()[i]) {
            processed[i] = denormalize(processed[i]);
          }
        }
      }
    }
    if (output_type == "np") {
      return processed.permute({0, 2, 3, 1}).contiguous();
    }
    return processed;
  }

 private:
  std::pair<int64_t, int64_t> get_default_height_width(
      const torch::Tensor& image,
      std::optional<int64_t> height = std::nullopt,
      std::optional<int64_t> width = std::nullopt) const {
    int64_t h, w;
    if (image.dim() == 3) {
      h = image.size(1);
      w = image.size(2);
    } else if (image.dim() == 4) {
      h = image.size(2);
      w = image.size(3);
    } else {
      LOG(FATAL) << "Unsupported image dimension: " << image.dim();
    }

    int64_t target_h = height.value_or(h);
    int64_t target_w = width.value_or(w);
    return adjust_dimensions(target_h, target_w);
  }

  torch::Tensor normalize(const torch::Tensor& tensor) const {
    return 2.0 * tensor - 1.0;
  }

  torch::Tensor denormalize(const torch::Tensor& tensor) const {
    return (tensor * 0.5 + 0.5).clamp(0.0, 1.0);
  }

 public:
  torch::Tensor resize(const torch::Tensor& image,
                       int64_t target_height,
                       int64_t target_width) const {
    return torch::nn::functional::interpolate(
        image,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{target_height, target_width})
            .mode(torch::kNearest));
  }

 private:
  int scale_factor_ = 8;
  int latent_channels_ = 4;
  bool do_resize_ = true;
  bool do_normalize_ = true;
  bool do_binarize_ = false;
  bool do_convert_rgb_ = false;
  bool do_convert_grayscale_ = false;
  torch::ScalarType dtype_ = torch::kFloat32;
};
TORCH_MODULE(VAEImageProcessor);

class QwenImageBaseModule : public torch::nn::Module {
 public:
  virtual torch::Tensor forward(const torch::Tensor& x,
                                std::vector<torch::Tensor>* feat_cache,
                                std::vector<int64_t>* feat_idx) = 0;
  virtual ~QwenImageBaseModule() = default;
};

const int64_t CACHE_T = 2;

class QwenImageCausalConv3dImpl : public torch::nn::Module {
 public:
  QwenImageCausalConv3dImpl(const ModelContext& context,
                            int64_t in_channels,
                            int64_t out_channels,
                            torch::IntArrayRef kernel_size,
                            torch::IntArrayRef stride = 1,
                            torch::IntArrayRef padding = 0) {
    conv_ = register_module(
        "conv",
        torch::nn::Conv3d(
            torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(0)
                .bias(true)));

    auto p = padding.size() == 1
                 ? std::vector<int64_t>{padding[0], padding[0], padding[0]}
                 : std::vector<int64_t>(padding.begin(), padding.end());

    padding_ = {p[2], p[2], p[1], p[1], 2 * p[0], 0};
  }

  torch::Tensor forward(const torch::Tensor& x,
                        const torch::Tensor& cache_x = torch::Tensor()) {
    auto padding_vec = padding_;
    auto result_x = x;

    if (cache_x.defined() && padding_[4] > 0) {
      auto device_x = result_x.device();
      auto cache_device = cache_x.to(device_x);
      result_x = torch::cat({cache_device, result_x}, 2);
      padding_vec[4] -= cache_x.size(2);
    }

    result_x = torch::nn::functional::pad(
        result_x, torch::nn::functional::PadFuncOptions(padding_vec));
    return conv_(result_x);
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict, "weight", conv_->weight, is_weight_loaded_);
    weight::load_weight(state_dict, "bias", conv_->bias, is_bias_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(is_weight_loaded_)
        << "weight is not loaded for " << prefix + "weight";
    CHECK(is_bias_loaded_) << "weight is not loaded for " << prefix + "bias";
  }

 private:
  bool is_weight_loaded_{false};
  bool is_bias_loaded_{false};
  torch::nn::Conv3d conv_ = nullptr;
  std::vector<int64_t> padding_;
};

TORCH_MODULE(QwenImageCausalConv3d);

class QwenImageRMS_normImpl : public torch::nn::Module {
 public:
  QwenImageRMS_normImpl(const ModelContext& context,
                        int64_t dim,
                        bool channel_first = true,
                        bool images = true,
                        bool is_bias = false,
                        bool fused = false)
      : channel_first_(channel_first), fused_(fused), is_bias_(is_bias) {
    auto broadcastable_dims =
        images ? std::vector<int64_t>{1, 1} : std::vector<int64_t>{1, 1, 1};
    auto shape = std::vector<int64_t>{dim};
    if (channel_first) {
      shape.insert(
          shape.end(), broadcastable_dims.begin(), broadcastable_dims.end());
    }

    scale_ = std::sqrt(dim);
    weight_ = register_parameter("gamma", torch::ones(shape));

    if (is_bias_) {
      bias_ = register_parameter("bias", torch::zeros(shape));
    }
  }

  torch::Tensor forward(const torch::Tensor& x) {
    if (fused_) {
      auto [output, rstd] =
          at_npu::native::custom_ops::npu_rms_norm(x, weight_, 0);

      if (is_bias_ && bias_.defined()) {
        output = output + bias_.to(output.device());
      }
      return output;
    } else {
      auto output = torch::nn::functional::normalize(
                        x,
                        torch::nn::functional::NormalizeFuncOptions().dim(
                            channel_first_ ? 1 : -1)) *
                    scale_ * weight_;
      if (is_bias_) {
        output = output + bias_;
      }
      return output;
    }
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict, "gamma", weight_, is_weight_loaded_);
    if (is_bias_) {
      weight::load_weight(state_dict, "bias", bias_, is_bias_loaded_);
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(is_weight_loaded_)
        << "weight is not loaded for " << prefix + "weight";
    CHECK(!is_bias_ || is_bias_loaded_)
        << "bias is not loaded for " << prefix + "bias";
  }

 private:
  bool channel_first_;
  double scale_;
  bool is_bias_;
  bool fused_;
  bool is_weight_loaded_{false};
  bool is_bias_loaded_{false};
  torch::Tensor weight_;
  torch::Tensor bias_;
  torch::TensorOptions options_;
};

TORCH_MODULE(QwenImageRMS_norm);

class QwenImageUpsampleImpl : public torch::nn::Module {
 public:
  QwenImageUpsampleImpl(
      const ModelContext& context,
      const torch::nn::functional::InterpolateFuncOptions options)
      : options_(options) {}

  torch::Tensor forward(const torch::Tensor& x) {
    // auto result = upsample_(x.to(torch::kFloat));
    auto result =
        torch::nn::functional::interpolate(x.to(torch::kFloat), options_);
    return result.to(x.dtype());
  }

 private:
  torch::nn::functional::InterpolateFuncOptions options_;
  torch::nn::Upsample upsample_ = nullptr;
};

TORCH_MODULE(QwenImageUpsample);

class QwenImageResampleImpl : public QwenImageBaseModule {
 public:
  QwenImageResampleImpl(const ModelContext& context,
                        int64_t dim,
                        const std::string& mode)
      : dim(dim), mode(mode) {
    if (mode == "upsample2d") {
      resample = register_module(
          "resample",
          torch::nn::Sequential(
              QwenImageUpsample(context,
                                torch::nn::functional::InterpolateFuncOptions()
                                    .scale_factor(std::vector<double>{2.0, 2.0})
                                    .mode(torch::kNearestExact)),
              torch::nn::Conv2d(
                  torch::nn::Conv2dOptions(dim, dim / 2, 3).padding(1))));
    } else if (mode == "upsample3d") {
      resample = register_module(
          "resample",
          torch::nn::Sequential(
              QwenImageUpsample(context,
                                torch::nn::functional::InterpolateFuncOptions()
                                    .scale_factor(std::vector<double>{2.0, 2.0})
                                    .mode(torch::kNearestExact)),
              torch::nn::Conv2d(
                  torch::nn::Conv2dOptions(dim, dim / 2, 3).padding(1))));

      time_conv =
          register_module("time_conv",
                          QwenImageCausalConv3d(context,
                                                dim,
                                                dim * 2,
                                                torch::IntArrayRef{3, 1, 1},
                                                torch::IntArrayRef{1, 1, 1},
                                                torch::IntArrayRef{1, 0, 0}));

    } else if (mode == "downsample2d") {
      resample = register_module(
          "resample",
          torch::nn::Sequential(
              torch::nn::ZeroPad2d(torch::nn::ZeroPad2dOptions({0, 1, 0, 1})),
              torch::nn::Conv2d(
                  torch::nn::Conv2dOptions(dim, dim, 3).stride(2))));
    } else if (mode == "downsample3d") {
      resample = register_module(
          "resample",
          torch::nn::Sequential(
              torch::nn::ZeroPad2d(torch::nn::ZeroPad2dOptions({0, 1, 0, 1})),
              torch::nn::Conv2d(
                  torch::nn::Conv2dOptions(dim, dim, 3).stride(2))));
      time_conv =
          register_module("time_conv",
                          QwenImageCausalConv3d(context,
                                                dim,
                                                dim,
                                                torch::IntArrayRef{3, 1, 1},
                                                torch::IntArrayRef{2, 1, 1},
                                                torch::IntArrayRef{0, 0, 0}));
    } else {
      resample = register_module("resample",
                                 torch::nn::Sequential(torch::nn::Identity()));
    }

    rep_tensor = register_parameter("rep_tensor", torch::tensor({-999.0}));
  }

  torch::Tensor forward(const torch::Tensor& x,
                        std::vector<torch::Tensor>* feat_cache = nullptr,
                        std::vector<int64_t>* feat_idx = {0}) override {
    auto sizes = x.sizes();
    auto b = sizes[0], c = sizes[1], t = sizes[2], h = sizes[3], w = sizes[4];
    auto result_x = x;

    if (mode == "upsample3d" && feat_cache && feat_idx) {
      auto idx = (*feat_idx)[0];

      if (idx < feat_cache->size() && feat_cache->at(idx).defined()) {
        auto cache_x = result_x
                           .index({torch::indexing::Slice(),
                                   torch::indexing::Slice(),
                                   torch::indexing::Slice(
                                       -CACHE_T, torch::indexing::None)})
                           .clone();

        if (cache_x.size(2) < 2 && feat_cache->at(idx).defined() &&
            !torch::equal(rep_tensor, feat_cache->at(idx))) {
          auto last_frame =
              feat_cache->at(idx)
                  .index({torch::indexing::Slice(),
                          torch::indexing::Slice(),
                          torch::indexing::Slice(-1, torch::indexing::None)})
                  .unsqueeze(2)
                  .to(cache_x.device());
          cache_x = torch::cat({last_frame, cache_x}, 2);
        }
        if (cache_x.size(2) < 2 && feat_cache->at(idx).defined() &&
            torch::equal(rep_tensor, feat_cache->at(idx))) {
          cache_x = torch::cat(
              {torch::zeros_like(cache_x).to(cache_x.device()), cache_x}, 2);
        }
        if (torch::equal(rep_tensor, feat_cache->at(idx))) {
          result_x = time_conv->forward(result_x);
        } else {
          result_x = time_conv->forward(result_x, feat_cache->at(idx));
        }
        feat_cache->at(idx) = cache_x;
        (*feat_idx)[0]++;

        result_x = result_x.reshape({b, 2, c, t, h, w});
        result_x = torch::stack({result_x.index({torch::indexing::Slice(), 0}),
                                 result_x.index({torch::indexing::Slice(), 1})},
                                3);
        result_x = result_x.reshape({b, c, t * 2, h, w});
      } else {
        feat_cache->at(idx) = rep_tensor;
        (*feat_idx)[0]++;
      }
    }

    t = result_x.size(2);
    result_x = result_x.permute({0, 2, 1, 3, 4}).reshape({b * t, c, h, w});
    result_x = resample->forward(result_x);
    result_x =
        result_x
            .view({b, t, result_x.size(1), result_x.size(2), result_x.size(3)})
            .permute({0, 2, 1, 3, 4});

    if (mode == "downsample3d" && feat_cache && feat_idx) {
      auto idx = (*feat_idx)[0];

      if (idx < feat_cache->size() && feat_cache->at(idx).defined()) {
        auto cache_x =
            result_x
                .index({torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(-1, torch::indexing::None)})
                .clone();

        auto concat_x = torch::cat(
            {feat_cache->at(idx).index(
                 {torch::indexing::Slice(),
                  torch::indexing::Slice(),
                  torch::indexing::Slice(-1, torch::indexing::None)}),
             result_x},
            2);

        result_x = time_conv->forward(concat_x);
        feat_cache->at(idx) = cache_x;
        (*feat_idx)[0]++;
      } else {
        feat_cache->at(idx) = result_x.clone();
        (*feat_idx)[0]++;
      }
    }

    return result_x;
  }

  void load_state_dict(const StateDict& state_dict) {
    auto params = resample->named_parameters();
    for (auto& param : params) {
      std::string name = param.key();
      if (name == "1.weight") {
        weight::load_weight(
            state_dict, "resample.1.weight", param.value(), is_weight_loaded_);
      } else if (name == "1.bias") {
        weight::load_weight(
            state_dict, "resample.1.bias", param.value(), is_bias_loaded_);
      }
    }
    if (time_conv) {
      time_conv->load_state_dict(state_dict.get_dict_with_prefix("time_conv."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(is_weight_loaded_)
        << "weight is not loaded for " << prefix + "weight";
    CHECK(is_bias_loaded_) << "bias is not loaded for " << prefix + "bias";
    if (time_conv) {
      time_conv->verify_loaded_weights("time_conv.");
    }
  }

 private:
  int64_t dim;
  std::string mode;
  bool is_weight_loaded_{false};
  bool is_bias_loaded_{false};
  torch::Tensor rep_tensor;
  torch::nn::Sequential resample{nullptr};
  QwenImageCausalConv3d time_conv{nullptr};
};

TORCH_MODULE(QwenImageResample);

class QwenImageResidualBlockImpl : public QwenImageBaseModule {
 public:
  QwenImageResidualBlockImpl(const ModelContext& context,
                             int64_t in_dim,
                             int64_t out_dim,
                             double dropout = 0.0,
                             const std::string& non_linearity = "silu")
      : in_dim(in_dim), out_dim(out_dim) {
    activation = register_module("silu", torch::nn::SiLU());

    norm1 = register_module(
        "norm1", QwenImageRMS_norm(context, in_dim, true, false, false, false));
    conv1 = register_module("conv1",
                            QwenImageCausalConv3d(context,
                                                  in_dim,
                                                  out_dim,
                                                  torch::IntArrayRef{3, 3, 3},
                                                  torch::IntArrayRef{1, 1, 1},
                                                  torch::IntArrayRef{1, 1, 1}));
    norm2 = register_module(
        "norm2",
        QwenImageRMS_norm(context, out_dim, true, false, false, false));
    dropout_layer = register_module("dropout", torch::nn::Dropout(dropout));
    conv2 = register_module("conv2",
                            QwenImageCausalConv3d(context,
                                                  out_dim,
                                                  out_dim,
                                                  torch::IntArrayRef{3, 3, 3},
                                                  torch::IntArrayRef{1, 1, 1},
                                                  torch::IntArrayRef{1, 1, 1}));

    if (in_dim != out_dim) {
      conv_shortcut =
          register_module("conv_shortcut",
                          QwenImageCausalConv3d(context,
                                                in_dim,
                                                out_dim,
                                                torch::IntArrayRef{1, 1, 1},
                                                torch::IntArrayRef{1, 1, 1},
                                                torch::IntArrayRef{0, 0, 0}));
    } else {
      identity = register_module("conv_shortcut", torch::nn::Identity());
    }
  }

  torch::Tensor forward(const torch::Tensor& x,
                        std::vector<torch::Tensor>* feat_cache = nullptr,
                        std::vector<int64_t>* feat_idx = {0}) override {
    torch::Tensor h = torch::empty({0});
    if (conv_shortcut) {
      h = conv_shortcut->forward(x);
    } else {
      h = identity->forward(x);
    }
    auto result_x = x;

    result_x = norm1->forward(result_x);
    result_x = activation->forward(result_x);

    if (feat_cache && feat_idx) {
      auto idx = (*feat_idx)[0];
      auto cache_x =
          result_x
              .index({torch::indexing::Slice(),
                      torch::indexing::Slice(),
                      torch::indexing::Slice(-CACHE_T, torch::indexing::None)})
              .clone();
      if (cache_x.size(2) < 2 && feat_cache->at(idx).defined()) {
        auto last_frame =
            feat_cache->at(idx)
                .index({torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(-1, torch::indexing::None)})
                .unsqueeze(2)
                .to(cache_x.device());
        cache_x = torch::cat({last_frame, cache_x}, 2);
      }

      result_x = conv1->forward(result_x, feat_cache->at(idx));
      feat_cache->at(idx) = cache_x;
      (*feat_idx)[0]++;
    } else {
      result_x = conv1->forward(result_x);
    }
    result_x = norm2->forward(result_x);
    result_x = activation->forward(result_x);
    result_x = dropout_layer->forward(result_x);

    if (feat_cache && feat_idx) {
      auto idx = (*feat_idx)[0];
      auto cache_x =
          result_x
              .index({torch::indexing::Slice(),
                      torch::indexing::Slice(),
                      torch::indexing::Slice(-CACHE_T, torch::indexing::None)})
              .clone();

      if (cache_x.size(2) < 2 && feat_cache->at(idx).defined()) {
        auto last_frame =
            feat_cache->at(idx)
                .index({torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(-1, torch::indexing::None)})
                .unsqueeze(2)
                .to(cache_x.device());
        cache_x = torch::cat({last_frame, cache_x}, 2);
      }
      result_x = conv2->forward(result_x, feat_cache->at(idx));
      feat_cache->at(idx) = cache_x;
      (*feat_idx)[0]++;
    } else {
      result_x = conv2->forward(result_x);
    }

    return result_x + h;
  }

  void load_state_dict(const StateDict& state_dict) {
    norm1->load_state_dict(state_dict.get_dict_with_prefix("norm1."));
    norm2->load_state_dict(state_dict.get_dict_with_prefix("norm2."));

    conv1->load_state_dict(state_dict.get_dict_with_prefix("conv1."));

    conv2->load_state_dict(state_dict.get_dict_with_prefix("conv2."));

    if (conv_shortcut) {
      conv_shortcut->load_state_dict(
          state_dict.get_dict_with_prefix("conv_shortcut."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    norm1->verify_loaded_weights("norm1.");
    norm2->verify_loaded_weights("norm2.");
    conv1->verify_loaded_weights("conv1.");
    conv2->verify_loaded_weights("conv2.");
    if (conv_shortcut) {
      conv_shortcut->verify_loaded_weights("conv_shortcut.");
    }
  }

 private:
  int64_t in_dim, out_dim;
  QwenImageRMS_norm norm1{nullptr}, norm2{nullptr};
  QwenImageCausalConv3d conv1{nullptr}, conv2{nullptr};
  QwenImageCausalConv3d conv_shortcut{nullptr};
  torch::nn::Dropout dropout_layer{nullptr};
  torch::nn::SiLU activation{nullptr};
  torch::nn::Identity identity{nullptr};
};

TORCH_MODULE(QwenImageResidualBlock);

class QwenImageAttentionBlockImpl : public QwenImageBaseModule {
 public:
  QwenImageAttentionBlockImpl(const ModelContext& context, int64_t dim)
      : dim(dim) {
    norm = register_module(
        "norm", QwenImageRMS_norm(context, dim, true, true, false, false));
    to_qkv = register_module(
        "to_qkv", torch::nn::Conv2d(torch::nn::Conv2dOptions(dim, dim * 3, 1)));
    proj = register_module(
        "proj", torch::nn::Conv2d(torch::nn::Conv2dOptions(dim, dim, 1)));
  }

  torch::Tensor forward(const torch::Tensor& x,
                        std::vector<torch::Tensor>* feat_cache = nullptr,
                        std::vector<int64_t>* feat_idx = nullptr) override {
    auto identity = x;
    auto sizes = x.sizes();
    auto b = sizes[0], c = sizes[1], t = sizes[2], h = sizes[3], w = sizes[4];

    auto reshaped_x = x.permute({0, 2, 1, 3, 4}).reshape({b * t, c, h, w});
    reshaped_x = norm->forward(reshaped_x);

    auto qkv = to_qkv->forward(reshaped_x);
    qkv = qkv.reshape({b * t, 1, c * 3, h * w});
    qkv = qkv.permute({0, 1, 3, 2}).contiguous();

    auto chunks = qkv.chunk(3, -1);
    auto q = chunks[0], k = chunks[1], v = chunks[2];

    auto results =
        at_npu::native::custom_ops::npu_fusion_attention(q,
                                                         k,
                                                         v,
                                                         1,
                                                         "BNSD",
                                                         torch::nullopt,
                                                         torch::nullopt,
                                                         torch::nullopt,
                                                         pow(c, -0.5),
                                                         1.0,
                                                         65535,
                                                         65535);
    auto attn_output = std::get<0>(results);
    attn_output =
        attn_output.squeeze(1).permute({0, 2, 1}).reshape({b * t, c, h, w});

    auto output = proj->forward(attn_output);

    output = output.view({b, t, c, h, w}).permute({0, 2, 1, 3, 4});

    return output + identity;
  }

  void load_state_dict(const StateDict& state_dict) {
    norm->load_state_dict(state_dict.get_dict_with_prefix("norm."));

    weight::load_weight(
        state_dict, "to_qkv.weight", to_qkv->weight, is_qkv_weight_loaded_);
    weight::load_weight(
        state_dict, "to_qkv.bias", to_qkv->bias, is_qkv_bias_loaded_);
    weight::load_weight(
        state_dict, "proj.weight", proj->weight, is_proj_weight_loaded_);
    weight::load_weight(
        state_dict, "proj.bias", proj->bias, is_proj_bias_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) {
    norm->verify_loaded_weights("norm.");

    CHECK(is_qkv_weight_loaded_)
        << "weight is not loaded for " << prefix + "weight";
    CHECK(is_qkv_bias_loaded_)
        << "weight is not loaded for " << prefix + "bias";
    CHECK(is_proj_weight_loaded_)
        << "weight is not loaded for " << prefix + "weight";
    CHECK(is_proj_bias_loaded_)
        << "weight is not loaded for " << prefix + "bias";
  }

 private:
  int64_t dim;
  QwenImageRMS_norm norm{nullptr};
  torch::nn::Conv2d to_qkv{nullptr};
  torch::nn::Conv2d proj{nullptr};
  bool is_qkv_weight_loaded_{false};
  bool is_qkv_bias_loaded_{false};
  bool is_proj_weight_loaded_{false};
  bool is_proj_bias_loaded_{false};
};

TORCH_MODULE(QwenImageAttentionBlock);

class QwenImageMidBlockImpl : public torch::nn::Module {
 public:
  QwenImageMidBlockImpl(const ModelContext& context,
                        int64_t dim,
                        double dropout = 0.0,
                        const std::string& non_linearity = "silu",
                        int64_t num_layers = 1)
      : dim(dim) {
    resnets = register_module("resnets", torch::nn::ModuleList());
    attentions = register_module("attentions", torch::nn::ModuleList());

    auto resnet_0 =
        QwenImageResidualBlock(context, dim, dim, dropout, non_linearity);
    resnets->push_back(resnet_0);

    for (int64_t i = 0; i < num_layers; i++) {
      auto attention = QwenImageAttentionBlock(context, dim);
      attentions->push_back(attention);

      auto resnet =
          QwenImageResidualBlock(context, dim, dim, dropout, non_linearity);
      resnets->push_back(resnet);
    }
  }

  torch::Tensor forward(const torch::Tensor& x,
                        std::vector<torch::Tensor>* feat_cache = nullptr,
                        std::vector<int64_t>* feat_idx = {0}) {
    auto result_x = x;

    result_x = resnets[0]->as<QwenImageResidualBlock>()->forward(
        result_x, feat_cache, feat_idx);

    for (size_t i = 0; i < attentions->size(); i++) {
      result_x =
          attentions[i]->as<QwenImageAttentionBlock>()->forward(result_x);
      result_x = resnets[i + 1]->as<QwenImageResidualBlock>()->forward(
          result_x, feat_cache, feat_idx);
    }

    return result_x;
  }

  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < resnets->size(); i++) {
      auto prefix = "resnets." + std::to_string(i) + ".";
      resnets[i]->as<QwenImageResidualBlock>()->load_state_dict(
          state_dict.get_dict_with_prefix(prefix));
    }

    for (size_t i = 0; i < attentions->size(); i++) {
      auto prefix = "attentions." + std::to_string(i) + ".";
      attentions[i]->as<QwenImageAttentionBlock>()->load_state_dict(
          state_dict.get_dict_with_prefix(prefix));
    }
  }

  void verify_loaded_weights(const std::string& prefix) {
    for (size_t i = 0; i < resnets->size(); i++) {
      auto prefix = "resnets." + std::to_string(i) + ".";
      resnets[i]->as<QwenImageResidualBlock>()->verify_loaded_weights(prefix);
    }

    for (size_t i = 0; i < attentions->size(); i++) {
      auto prefix = "attentions." + std::to_string(i) + ".";
      attentions[i]->as<QwenImageAttentionBlock>()->verify_loaded_weights(
          prefix);
    }
  }

 private:
  int64_t dim;
  torch::nn::ModuleList resnets;
  torch::nn::ModuleList attentions;
};

TORCH_MODULE(QwenImageMidBlock);

class QwenImageEncoder3dImpl : public torch::nn::Module {
 public:
  QwenImageEncoder3dImpl(const ModelContext& context,
                         int64_t dim = 128,
                         int64_t z_dim = 4,
                         std::vector<int64_t> dim_mult = {1, 2, 4, 4},
                         int64_t num_res_blocks = 2,
                         std::vector<double> attn_scales = {},
                         std::vector<bool> temperal_downsample = {true,
                                                                  true,
                                                                  false},
                         double dropout = 0.0,
                         const std::string& non_linearity = "silu")
      : dim(dim),
        z_dim(z_dim),
        dim_mult(dim_mult),
        num_res_blocks(num_res_blocks),
        attn_scales(attn_scales),
        temperal_downsample(temperal_downsample) {
    nonlinearity = register_module("silu", torch::nn::SiLU());

    std::vector<int64_t> dims = {dim * 1};
    for (auto u : dim_mult) {
      dims.push_back(dim * u);
    }

    double scale = 1.0;

    conv_in =
        register_module("conv_in",
                        QwenImageCausalConv3d(context,
                                              3,
                                              dims[0],
                                              torch::IntArrayRef{3, 3, 3},
                                              torch::IntArrayRef{1, 1, 1},
                                              torch::IntArrayRef{1, 1, 1}));

    down_blocks = register_module("down_blocks", torch::nn::ModuleList());

    size_t counter = 0;
    for (size_t i = 0; i < dims.size() - 1; i++) {
      int64_t in_dim = dims[i];
      int64_t out_dim = dims[i + 1];

      for (int64_t j = 0; j < num_res_blocks; j++) {
        auto res_block = QwenImageResidualBlock(
            context, in_dim, out_dim, dropout, non_linearity);
        down_blocks->push_back(res_block);
        resnet_blocks_idx.push_back(counter);
        counter += 1;

        if (std::find(attn_scales.begin(), attn_scales.end(), scale) !=
            attn_scales.end()) {
          auto attn_block = QwenImageAttentionBlock(context, out_dim);
          down_blocks->push_back(attn_block);
          attention_blocks_idx.push_back(counter);
          counter += 1;
        }
        in_dim = out_dim;
      }

      if (i != dim_mult.size() - 1) {
        std::string mode =
            temperal_downsample[i] ? "downsample3d" : "downsample2d";
        auto downsample = QwenImageResample(context, out_dim, mode);
        down_blocks->push_back(downsample);
        resample_blocks_idx.push_back(counter);
        counter += 1;
        scale /= 2.0;
      }
    }

    mid_block = register_module(
        "mid_block",
        QwenImageMidBlock(context, dims.back(), dropout, non_linearity, 1));

    norm_out = register_module(
        "norm_out",
        QwenImageRMS_norm(context, dims.back(), true, false, false, false));
    conv_out =
        register_module("conv_out",
                        QwenImageCausalConv3d(context,
                                              dims.back(),
                                              z_dim,
                                              torch::IntArrayRef{3, 3, 3},
                                              torch::IntArrayRef{1, 1, 1},
                                              torch::IntArrayRef{1, 1, 1}));
  }

  torch::Tensor forward(const torch::Tensor& x,
                        std::vector<torch::Tensor>* feat_cache = nullptr,
                        std::vector<int64_t>* feat_idx = {0}) {
    torch::Tensor result_x;

    if (feat_cache && feat_idx) {
      auto idx = (*feat_idx)[0];
      auto cache_x =
          x.index({torch::indexing::Slice(),
                   torch::indexing::Slice(),
                   torch::indexing::Slice(-CACHE_T, torch::indexing::None)})
              .clone();

      if (cache_x.size(2) < 2 && feat_cache->at(idx).defined()) {
        auto last_frame =
            feat_cache->at(idx)
                .index({torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(-1, torch::indexing::None)})
                .unsqueeze(2)
                .to(cache_x.device());
        cache_x = torch::cat({last_frame, cache_x}, 2);
      }
      result_x = conv_in->forward(x, feat_cache->at(idx));
      feat_cache->at(idx) = cache_x;
      (*feat_idx)[0]++;
    } else {
      result_x = conv_in->forward(x);
    }

    int64_t counter = 0;
    for (auto& layer : *down_blocks) {
      if (feat_cache) {
        counter = counter + 1;
        result_x =
            std::dynamic_pointer_cast<QwenImageBaseModule>(layer)->forward(
                result_x, feat_cache, feat_idx);
      } else {
        result_x =
            std::dynamic_pointer_cast<QwenImageBaseModule>(layer)->forward(
                result_x,
                nullptr,
                std::make_unique<std::vector<int64_t>>(std::vector<int64_t>{0})
                    .get());
      }
    }

    result_x = mid_block->forward(result_x, feat_cache, feat_idx);

    result_x = norm_out->forward(result_x);
    result_x = nonlinearity->forward(result_x);

    if (feat_cache && feat_idx) {
      auto idx = (*feat_idx)[0];
      auto cache_x =
          result_x
              .index({torch::indexing::Slice(),
                      torch::indexing::Slice(),
                      torch::indexing::Slice(-CACHE_T, torch::indexing::None)})
              .clone();

      if (cache_x.size(2) < 2 && idx < feat_cache->size() &&
          feat_cache->at(idx).defined()) {
        auto last_frame =
            feat_cache->at(idx)
                .index({torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(-1, torch::indexing::None)})
                .unsqueeze(2)
                .to(cache_x.device());
        cache_x = torch::cat({last_frame, cache_x}, 2);
      }

      result_x = conv_out->forward(result_x, feat_cache->at(idx));
      feat_cache->at(idx) = cache_x;
      (*feat_idx)[0]++;
    } else {
      result_x = conv_out->forward(result_x);
    }

    return result_x;
  }

  void load_state_dict(const StateDict& state_dict) {
    conv_in->load_state_dict(state_dict.get_dict_with_prefix("conv_in."));

    for (size_t resnet_idx : resnet_blocks_idx) {
      down_blocks[resnet_idx]->as<QwenImageResidualBlock>()->load_state_dict(
          state_dict.get_dict_with_prefix("down_blocks." +
                                          std::to_string(resnet_idx) + "."));
    }

    for (size_t attention_idx : attention_blocks_idx) {
      down_blocks[attention_idx]
          ->as<QwenImageAttentionBlock>()
          ->load_state_dict(state_dict.get_dict_with_prefix(
              "down_blocks." + std::to_string(attention_idx) + "."));
    }

    for (size_t resample_idx : resample_blocks_idx) {
      down_blocks[resample_idx]->as<QwenImageResample>()->load_state_dict(
          state_dict.get_dict_with_prefix("down_blocks." +
                                          std::to_string(resample_idx) + "."));
    }

    mid_block->load_state_dict(state_dict.get_dict_with_prefix("mid_block."));
    norm_out->load_state_dict(state_dict.get_dict_with_prefix("norm_out."));
    conv_out->load_state_dict(state_dict.get_dict_with_prefix("conv_out."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    conv_in->verify_loaded_weights("conv_in.");
    for (size_t resnet_idx : resnet_blocks_idx) {
      down_blocks[resnet_idx]
          ->as<QwenImageResidualBlock>()
          ->verify_loaded_weights(std::to_string(resnet_idx) + ".");
    }

    for (size_t attention_idx : attention_blocks_idx) {
      down_blocks[attention_idx]
          ->as<QwenImageAttentionBlock>()
          ->verify_loaded_weights(std::to_string(attention_idx) + ".");
    }

    for (size_t resample_idx : resample_blocks_idx) {
      down_blocks[resample_idx]->as<QwenImageResample>()->verify_loaded_weights(
          std::to_string(resample_idx) + ".");
    }
    mid_block->verify_loaded_weights("mid_block.");
    norm_out->verify_loaded_weights("norm_out.");
    conv_out->verify_loaded_weights("conv_out.");
  }

 private:
  int64_t dim, z_dim;
  std::vector<int64_t> dim_mult;
  std::vector<size_t> resnet_blocks_idx;
  std::vector<size_t> attention_blocks_idx;
  std::vector<size_t> resample_blocks_idx;
  int64_t num_res_blocks;
  std::vector<double> attn_scales;
  std::vector<bool> temperal_downsample;

  torch::nn::SiLU nonlinearity{nullptr};
  QwenImageCausalConv3d conv_in{nullptr};
  torch::nn::ModuleList down_blocks{nullptr};
  QwenImageMidBlock mid_block{nullptr};
  QwenImageRMS_norm norm_out{nullptr};
  QwenImageCausalConv3d conv_out{nullptr};
};

TORCH_MODULE(QwenImageEncoder3d);

class QwenImageUpBlockImpl : public torch::nn::Module {
 public:
  QwenImageUpBlockImpl(const ModelContext& context,
                       int64_t in_dim,
                       int64_t out_dim,
                       int64_t num_res_blocks,
                       double dropout = 0.0,
                       const std::string& upsample_mode = "",
                       const std::string& non_linearity = "silu")
      : in_dim(in_dim), out_dim(out_dim) {
    resnets = register_module("resnets", torch::nn::ModuleList());
    int64_t current_dim = in_dim;

    for (int64_t i = 0; i < num_res_blocks + 1; i++) {
      auto resnet = QwenImageResidualBlock(
          context, current_dim, out_dim, dropout, non_linearity);
      resnets->push_back(resnet);
      current_dim = out_dim;
    }

    if (!upsample_mode.empty()) {
      upsamplers = register_module("upsamplers", torch::nn::ModuleList());
      auto upsample = QwenImageResample(context, out_dim, upsample_mode);
      upsamplers->push_back(upsample);
    }
  }

  torch::Tensor forward(const torch::Tensor& x,
                        std::vector<torch::Tensor>* feat_cache = nullptr,
                        std::vector<int64_t>* feat_idx = {0}) {
    auto result_x = x;

    for (auto& resnet : *resnets) {
      if (feat_cache && feat_idx) {
        result_x =
            std::dynamic_pointer_cast<QwenImageBaseModule>(resnet)->forward(
                result_x, feat_cache, feat_idx);
      } else {
        result_x =
            std::dynamic_pointer_cast<QwenImageBaseModule>(resnet)->forward(
                result_x,
                nullptr,
                std::make_unique<std::vector<int64_t>>(std::vector<int64_t>{0})
                    .get());
      }
    }

    if (upsamplers) {
      if (feat_cache && feat_idx) {
        result_x = std::dynamic_pointer_cast<QwenImageBaseModule>(upsamplers[0])
                       ->forward(result_x, feat_cache, feat_idx);
      } else {
        result_x = std::dynamic_pointer_cast<QwenImageBaseModule>(upsamplers[0])
                       ->forward(result_x,
                                 nullptr,
                                 std::make_unique<std::vector<int64_t>>(
                                     std::vector<int64_t>{0})
                                     .get());
      }
    }

    return result_x;
  }

  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < resnets->size(); i++) {
      auto prefix = "resnets." + std::to_string(i) + ".";
      resnets[i]->as<QwenImageResidualBlock>()->load_state_dict(
          state_dict.get_dict_with_prefix(prefix));
    }

    if (upsamplers) {
      upsamplers[0]->as<QwenImageResample>()->load_state_dict(
          state_dict.get_dict_with_prefix("upsamplers.0."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) {
    for (size_t i = 0; i < resnets->size(); i++) {
      auto prefix = "resnets." + std::to_string(i) + ".";
      resnets[i]->as<QwenImageResidualBlock>()->verify_loaded_weights(prefix);
    }

    if (upsamplers) {
      upsamplers[0]->as<QwenImageResample>()->verify_loaded_weights(
          "upsamplers.0.");
    }
  }

 private:
  int64_t in_dim, out_dim;
  torch::nn::ModuleList resnets{nullptr};
  torch::nn::ModuleList upsamplers{nullptr};
};

TORCH_MODULE(QwenImageUpBlock);

class QwenImageDecoder3dImpl : public torch::nn::Module {
 public:
  QwenImageDecoder3dImpl(const ModelContext& context,
                         int64_t dim = 128,
                         int64_t z_dim = 4,
                         std::vector<int64_t> dim_mult = {1, 2, 4, 4},
                         int64_t num_res_blocks = 2,
                         std::vector<double> attn_scales = {},
                         std::vector<bool> temperal_upsample = {false,
                                                                true,
                                                                true},
                         double dropout = 0.0,
                         const std::string& non_linearity = "silu")
      : dim(dim),
        z_dim(z_dim),
        dim_mult(dim_mult),
        num_res_blocks(num_res_blocks),
        attn_scales(attn_scales),
        temperal_upsample(temperal_upsample) {
    nonlinearity = register_module("silu", torch::nn::SiLU());

    std::vector<int64_t> dims = {dim * dim_mult.back()};
    for (int64_t i = dim_mult.size() - 1; i >= 0; i--) {
      dims.push_back(dim * dim_mult.at(i));
    }

    double scale = 1.0 / std::pow(2, dim_mult.size() - 2);

    conv_in =
        register_module("conv_in",
                        QwenImageCausalConv3d(context,
                                              z_dim,
                                              dims[0],
                                              torch::IntArrayRef{3, 3, 3},
                                              torch::IntArrayRef{1, 1, 1},
                                              torch::IntArrayRef{1, 1, 1}));

    mid_block = register_module(
        "mid_block",
        QwenImageMidBlock(context, dims[0], dropout, non_linearity, 1));

    up_blocks = register_module("up_blocks", torch::nn::ModuleList());
    for (size_t i = 0; i < dims.size() - 1; i++) {
      int64_t in_dim = dims[i];
      int64_t out_dim = dims[i + 1];

      if (i > 0) {
        in_dim = in_dim / 2;
      }

      std::string upsample_mode;
      if (i != dim_mult.size() - 1) {
        upsample_mode = temperal_upsample[i] ? "upsample3d" : "upsample2d";
      }

      auto up_block = QwenImageUpBlock(context,
                                       in_dim,
                                       out_dim,
                                       num_res_blocks,
                                       dropout,
                                       upsample_mode,
                                       non_linearity);
      up_blocks->push_back(up_block);

      if (!upsample_mode.empty()) {
        scale *= 2.0;
      }
    }

    norm_out = register_module(
        "norm_out",
        QwenImageRMS_norm(context, dims.back(), true, false, false, false));
    conv_out =
        register_module("conv_out",
                        QwenImageCausalConv3d(context,
                                              dims.back(),
                                              3,
                                              torch::IntArrayRef{3, 3, 3},
                                              torch::IntArrayRef{1, 1, 1},
                                              torch::IntArrayRef{1, 1, 1}));
  }

  torch::Tensor forward(const torch::Tensor& x,
                        std::vector<torch::Tensor>* feat_cache = nullptr,
                        std::vector<int64_t>* feat_idx = {0}) {
    auto result_x = x;

    if (feat_cache) {
      auto idx = (*feat_idx)[0];
      auto cache_x =
          result_x
              .index({torch::indexing::Slice(),
                      torch::indexing::Slice(),
                      torch::indexing::Slice(-CACHE_T, torch::indexing::None)})
              .clone();

      if (cache_x.size(2) < 2 && feat_cache->at(idx).defined()) {
        auto last_frame =
            feat_cache->at(idx)
                .index({torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(-1, torch::indexing::None)})
                .unsqueeze(2)
                .to(cache_x.device());
        cache_x = torch::cat({last_frame, cache_x}, 2);
      }

      result_x = conv_in->forward(result_x, feat_cache->at(idx));
      feat_cache->at(idx) = cache_x;
      (*feat_idx)[0]++;
    } else {
      result_x = conv_in->forward(result_x);
    }

    result_x = mid_block->forward(result_x, feat_cache, feat_idx);

    for (auto& up_block : *up_blocks) {
      result_x = up_block->as<QwenImageUpBlock>()->forward(
          result_x, feat_cache, feat_idx);
    }

    result_x = norm_out->forward(result_x);
    result_x = nonlinearity->forward(result_x);

    if (feat_cache) {
      auto idx = (*feat_idx)[0];
      auto cache_x =
          result_x
              .index({torch::indexing::Slice(),
                      torch::indexing::Slice(),
                      torch::indexing::Slice(-CACHE_T, torch::indexing::None)})
              .clone();

      if (cache_x.size(2) < 2 && idx < feat_cache->size() &&
          feat_cache->at(idx).defined()) {
        auto last_frame =
            feat_cache->at(idx)
                .index({torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(-1, torch::indexing::None)})
                .unsqueeze(2)
                .to(cache_x.device());
        cache_x = torch::cat({last_frame, cache_x}, 2);
      }

      result_x = conv_out->forward(result_x, feat_cache->at(idx));
      feat_cache->at(idx) = cache_x;
      (*feat_idx)[0]++;
    } else {
      result_x = conv_out->forward(result_x);
    }
    return result_x;
  }

  void load_state_dict(const StateDict& state_dict) {
    conv_in->load_state_dict(state_dict.get_dict_with_prefix("conv_in."));
    mid_block->load_state_dict(state_dict.get_dict_with_prefix("mid_block."));

    for (size_t i = 0; i < up_blocks->size(); i++) {
      auto prefix = "up_blocks." + std::to_string(i) + ".";
      up_blocks[i]->as<QwenImageUpBlock>()->load_state_dict(
          state_dict.get_dict_with_prefix(prefix));
    }

    norm_out->load_state_dict(state_dict.get_dict_with_prefix("norm_out."));
    conv_out->load_state_dict(state_dict.get_dict_with_prefix("conv_out."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    conv_in->verify_loaded_weights("conv_in.");

    mid_block->verify_loaded_weights("mid_block.");
    for (size_t i = 0; i < up_blocks->size(); i++) {
      auto prefix = "up_blocks." + std::to_string(i) + ".";
      up_blocks[i]->as<QwenImageUpBlock>()->verify_loaded_weights(prefix);
    }

    norm_out->verify_loaded_weights("norm_out.");
    conv_out->verify_loaded_weights("conv_out.");
  }

  std::vector<std::shared_ptr<Module>> get_modules() const {
    std::vector<std::shared_ptr<Module>> module = modules();
    return module;
  }

 private:
  int64_t dim, z_dim;
  std::vector<int64_t> dim_mult;
  int64_t num_res_blocks;
  std::vector<double> attn_scales;
  std::vector<bool> temperal_upsample;

  torch::nn::SiLU nonlinearity{nullptr};
  QwenImageCausalConv3d conv_in{nullptr};
  QwenImageMidBlock mid_block{nullptr};
  torch::nn::ModuleList up_blocks{nullptr};
  QwenImageRMS_norm norm_out{nullptr};
  QwenImageCausalConv3d conv_out{nullptr};
};

TORCH_MODULE(QwenImageDecoder3d);

class DiagonalGaussianDistribution {
 public:
  DiagonalGaussianDistribution(torch::Tensor parameters,
                               bool deterministic = false)
      : parameters_(std::move(parameters)), deterministic_(deterministic) {
    auto chunks = parameters_.chunk(2, 1);
    mean_ = chunks[0];
    logvar_ = chunks[1];

    logvar_ = torch::clamp(logvar_, -30.0f, 20.0f);

    std_ = torch::exp(0.5f * logvar_);
    var_ = torch::exp(logvar_);

    if (deterministic_) {
      std_.fill_(0.0f);
      var_.fill_(0.0f);
    }
  }

  torch::Tensor sample(int64_t seed) const {
    torch::TensorOptions options = mean_.options();
    std::vector<int64_t> shape(mean_.sizes().begin(), mean_.sizes().end());
    return mean_ + std_ * randn_tensor(shape, seed, options);
  }

  torch::Tensor kl(const std::optional<DiagonalGaussianDistribution>& other =
                       std::nullopt) const {
    if (deterministic_) {
      return torch::tensor(0.0f, mean_.options());
    }

    if (!other.has_value()) {
      return 0.5f * torch::sum(torch::pow(mean_, 2) + var_ - 1.0f - logvar_,
                               {1, 2, 3});
    } else {
      const auto& other_dist = other.value();
      return 0.5f * torch::sum(torch::pow(mean_ - other_dist.mean_, 2) /
                                       other_dist.var_ +
                                   var_ / other_dist.var_ - 1.0f - logvar_ +
                                   other_dist.logvar_,
                               {1, 2, 3});
    }
  }

  torch::Tensor nll(const torch::Tensor& sample,
                    const std::vector<int64_t>& dims = {1, 2, 3}) const {
    if (deterministic_) {
      return torch::tensor(0.0f, mean_.options());
    }
    const float logtwopi = std::log(2.0f * M_PI);
    return 0.5f *
           torch::sum(logtwopi + logvar_ + torch::pow(sample - mean_, 2) / var_,
                      dims);
  }

  torch::Tensor mode() const { return mean_; }

  const torch::Tensor& mean() const { return mean_; }
  const torch::Tensor& std() const { return std_; }
  const torch::Tensor& var() const { return var_; }
  const torch::Tensor& logvar() const { return logvar_; }

 private:
  torch::Tensor parameters_;
  torch::Tensor mean_;
  torch::Tensor logvar_;
  torch::Tensor std_;
  torch::Tensor var_;
  bool deterministic_;
};

struct AutoencoderKLOutput {
  DiagonalGaussianDistribution latent_dist;
  AutoencoderKLOutput(DiagonalGaussianDistribution dist)
      : latent_dist(std::move(dist)) {}
};

struct DecoderOutput {
  torch::Tensor sample;
  DecoderOutput(torch::Tensor sample) : sample(std::move(sample)) {}
};

class AutoencoderKLQwenImageImpl : public torch::nn::Module {
 public:
  AutoencoderKLQwenImageImpl(const ModelContext& context)
      : args_(context.get_model_args()),
        z_dim(context.get_model_args().z_dim()),
        temperal_downsample(context.get_model_args().temperal_downsample()),
        base_dim(context.get_model_args().base_dim()),
        dim_mult(context.get_model_args().dim_mult()),
        num_res_blocks(context.get_model_args().num_res_blocks()),
        attn_scales(context.get_model_args().attn_scales()),
        dropout(context.get_model_args().dropout()) {
    temperal_upsample = std::vector<bool>(temperal_downsample.rbegin(),
                                          temperal_downsample.rend());

    encoder = register_module("encoder",
                              QwenImageEncoder3d(context,
                                                 base_dim,
                                                 z_dim * 2,
                                                 dim_mult,
                                                 num_res_blocks,
                                                 attn_scales,
                                                 temperal_downsample,
                                                 dropout));

    quant_conv =
        register_module("quant_conv",
                        QwenImageCausalConv3d(context,
                                              z_dim * 2,
                                              z_dim * 2,
                                              torch::IntArrayRef{1, 1, 1},
                                              torch::IntArrayRef{1, 1, 1},
                                              torch::IntArrayRef{0, 0, 0}));

    post_quant_conv =
        register_module("post_quant_conv",
                        QwenImageCausalConv3d(context,
                                              z_dim,
                                              z_dim,
                                              torch::IntArrayRef{1, 1, 1},
                                              torch::IntArrayRef{1, 1, 1},
                                              torch::IntArrayRef{0, 0, 0}));

    decoder = register_module("decoder",
                              QwenImageDecoder3d(context,
                                                 base_dim,
                                                 z_dim,
                                                 dim_mult,
                                                 num_res_blocks,
                                                 attn_scales,
                                                 temperal_upsample,
                                                 dropout));

    spatial_compression_ratio =
        static_cast<int64_t>(std::pow(2, temperal_downsample.size()));

    use_slicing = false;
    use_tiling = false;
    tile_sample_min_height = 256;
    tile_sample_min_width = 256;
    tile_sample_stride_height = 192;
    tile_sample_stride_width = 192;

    cached_conv_counts = {{"decoder", count_conv3d_modules(*decoder)},
                          {"encoder", count_conv3d_modules(*encoder)}};
  }

  void enable_tiling(int64_t tile_sample_min_height = -1,
                     int64_t tile_sample_min_width = -1,
                     int64_t tile_sample_stride_height = -1,
                     int64_t tile_sample_stride_width = -1) {
    use_tiling = true;
    if (tile_sample_min_height > 0)
      this->tile_sample_min_height = tile_sample_min_height;
    if (tile_sample_min_width > 0)
      this->tile_sample_min_width = tile_sample_min_width;
    if (tile_sample_stride_height > 0)
      this->tile_sample_stride_height = tile_sample_stride_height;
    if (tile_sample_stride_width > 0)
      this->tile_sample_stride_width = tile_sample_stride_width;
  }

  void clear_cache() {
    conv_num = count_conv3d_modules(*decoder);
    conv_idx = {0};

    feat_map = std::vector<torch::Tensor>(conv_num);

    enc_conv_num = count_conv3d_modules(*encoder);
    enc_conv_idx = {0};
    enc_feat_map = std::vector<torch::Tensor>(enc_conv_num);
  }

  torch::Tensor _encode(const torch::Tensor& x) {
    auto sizes = x.sizes();
    auto b = sizes[0], c = sizes[1], num_frame = sizes[2], height = sizes[3],
         width = sizes[4];

    if (use_tiling &&
        (width > tile_sample_min_width || height > tile_sample_min_height)) {
      return tiled_encode(x);
    }

    clear_cache();
    auto iter = 1 + (num_frame - 1) / 4;
    torch::Tensor out;

    for (int64_t i = 0; i < iter; i++) {
      enc_conv_idx[0] = 0;
      torch::Tensor tile;

      if (i == 0) {
        tile = x.index({torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(0, 1),
                        torch::indexing::Slice(),
                        torch::indexing::Slice()});
      } else {
        tile = x.index({torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(1 + 4 * (i - 1), 1 + 4 * i),
                        torch::indexing::Slice(),
                        torch::indexing::Slice()});
      }

      auto encoded_tile = encoder->forward(tile, &enc_feat_map, &enc_conv_idx);

      if (i == 0) {
        out = encoded_tile;
      } else {
        out = torch::cat({out, encoded_tile}, 2);
      }
    }

    auto enc = quant_conv->forward(out);
    clear_cache();
    return enc;
  }

  AutoencoderKLOutput encode(const torch::Tensor& x, bool return_dict = true) {
    torch::Tensor h;

    if (use_slicing && x.size(0) > 1) {
      std::vector<torch::Tensor> encoded_slices;
      auto slices = x.split(1);
      for (auto& slice : slices) {
        encoded_slices.push_back(_encode(slice));
      }
      h = torch::cat(encoded_slices);
    } else {
      h = _encode(x);
    }

    auto posterior = DiagonalGaussianDistribution(h);

    if (!return_dict) {
      return {posterior};
    }

    AutoencoderKLOutput output(posterior);
    return output;
  }

  DecoderOutput _decode(const torch::Tensor& z, bool return_dict = true) {
    auto sizes = z.sizes();
    auto b = sizes[0], c = sizes[1], num_frame = sizes[2], height = sizes[3],
         width = sizes[4];

    auto tile_latent_min_height =
        tile_sample_min_height / spatial_compression_ratio;
    auto tile_latent_min_width =
        tile_sample_min_width / spatial_compression_ratio;

    if (use_tiling &&
        (width > tile_latent_min_width || height > tile_latent_min_height)) {
      return tiled_decode(z, return_dict);
    }

    clear_cache();
    auto x = post_quant_conv->forward(z);
    torch::Tensor out;

    for (int64_t i = 0; i < num_frame; i++) {
      conv_idx[0] = 0;
      auto frame = x.index({torch::indexing::Slice(),
                            torch::indexing::Slice(),
                            torch::indexing::Slice(i, i + 1),
                            torch::indexing::Slice(),
                            torch::indexing::Slice()});

      auto decoded_frame = decoder->forward(frame, &feat_map, &conv_idx);

      if (i == 0) {
        out = decoded_frame;
      } else {
        out = torch::cat({out, decoded_frame}, 2);
      }
    }

    out = torch::clamp(out, -1.0, 1.0);
    clear_cache();

    if (!return_dict) {
      return {out};
    }
    DecoderOutput output(out);

    return output;
  }

  DecoderOutput decode(const torch::Tensor& z, bool return_dict = true) {
    torch::Tensor decoded;

    if (use_slicing && z.size(0) > 1) {
      std::vector<torch::Tensor> decoded_slices;
      auto slices = z.split(1);
      for (auto& slice : slices) {
        auto output = _decode(slice, true);
        decoded_slices.push_back(output.sample);
      }
      decoded = torch::cat(decoded_slices);
    } else {
      auto output = _decode(z, true);
      decoded = output.sample;
    }

    if (!return_dict) {
      return {decoded};
    }
    DecoderOutput output(decoded);

    return output;
  }

  torch::Tensor blend_v(const torch::Tensor& a,
                        const torch::Tensor& b,
                        int64_t blend_extent) {
    auto result_b = b.clone();
    blend_extent = std::min({a.size(3), b.size(3), blend_extent});

    for (int64_t y = 0; y < blend_extent; y++) {
      auto weight_a = 1.0 - static_cast<double>(y) / blend_extent;
      auto weight_b = static_cast<double>(y) / blend_extent;

      auto a_slice = a.index(
          {torch::indexing::Slice(),
           torch::indexing::Slice(),
           torch::indexing::Slice(),
           torch::indexing::Slice(-blend_extent + y, -blend_extent + y + 1),
           torch::indexing::Slice()});

      auto b_slice = result_b.index({torch::indexing::Slice(),
                                     torch::indexing::Slice(),
                                     torch::indexing::Slice(),
                                     torch::indexing::Slice(y, y + 1),
                                     torch::indexing::Slice()});

      auto blended = a_slice * weight_a + b_slice * weight_b;
      result_b.index_put_({torch::indexing::Slice(),
                           torch::indexing::Slice(),
                           torch::indexing::Slice(),
                           torch::indexing::Slice(y, y + 1),
                           torch::indexing::Slice()},
                          blended);
    }

    return result_b;
  }

  torch::Tensor blend_h(const torch::Tensor& a,
                        const torch::Tensor& b,
                        int64_t blend_extent) {
    auto result_b = b.clone();
    blend_extent = std::min({a.size(4), b.size(4), blend_extent});

    for (int64_t x = 0; x < blend_extent; x++) {
      auto weight_a = 1.0 - static_cast<double>(x) / blend_extent;
      auto weight_b = static_cast<double>(x) / blend_extent;

      auto a_slice = a.index(
          {torch::indexing::Slice(),
           torch::indexing::Slice(),
           torch::indexing::Slice(),
           torch::indexing::Slice(),
           torch::indexing::Slice(-blend_extent + x, -blend_extent + x + 1)});

      auto b_slice = result_b.index({torch::indexing::Slice(),
                                     torch::indexing::Slice(),
                                     torch::indexing::Slice(),
                                     torch::indexing::Slice(),
                                     torch::indexing::Slice(x, x + 1)});

      auto blended = a_slice * weight_a + b_slice * weight_b;
      result_b.index_put_({torch::indexing::Slice(),
                           torch::indexing::Slice(),
                           torch::indexing::Slice(),
                           torch::indexing::Slice(),
                           torch::indexing::Slice(x, x + 1)},
                          blended);
    }

    return result_b;
  }

  torch::Tensor tiled_encode(const torch::Tensor& x) {
    auto sizes = x.sizes();
    auto b = sizes[0], c = sizes[1], num_frames = sizes[2], height = sizes[3],
         width = sizes[4];

    auto latent_height = height / spatial_compression_ratio;
    auto latent_width = width / spatial_compression_ratio;

    auto tile_latent_min_height =
        tile_sample_min_height / spatial_compression_ratio;
    auto tile_latent_min_width =
        tile_sample_min_width / spatial_compression_ratio;
    auto tile_latent_stride_height =
        tile_sample_stride_height / spatial_compression_ratio;
    auto tile_latent_stride_width =
        tile_sample_stride_width / spatial_compression_ratio;

    auto blend_height = tile_latent_min_height - tile_latent_stride_height;
    auto blend_width = tile_latent_min_width - tile_latent_stride_width;

    std::vector<std::vector<torch::Tensor>> rows;

    for (int64_t i = 0; i < height; i += tile_sample_stride_height) {
      std::vector<torch::Tensor> row;

      for (int64_t j = 0; j < width; j += tile_sample_stride_width) {
        clear_cache();
        std::vector<torch::Tensor> time_frames;
        auto frame_range = 1 + (num_frames - 1) / 4;

        for (int64_t k = 0; k < frame_range; k++) {
          enc_conv_idx[0] = 0;
          torch::Tensor tile;

          if (k == 0) {
            tile =
                x.index({torch::indexing::Slice(),
                         torch::indexing::Slice(),
                         torch::indexing::Slice(0, 1),
                         torch::indexing::Slice(i, i + tile_sample_min_height),
                         torch::indexing::Slice(j, j + tile_sample_min_width)});
          } else {
            tile =
                x.index({torch::indexing::Slice(),
                         torch::indexing::Slice(),
                         torch::indexing::Slice(1 + 4 * (k - 1), 1 + 4 * k),
                         torch::indexing::Slice(i, i + tile_sample_min_height),
                         torch::indexing::Slice(j, j + tile_sample_min_width)});
          }

          auto encoded_tile =
              encoder->forward(tile, &enc_feat_map, &enc_conv_idx);
          auto quantized_tile = quant_conv->forward(encoded_tile);
          time_frames.push_back(quantized_tile);
        }

        row.push_back(torch::cat(time_frames, 2));
      }
      rows.push_back(row);
    }
    clear_cache();

    std::vector<torch::Tensor> result_rows;

    for (int64_t i = 0; i < static_cast<int64_t>(rows.size()); i++) {
      std::vector<torch::Tensor> result_row;

      for (int64_t j = 0; j < static_cast<int64_t>(rows[i].size()); j++) {
        auto tile = rows[i][j];

        if (i > 0) {
          tile = blend_v(rows[i - 1][j], tile, blend_height);
        }
        if (j > 0) {
          tile = blend_h(rows[i][j - 1], tile, blend_width);
        }

        result_row.push_back(
            tile.index({torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(0, tile_latent_stride_height),
                        torch::indexing::Slice(0, tile_latent_stride_width)}));
      }

      result_rows.push_back(torch::cat(result_row, -1));
    }

    auto enc = torch::cat(result_rows, 3)
                   .index({torch::indexing::Slice(),
                           torch::indexing::Slice(),
                           torch::indexing::Slice(),
                           torch::indexing::Slice(0, latent_height),
                           torch::indexing::Slice(0, latent_width)});

    return enc;
  }

  DecoderOutput tiled_decode(const torch::Tensor& z, bool return_dict = true) {
    auto sizes = z.sizes();
    auto b = sizes[0], c = sizes[1], num_frames = sizes[2], height = sizes[3],
         width = sizes[4];

    auto sample_height = height * spatial_compression_ratio;
    auto sample_width = width * spatial_compression_ratio;

    auto tile_latent_min_height =
        tile_sample_min_height / spatial_compression_ratio;
    auto tile_latent_min_width =
        tile_sample_min_width / spatial_compression_ratio;
    auto tile_latent_stride_height =
        tile_sample_stride_height / spatial_compression_ratio;
    auto tile_latent_stride_width =
        tile_sample_stride_width / spatial_compression_ratio;

    auto blend_height = tile_sample_min_height - tile_sample_stride_height;
    auto blend_width = tile_sample_min_width - tile_sample_stride_width;

    std::vector<std::vector<torch::Tensor>> rows;

    for (int64_t i = 0; i < height; i += tile_latent_stride_height) {
      std::vector<torch::Tensor> row;

      for (int64_t j = 0; j < width; j += tile_latent_stride_width) {
        clear_cache();
        std::vector<torch::Tensor> time_frames;

        for (int64_t k = 0; k < num_frames; k++) {
          conv_idx[0] = 0;
          auto tile =
              z.index({torch::indexing::Slice(),
                       torch::indexing::Slice(),
                       torch::indexing::Slice(k, k + 1),
                       torch::indexing::Slice(i, i + tile_latent_min_height),
                       torch::indexing::Slice(j, j + tile_latent_min_width)});

          auto post_quant_tile = post_quant_conv->forward(tile);
          auto decoded_tile =
              decoder->forward(post_quant_tile, &feat_map, &conv_idx);
          time_frames.push_back(decoded_tile);
        }

        row.push_back(torch::cat(time_frames, 2));
      }
      rows.push_back(row);
    }
    clear_cache();

    std::vector<torch::Tensor> result_rows;

    for (int64_t i = 0; i < static_cast<int64_t>(rows.size()); i++) {
      std::vector<torch::Tensor> result_row;

      for (int64_t j = 0; j < static_cast<int64_t>(rows[i].size()); j++) {
        auto tile = rows[i][j];

        if (i > 0) {
          tile = blend_v(rows[i - 1][j], tile, blend_height);
        }
        if (j > 0) {
          tile = blend_h(rows[i][j - 1], tile, blend_width);
        }

        result_row.push_back(
            tile.index({torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(0, tile_sample_stride_height),
                        torch::indexing::Slice(0, tile_sample_stride_width)}));
      }

      result_rows.push_back(torch::cat(result_row, -1));
    }

    auto dec = torch::cat(result_rows, 3)
                   .index({torch::indexing::Slice(),
                           torch::indexing::Slice(),
                           torch::indexing::Slice(),
                           torch::indexing::Slice(0, sample_height),
                           torch::indexing::Slice(0, sample_width)});

    if (!return_dict) {
      return {dec};
    }
    DecoderOutput output(dec);
    return output;
  }

  DecoderOutput forward(const torch::Tensor& sample,
                        bool sample_posterior = false,
                        bool return_dict = true,
                        int64_t seed = 42) {
    auto x = sample;

    auto encode_output = encode(x, true);
    auto posterior = encode_output.latent_dist;

    torch::Tensor z;
    if (sample_posterior) {
      z = posterior.sample(seed);
    } else {
      z = posterior.mode();
    }

    auto dec = decode(z, return_dict);
    return dec;
  }

  void load_state_dict(const StateDict& state_dict) {
    encoder->load_state_dict(state_dict.get_dict_with_prefix("encoder."));
    decoder->load_state_dict(state_dict.get_dict_with_prefix("decoder."));
    quant_conv->load_state_dict(state_dict.get_dict_with_prefix("quant_conv."));
    post_quant_conv->load_state_dict(
        state_dict.get_dict_with_prefix("post_quant_conv."));
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      encoder->load_state_dict(state_dict->get_dict_with_prefix("encoder."));
      decoder->load_state_dict(state_dict->get_dict_with_prefix("decoder."));
      quant_conv->load_state_dict(
          state_dict->get_dict_with_prefix("quant_conv."));
      post_quant_conv->load_state_dict(
          state_dict->get_dict_with_prefix("post_quant_conv."));
    }
    verify_loaded_weights("");
  }

  void verify_loaded_weights(const std::string& prefix) {
    encoder->verify_loaded_weights("encoder.");
    decoder->verify_loaded_weights("decoder.");
    quant_conv->verify_loaded_weights("quant_conv.");
    post_quant_conv->verify_loaded_weights("post_quant_conv.");
  }

 private:
  template <typename ModuleType>
  int64_t count_conv3d_modules(const ModuleType& module) {
    int64_t count = 0;
    for (const auto& m : module.named_modules()) {
      if (auto conv =
              dynamic_cast<QwenImageCausalConv3dImpl*>(m.value().get())) {
        count++;
      }
    }
    return count;
  }

  int64_t base_dim, z_dim;
  std::vector<int64_t> dim_mult;
  int64_t num_res_blocks;
  std::vector<double> attn_scales;
  std::vector<bool> temperal_downsample, temperal_upsample;
  double dropout;

  int64_t spatial_compression_ratio;
  bool use_slicing, use_tiling;
  int64_t tile_sample_min_height, tile_sample_min_width;
  int64_t tile_sample_stride_height, tile_sample_stride_width;

  std::unordered_map<std::string, int64_t> cached_conv_counts;

  int64_t conv_num, enc_conv_num;
  std::vector<int64_t> conv_idx, enc_conv_idx;
  std::vector<torch::Tensor> feat_map, enc_feat_map;

  QwenImageEncoder3d encoder{nullptr};
  QwenImageCausalConv3d quant_conv{nullptr};
  QwenImageCausalConv3d post_quant_conv{nullptr};
  QwenImageDecoder3d decoder{nullptr};

  ModelArgs args_;
};

TORCH_MODULE(AutoencoderKLQwenImage);

REGISTER_MODEL_ARGS(AutoencoderKLQwenImage, [&] {
  LOAD_ARG_OR(base_dim, "base_dim", 96);
  LOAD_ARG_OR(z_dim, "z_dim", 16);
  LOAD_ARG_OR(dim_mult, "dim_mult", (std::vector<int64_t>{1, 2, 4, 4}));
  LOAD_ARG_OR(attn_scales, "attn_scales", (std::vector<double>{}));
  LOAD_ARG_OR(temperal_downsample,
              "temperal_downsample",
              (std::vector<bool>{false, true, true}));
  LOAD_ARG_OR(num_res_blocks, "num_res_blocks", 2);
  LOAD_ARG_OR(dropout, "dropout", 0);
  LOAD_ARG_OR(latents_mean,
              "latents_mean",
              (std::vector<double>{-0.7571,
                                   -0.7089,
                                   -0.9113,
                                   0.1075,
                                   -0.1745,
                                   0.9653,
                                   -0.1517,
                                   1.5508,
                                   0.4134,
                                   -0.0715,
                                   0.5517,
                                   -0.3632,
                                   -0.1922,
                                   -0.9497,
                                   0.2503,
                                   -0.2921}));
  LOAD_ARG_OR(latents_std,
              "latents_std",
              (std::vector<double>{2.8184,
                                   1.4541,
                                   2.3275,
                                   2.6558,
                                   1.2196,
                                   1.7708,
                                   2.6052,
                                   2.0743,
                                   3.2687,
                                   2.1526,
                                   2.8652,
                                   1.5579,
                                   1.6382,
                                   1.1253,
                                   2.8251,
                                   1.916}));
});

}  // namespace qwenimage
}  // namespace xllm
