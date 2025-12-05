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

#include <torch/torch.h>

#include "core/framework/state_dict/utils.h"
namespace xllm {
namespace F = torch::nn::functional;

class DiTLinearImpl : public torch::nn::Module {
 public:
  DiTLinearImpl(int64_t in, int64_t out, bool with_bias = true) {
    if (with_bias) {
      // the weight needs to be transposed when using addmm
      weight = register_parameter("weight", torch::empty({in, out}));
      bias = register_parameter("bias", torch::empty(out));
    } else {
      weight = register_parameter("weight", torch::empty({out, in}));
      bias = register_parameter("bias", {}, false);
    }
  }

  torch::Tensor forward(const torch::Tensor& x) {
    // use addmm when bias is provided
    if (bias.defined()) {
      auto sizes = x.sizes();
      if (sizes.size() == 3) {
        torch::Tensor x_;
        x_ = x.reshape({sizes[0] * sizes[1], sizes[2]});
        return torch::addmm(bias, x_, weight, 1, 1)
            .reshape({sizes[0], sizes[1], weight.size(1)});
      } else {
        return torch::addmm(bias, x, weight, 1, 1);
      }
    } else {
      return F::linear(x, weight, bias);
    }
  }

  void load_state_dict(const StateDict& state_dict) {
    // an overloaded load_weight function was used to load transpoesd weights
    weight::load_weight(
        state_dict, "weight", weight, weight_is_loaded_, bias.defined());
    if (bias.defined()) {
      weight::load_weight(state_dict, "bias", bias, bias_is_loaded_);
    }
  }

  void to(torch::TensorOptions options) {
    weight.set_data(weight.to(options));
    if (bias.defined()) {
      bias.set_data(bias.to(options));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(weight_is_loaded_)
        << "weight is not loaded for " << prefix + "weight";
    if (bias.defined()) {
      CHECK(bias_is_loaded_) << "bias is not loaded for " << prefix + "bias";
    }
  }

  torch::Tensor weight;
  torch::Tensor bias;

 private:
  bool weight_is_loaded_{false};
  bool bias_is_loaded_{false};
  bool transposed_{false};
};

TORCH_MODULE(DiTLinear);
}  // namespace xllm
