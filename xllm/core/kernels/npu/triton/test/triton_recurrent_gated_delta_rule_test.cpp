/* Copyright 2025 The xLLM Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/jd-opensource/xllm/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 */

#include <acl/acl.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include "kernel_loader.h"
#include "test_utils.h"
#include "torch_api/npu_triton_recurrent_gated_delta_rule.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

namespace xllm::kernel::npu {

constexpr float kTolerance = 1e-3f;
constexpr float kStateTolerance = 1e-2f;
constexpr int32_t kDeviceId = 0;

torch::Tensor l2norm(const torch::Tensor& x,
                     int64_t dim = -1,
                     float eps = 1e-6f) {
  auto inv_norm = torch::rsqrt((x * x).sum(dim, /*keepdim=*/true) + eps);
  return x * inv_norm;
}

std::pair<torch::Tensor, torch::Tensor> torch_recurrent_gated_delta_rule(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& g,
    const torch::Tensor& beta,
    const c10::optional<torch::Tensor>& initial_state,
    bool output_final_state,
    bool use_qk_l2norm_in_kernel = false) {
  auto initial_dtype = query.dtype();

  torch::Tensor q, k, v, b, g_float;
  if (use_qk_l2norm_in_kernel) {
    q = l2norm(query, -1, 1e-6f)
            .transpose(1, 2)
            .contiguous()
            .to(torch::kFloat32);
    k = l2norm(key, -1, 1e-6f).transpose(1, 2).contiguous().to(torch::kFloat32);
  } else {
    q = query.transpose(1, 2).contiguous().to(torch::kFloat32);
    k = key.transpose(1, 2).contiguous().to(torch::kFloat32);
  }
  v = value.transpose(1, 2).contiguous().to(torch::kFloat32);
  b = beta.transpose(1, 2).contiguous().to(torch::kFloat32);
  g_float = g.transpose(1, 2).contiguous().to(torch::kFloat32);

  int64_t batch_size = k.size(0);
  int64_t num_heads = k.size(1);
  int64_t sequence_length = k.size(2);
  int64_t k_head_dim = k.size(3);
  int64_t v_head_dim = v.size(3);

  float scale = 1.0f / std::sqrt(static_cast<float>(q.size(3)));
  q = q * scale;

  auto core_attn_out = torch::zeros(
      {batch_size, num_heads, sequence_length, v_head_dim},
      torch::TensorOptions().dtype(torch::kFloat32).device(value.device()));

  torch::Tensor last_recurrent_state;
  if (initial_state.has_value()) {
    last_recurrent_state = initial_state.value().to(torch::kFloat32).clone();
  } else {
    last_recurrent_state = torch::zeros(
        {batch_size, num_heads, k_head_dim, v_head_dim},
        torch::TensorOptions().dtype(torch::kFloat32).device(value.device()));
  }

  for (int64_t i = 0; i < sequence_length; ++i) {
    auto q_t = q.index({torch::indexing::Slice(), torch::indexing::Slice(), i});
    auto k_t = k.index({torch::indexing::Slice(), torch::indexing::Slice(), i});
    auto v_t = v.index({torch::indexing::Slice(), torch::indexing::Slice(), i});
    auto g_t =
        g_float.index({torch::indexing::Slice(), torch::indexing::Slice(), i})
            .exp()
            .unsqueeze(-1)
            .unsqueeze(-1);
    auto beta_t =
        b.index({torch::indexing::Slice(), torch::indexing::Slice(), i})
            .unsqueeze(-1);

    last_recurrent_state = last_recurrent_state * g_t;
    auto kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(-2);
    auto delta = (v_t - kv_mem) * beta_t;
    last_recurrent_state =
        last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2);
    core_attn_out.index(
        {torch::indexing::Slice(), torch::indexing::Slice(), i}) =
        (last_recurrent_state * q_t.unsqueeze(-1)).sum(-2);
  }

  if (!output_final_state) {
    last_recurrent_state = torch::Tensor();
  }

  core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype);
  return std::make_pair(core_attn_out, last_recurrent_state);
}

class TritonRecurrentGatedDeltaRuleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    try {
      torch::zeros({1}, torch::TensorOptions().device("npu:0"));
      tensor_options_ =
          torch::TensorOptions().dtype(torch::kFloat16).device("npu:0");
      npu_available_ = true;
    } catch (...) {
      tensor_options_ =
          torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU);
      npu_available_ = false;
      return;
    }

    torch::manual_seed(42);
    torch_npu::init_npu(device_str_);
  }

  void TearDown() override {
    if (npu_available_) {
      try {
        torch_npu::finalize_npu();
      } catch (...) {
      }
    }
  }

  torch::TensorOptions tensor_options_;
  bool npu_available_ = false;
  std::string device_str_ = "npu:" + std::to_string(kDeviceId);
};

TEST_F(TritonRecurrentGatedDeltaRuleTest, MultiBatchTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "NPU device not available";
  }

  auto device = at::Device(device_str_);
  constexpr int64_t batch = 4;
  constexpr int64_t T = 1;
  constexpr int64_t num_heads = 4;
  constexpr int64_t num_v_heads = 8;
  constexpr int64_t k_head_dim = 128;
  constexpr int64_t v_head_dim = 128;
  constexpr bool use_qk_l2norm_in_kernel = true;

  auto q = torch::randn(
      {batch, T, num_heads, k_head_dim},
      torch::TensorOptions().dtype(torch::kFloat16).device(device));
  auto k = torch::randn(
      {batch, T, num_heads, k_head_dim},
      torch::TensorOptions().dtype(torch::kFloat16).device(device));
  auto v = torch::randn(
      {batch, T, num_v_heads, v_head_dim},
      torch::TensorOptions().dtype(torch::kFloat16).device(device));
  auto g = torch::randn(
      {batch, T, num_v_heads},
      torch::TensorOptions().dtype(torch::kFloat32).device(device));
  auto beta = torch::randn(
      {batch, T, num_v_heads},
      torch::TensorOptions().dtype(torch::kFloat32).device(device));
  auto initial_state = torch::randn(
      {batch, T, num_v_heads, k_head_dim, v_head_dim},
      torch::TensorOptions().dtype(torch::kFloat32).device(device));

  int64_t L = batch * T;
  auto q_flat = q.reshape({1, L, num_heads, k_head_dim});
  auto k_flat = k.reshape({1, L, num_heads, k_head_dim});
  auto v_flat = v.reshape({1, L, num_v_heads, v_head_dim});
  auto g_flat = g.reshape({1, L, num_v_heads});
  auto beta_flat = beta.reshape({1, L, num_v_heads});

  std::vector<int64_t> cu_seqlens_vec;
  for (int64_t i = 0; i <= batch; ++i) {
    cu_seqlens_vec.push_back(i * T);
  }
  auto cu_seqlens =
      torch::tensor(cu_seqlens_vec,
                    torch::TensorOptions().dtype(torch::kInt64).device(device));

  auto initial_state_reshaped = initial_state.index({torch::indexing::Slice(),
                                                     0,
                                                     torch::indexing::Slice(),
                                                     torch::indexing::Slice(),
                                                     torch::indexing::Slice()});

  auto q_cpu = q.cpu();
  auto k_cpu = k.cpu();
  auto v_cpu = v.cpu();
  auto g_cpu = g.cpu();
  auto beta_cpu = beta.cpu();
  auto initial_state_cpu = initial_state.cpu();

  torch::Tensor q_golden = q_cpu;
  torch::Tensor k_golden = k_cpu;
  if (num_v_heads > num_heads) {
    int64_t repeat_factor = num_v_heads / num_heads;
    q_golden = q_cpu.repeat_interleave(repeat_factor, 2);
    k_golden = k_cpu.repeat_interleave(repeat_factor, 2);
  }

  auto initial_state_golden =
      initial_state_cpu.index({torch::indexing::Slice(),
                               0,
                               torch::indexing::Slice(),
                               torch::indexing::Slice(),
                               torch::indexing::Slice()});

  auto [torch_o, torch_state] =
      torch_recurrent_gated_delta_rule(q_golden,
                                       k_golden,
                                       v_cpu,
                                       g_cpu,
                                       beta_cpu,
                                       c10::make_optional(initial_state_golden),
                                       true,
                                       use_qk_l2norm_in_kernel);

  auto npu_stream = c10_npu::getCurrentNPUStream(0);
  auto [triton_o, triton_state] = npu_fused_recurrent_gated_delta_rule(
      q_flat,
      k_flat,
      v_flat,
      g_flat,
      c10::make_optional(beta_flat),
      c10::nullopt,
      c10::make_optional(initial_state_reshaped),
      false,
      c10::make_optional(cu_seqlens),
      c10::nullopt,
      c10::nullopt,
      use_qk_l2norm_in_kernel);
  aclrtSynchronizeStream(npu_stream.stream());

  auto triton_o_reshaped =
      triton_o.reshape({batch, T, num_v_heads, v_head_dim});
  auto triton_o_cpu = triton_o_reshaped.cpu();
  auto triton_state_cpu = triton_state.cpu();

  torch::Tensor triton_state_reshaped;
  if (triton_state_cpu.dim() == 4) {
    int64_t L = batch * T;
    if (triton_state_cpu.size(0) == L) {
      std::vector<torch::Tensor> state_slices;
      for (int64_t i = 0; i < batch; ++i) {
        int64_t last_token_idx = (i + 1) * T - 1;
        state_slices.push_back(
            triton_state_cpu.index({last_token_idx,
                                    torch::indexing::Slice(),
                                    torch::indexing::Slice(),
                                    torch::indexing::Slice()}));
      }
      triton_state_reshaped = torch::stack(state_slices, 0);
    } else {
      triton_state_reshaped = triton_state_cpu;
    }
  } else {
    triton_state_reshaped = triton_state_cpu;
  }

  auto o_diff = torch::abs(torch_o - triton_o_cpu);
  float o_max_diff = torch::max(o_diff).template item<float>();
  EXPECT_LT(o_max_diff, kTolerance) << "Output: max diff (" << o_max_diff
                                    << ") > tolerance (" << kTolerance << ")";

  if (torch_state.defined() && triton_state_reshaped.defined()) {
    auto state_diff = torch::abs(torch_state - triton_state_reshaped);
    float state_max_diff = torch::max(state_diff).template item<float>();
    EXPECT_LT(state_max_diff, kStateTolerance)
        << "State: max diff (" << state_max_diff << ") > tolerance ("
        << kStateTolerance << ")";
  }
}

}  // namespace xllm::kernel::npu

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  bool npu_available = false;
  std::string device_str =
      "npu:" + std::to_string(xllm::kernel::npu::kDeviceId);
  try {
    auto test_tensor =
        torch::zeros({1}, torch::TensorOptions().device(device_str));
    (void)test_tensor;
    npu_available = true;
  } catch (...) {
    npu_available = false;
  }

  if (!npu_available) {
    LOG(WARNING) << "NPU device not available, skipping all tests.";
    return 0;
  }

  int result = RUN_ALL_TESTS();
  return result;
}
