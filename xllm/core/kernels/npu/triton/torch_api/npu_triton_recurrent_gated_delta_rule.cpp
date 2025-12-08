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

#include "npu_triton_recurrent_gated_delta_rule.h"

#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "experiment/runtime/runtime/rt.h"
#include "kernel_launchers.h"

namespace xllm::kernel::npu {
namespace {

void validate_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.defined(), name, " tensor is not defined");
  TORCH_CHECK(tensor.is_contiguous(), name, " tensor must be contiguous");
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
              name,
              " tensor must be on NPU device");
}

int64_t next_power_of_2(int64_t x) {
  if (x <= 1) return 1;
  TORCH_CHECK(x > 0 && x <= (int64_t(1) << 62),
              "x too large for next_power_of_2");
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x |= x >> 32;
  return x + 1;
}

int64_t cdiv(int64_t a, int64_t div) {
  TORCH_CHECK(div > 0, "cdiv divisor must be positive");
  return (a + div - 1) / div;
}

}  // namespace

// default use_qk_l2norm_in_kernel is True
std::pair<torch::Tensor, torch::Tensor> fused_recurrent_gated_delta_rule(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& g,
    const c10::optional<torch::Tensor>& beta,
    const c10::optional<float>& scale,
    const c10::optional<torch::Tensor>& initial_state,
    bool inplace_final_state,
    const c10::optional<torch::Tensor>& cu_seqlens,
    const c10::optional<torch::Tensor>& ssm_state_indices,
    const c10::optional<torch::Tensor>& num_accepted_tokens,
    bool use_qk_l2norm_in_kernel) {
  // validate
  auto q_shape = q.sizes();
  auto k_shape = k.sizes();
  auto v_shape = v.sizes();
  if (cu_seqlens == c10::nullopt && q_shape[0] != 1) {
    // TODO error handle batch size is not 1 when cu_seqlens is not provided
    TORCH_CHECK(false, "batch size is not 1 when cu_seqlens is not provided");
  }
  float scale_value = 1.0f;
  if (scale.has_value()) {
    scale_value = scale.value();
  } else {
    scale_value = 1.0 / std::sqrt(k_shape.back());
  }

  torch::Tensor beta_tensor;
  if (beta == c10::nullopt) {
    std::vector<int64_t> beta_shape(q_shape.begin(), q_shape.end() - 1);
    beta_tensor = torch::ones(
        beta_shape,
        torch::TensorOptions().dtype(torch::kFloat32).device(q.device()));
  } else {
    beta_tensor = beta.value();
  }

  int64_t batch = k_shape[0];           // B
  int64_t seq = k_shape[1];             // T
  int64_t num_k_head = k_shape[2];      // H
  int64_t k_head_dim = k_shape[3];      // K
  int64_t num_v_head = v_shape[2];      // HV
  int64_t v_head_dim = v_shape.back();  // V
  int N = batch;
  if (cu_seqlens.has_value()) {
    // cu_seqlens is a 1D LongTensor
    N = cu_seqlens.value().numel() - 1;
  }

  int64_t BK = next_power_of_2(k_head_dim);
  int64_t BV = std::min(next_power_of_2(v_head_dim), static_cast<int64_t>(8));
  int64_t NK = cdiv(k_head_dim, BK);
  int64_t NV = cdiv(v_head_dim, BV);
  TORCH_CHECK(NK == 1, "NK > 1 is not supported yet");
  int64_t num_stages = 3;
  int64_t num_warps = 1;

  std::vector<int64_t> o_shape{NK};
  o_shape.insert(o_shape.end(), v_shape.begin(), v_shape.end());
  torch::Tensor o = torch::empty(
      o_shape, torch::TensorOptions().dtype(q.dtype()).device(q.device()));

  torch::Tensor final_state;
  if (inplace_final_state) {
    TORCH_CHECK(
        initial_state.has_value(),
        "initial_state must be provided when inplace_final_state is true");
    final_state = initial_state.value();
  } else {
    TORCH_CHECK(
        initial_state.has_value(),
        "initial_state must be provided when inplace_final_state is false");
    std::vector<int64_t> final_state_shape{
        seq, num_v_head, k_head_dim, v_head_dim};
    final_state = torch::empty(final_state_shape,
                               torch::TensorOptions()
                                   .dtype(initial_state.value().dtype())
                                   .device(initial_state.value().device()));
  }

  // in triton kernel stride param will be tl.constexpr
  /*
  int64_t stride_init_state_token = initial_state.stride(0);
  int64_t stride_final_state_token = final_state.stride(0);
  int64_t stride_indices_seq, stride_indices_tok = 0;
  if (ssm_state_indices == c10::nullopt) {
      stride_indices_seq = 1;
      stride_indices_tok = 1;
  } else if (ssm_state_indices.value().ndim() == 1) {
      stride_indices_seq = ssm_state_indices.value().stride(0);
      stride_indices_tok = 1;
  } else {
      stride_indices_seq = ssm_state_indices.value().stride();
  }
  */

  auto npu_stream = c10_npu::getCurrentNPUStream();
  rtStream_t stream = static_cast<rtStream_t>(npu_stream.stream());

  // prepare launcher input
  int32_t gridX = NK;
  int32_t gridY = NV;
  int32_t gridZ = N * num_v_head;
  void* q_ptr = const_cast<void*>(q.data_ptr());
  void* k_ptr = const_cast<void*>(k.data_ptr());
  void* v_ptr = const_cast<void*>(v.data_ptr());
  void* g_ptr = const_cast<void*>(g.data_ptr());
  void* beta_ptr = const_cast<void*>(beta_tensor.data_ptr());
  void* o_ptr = o.data_ptr();
  void* initial_state_ptr =
      initial_state.has_value()
          ? const_cast<void*>(initial_state.value().data_ptr())
          : nullptr;
  void* final_state_ptr = final_state.data_ptr();
  void* cu_seqlens_ptr =
      cu_seqlens.has_value() ? cu_seqlens.value().data_ptr() : nullptr;
  void* ssm_state_indices_ptr = ssm_state_indices.has_value()
                                    ? ssm_state_indices.value().data_ptr()
                                    : nullptr;
  void* num_accepted_tokens_ptr = num_accepted_tokens.has_value()
                                      ? num_accepted_tokens.value().data_ptr()
                                      : nullptr;

  auto ret = launchers::fused_recurrent_gated_delta_rule_kernel(
      stream,
      gridX,
      gridY,
      gridZ,
      nullptr,
      nullptr,
      q_ptr,
      k_ptr,
      v_ptr,
      g_ptr,
      beta_ptr,
      o_ptr,
      initial_state_ptr,
      final_state_ptr,
      cu_seqlens_ptr,
      ssm_state_indices_ptr,
      num_accepted_tokens_ptr,
      scale_value,
      static_cast<int32_t>(N),
      static_cast<int32_t>(seq));

  TORCH_CHECK(
      ret == RT_ERROR_NONE,
      "launch_fused_recurrent_gated_delta_rule_kernel failed with error ",
      ret);

  o = o.squeeze(0);
  return std::make_pair(o, final_state);
}

}  // namespace xllm::kernel::npu
