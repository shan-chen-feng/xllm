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

 #pragma once

 #include <torch/torch.h>
 #include <torch_npu/csrc/aten/NPUNativeFunctions.h>
 #include <torch_npu/csrc/core/npu/NPUStream.h>
 #include <torch_npu/torch_npu.h>
 #include <c10/util/Optional.h>
 
namespace xllm::kernel::npu {

std::pair<torch::Tensor, torch::Tensor> fused_recurrent_gated_delta_rule(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& g,
    const c10::optional<torch::Tensor>& beta = c10::nullopt,
    const c10::optional<float>& scale = c10::nullopt,
    const c10::optional<torch::Tensor>& initial_state = c10::nullopt,
    bool inplace_final_state = true,
    const c10::optional<torch::Tensor>& cu_seqlens = c10::nullopt,
    const c10::optional<torch::Tensor>& ssm_state_indices = c10::nullopt,
    const c10::optional<torch::Tensor>& num_accepted_tokens = c10::nullopt,
    bool use_qk_l2norm_in_kernel = false
);

}  // namespace xllm::kernel::npu
 
 
 