#pragma once

#include <torch/torch.h>
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/torch_npu.h>

namespace xllm::kernel::npu {

torch::Tensor layer_norm_fwd(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    double eps,
    const c10::optional<torch::Tensor>& z = c10::nullopt,
    int64_t group_size = -1,
    bool norm_before_gate = true,
    bool is_rms_norm = false);

}  // namespace xllm::kernel::npu
