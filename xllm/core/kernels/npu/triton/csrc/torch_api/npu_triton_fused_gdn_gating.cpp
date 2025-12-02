#include "npu_triton_fused_gdn_gating.h"

#include <torch_npu/csrc/core/npu/NPUStream.h>

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

}  // namespace

std::pair<torch::Tensor, torch::Tensor> npu_fused_gdn_gating(
    const torch::Tensor& A_log,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& dt_bias,
    float beta,
    float threshold) {
  validate_tensor(A_log, "A_log");
  validate_tensor(a, "a");
  validate_tensor(b, "b");
  validate_tensor(dt_bias, "dt_bias");

  TORCH_CHECK(A_log.dtype() == torch::kFloat32, "A_log must be float32");
  TORCH_CHECK(a.dtype() == torch::kFloat16, "a must be float16");
  TORCH_CHECK(b.dtype() == torch::kFloat16, "b must be float16");
  TORCH_CHECK(dt_bias.dtype() == torch::kFloat32, "dt_bias must be float32");

  TORCH_CHECK(A_log.dim() == 1, "A_log must be 1D tensor");
  TORCH_CHECK(a.dim() == 2, "a must be 2D tensor (batch, num_heads)");
  TORCH_CHECK(b.dim() == 2, "b must be 2D tensor (batch, num_heads)");
  TORCH_CHECK(dt_bias.dim() == 1, "dt_bias must be 1D tensor");
  // TORCH_CHECK(std::abs(beta - 1) < std::numeric_limits<float>::epsilon(),
  // "beta must be 1, because this param compiled with tl.constexpr in triton
  // kernel"); TORCH_CHECK(std::abs(threshold - 1) <
  // std::numeric_limits<float>::epsilon(), "threshold must be 20, because this
  // param, default triton kernel compiled with tl.constexpr");

  int64_t batch = a.size(0);
  int64_t num_heads = a.size(1);

  TORCH_CHECK(A_log.size(0) == num_heads,
              "A_log size must match num_heads: ",
              A_log.size(0),
              " != ",
              num_heads);
  TORCH_CHECK(b.size(0) == batch && b.size(1) == num_heads,
              "b shape must match a shape");
  TORCH_CHECK(dt_bias.size(0) == num_heads,
              "dt_bias size must match num_heads: ",
              dt_bias.size(0),
              " != ",
              num_heads);

  torch::Tensor g = torch::empty(
      {1, batch, num_heads},
      torch::TensorOptions().dtype(torch::kFloat32).device(a.device()));

  torch::Tensor beta_output =
      torch::empty({1, batch, num_heads},
                   torch::TensorOptions().dtype(b.dtype()).device(b.device()));

  int32_t seq_len = 1;
  int32_t gridX = static_cast<int32_t>(batch);
  int32_t gridY = seq_len;
  int32_t gridZ =
      static_cast<int32_t>((num_heads + 7) / 8);  // ceil(num_heads / 8)

  auto npuStream = c10_npu::getCurrentNPUStream();
  rtStream_t stream = static_cast<rtStream_t>(npuStream.stream());

  void* gPtr = g.data_ptr();
  void* betaOutputPtr = beta_output.data_ptr();
  void* ALogPtr = const_cast<void*>(A_log.data_ptr());
  void* aPtr = const_cast<void*>(a.data_ptr());
  void* bPtr = const_cast<void*>(b.data_ptr());
  void* dtBiasPtr = const_cast<void*>(dt_bias.data_ptr());

  rtError_t ret = TritonLauncher::fused_gdn_gating_head8_kernel(
      stream,
      gridX,
      gridY,
      gridZ,
      nullptr,
      nullptr,
      gPtr,
      betaOutputPtr,
      ALogPtr,
      aPtr,
      bPtr,
      dtBiasPtr,
      seq_len);

  TORCH_CHECK(ret == RT_ERROR_NONE,
              "launch_fused_gdn_gating_kernel failed with error ",
              ret);

  return std::make_pair(g, beta_output);
}

}  // namespace xllm::kernel::npu
