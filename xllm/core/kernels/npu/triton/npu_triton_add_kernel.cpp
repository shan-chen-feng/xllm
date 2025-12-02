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

#include "npu_triton_add_kernel.h"

namespace xllm::kernel::npu {

namespace {

void validate_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.defined(), name, " tensor is not defined");
  TORCH_CHECK(tensor.is_contiguous(), name, " tensor must be contiguous");
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1);
}

}  // namespace

torch::Tensor npu_triton_add_kernel(const torch::Tensor& x,
                                    const torch::Tensor& y,
                                    int64_t n_elements,
                                    int32_t grid_x,
                                    int32_t grid_y,
                                    int32_t grid_z) {
  validate_tensor(x, "input_x");
  validate_tensor(y, "input_y");
  TORCH_CHECK(x.sizes() == y.sizes(), "input sizes must be same.");
  TORCH_CHECK(grid_x > 0, "grid_x (BLOCK_SIZE) must > 0.");

  torch::Tensor out = torch::empty_like(x);

  int32_t device_id = x.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();

  void* x_ptr = const_cast<void*>(x.data_ptr());
  void* y_ptr = const_cast<void*>(y.data_ptr());
  void* out_ptr = out.data_ptr();

  auto ret = TritonLauncher::add_kernel(stream,
                                        grid_x,
                                        grid_y,
                                        grid_z,
                                        nullptr,
                                        nullptr,
                                        x_ptr,
                                        y_ptr,
                                        out_ptr,
                                        static_cast<int32_t>(n_elements));
  TORCH_CHECK(
      ret == RT_ERROR_NONE, "launch_add_kernel failed with error ", ret);
  return out;
}

}  // namespace xllm::kernel::npu
