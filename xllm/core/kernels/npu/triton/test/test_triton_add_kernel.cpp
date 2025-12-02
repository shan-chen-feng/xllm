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
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include <glog/logging.h>

#include "../npu_triton_add_kernel.h"
#include "kernel_loader.h"

namespace {

constexpr int64_t kSize = 98432;
constexpr int32_t kBlockSize = 1024;

int run_add_kernel_test(const std::string& binary_path,
                        const std::string& kernel_name) {
  aclError ret = aclInit(nullptr);
  if (ret != ACL_SUCCESS) {
    LOG(ERROR) << "aclInit failed: error = " << ret;
    return -1;
  }

  int32_t device_id = 0;
  ret = aclrtSetDevice(device_id);
  if (ret != ACL_SUCCESS) {
    LOG(ERROR) << "Failed to set device: error = " << ret;
    return -1;
  }

  auto kernel_handle =
      xllm::kernel::npu::KernelLoader::load_kernel(kernel_name, binary_path);
  if (!kernel_handle.is_valid()) {
    LOG(ERROR) << "Failed to load kernel: " << kernel_name;
    return -1;
  }

  aclrtStream stream;
  ret = aclrtCreateStream(&stream);
  if (ret != ACL_SUCCESS) {
    LOG(ERROR) << "Failed to create stream, error = " << ret;
    return -1;
  }

  torch::manual_seed(0);

  torch::Tensor x = torch::rand(
      {kSize}, torch::TensorOptions().dtype(torch::kFloat32).device("npu:0"));
  torch::Tensor y = torch::rand(
      {kSize}, torch::TensorOptions().dtype(torch::kFloat32).device("npu:0"));

  torch::Tensor output_torch = x + y;
  torch::Tensor output_triton =
      xllm::kernel::npu::npu_triton_add_kernel(x, y, kSize, kBlockSize);

  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    LOG(ERROR) << "Stream synchronize failed: error = " << ret;
    return -1;
  }

  torch::Tensor diff = torch::abs(output_torch - output_triton);
  float max_diff = torch::max(diff).item<float>();
  const float kTolerance = 1e-5f;
  bool passed = max_diff < kTolerance;

  if (passed) {
    LOG(INFO) << "add kernel test passed, max diff = " << max_diff;
  } else {
    LOG(ERROR) << "add kernel test failed, max diff = " << max_diff;
  }

  ret = aclrtDestroyStream(stream);
  if (ret != ACL_SUCCESS) {
    LOG(ERROR) << "aclrtDestroyStream failed: error = " << ret;
    return -1;
  }

  ret = aclFinalize();
  if (ret != ACL_SUCCESS) {
    LOG(ERROR) << "aclFinalize failed: error = " << ret;
    return -1;
  }

  return passed ? 0 : -1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    LOG(ERROR) << "Usage: " << argv[0] << " <kernel_bin_path> <kernel_name>";
    return -1;
  }

  std::string binary_path = argv[1];
  std::string kernel_name = argv[2];

  return run_add_kernel_test(binary_path, kernel_name);
}
