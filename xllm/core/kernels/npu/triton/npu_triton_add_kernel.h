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

#include <limits>

#include "experiment/runtime/runtime/rt.h"
#include "kernel_launchers.h"

namespace xllm::kernel::npu {

torch::Tensor npu_triton_add_kernel(const torch::Tensor& x,
                                    const torch::Tensor& y,
                                    int64_t n_elements,
                                    int32_t grid_x,
                                    int32_t grid_y = 1,
                                    int32_t grid_z = 1);

}  // namespace xllm::kernel::npu
