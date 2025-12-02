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

#include "macros.h"

namespace xllm::kernel::npu {
namespace TritonLauncher {
/* add_kernel */
#define ADD_KERNEL_ARG_LIST(OP) \
  OP(void*, arg0)               \
  OP(void*, arg1)               \
  OP(void*, arg2)               \
  OP(int32_t, arg3)

// param 1 must be same as "kernel_name" in ${kernel_name}.json
REG_KERNEL_ARGS(add_kernel, ADD_KERNEL_ARG_LIST)
REG_KERNEL_LAUNCHER(add_kernel, ADD_KERNEL_ARG_LIST)
// 调用REG_KERNEL_LAUNCHER后 会产生相应的kernel接口 与triton python接口一致
 // add_kernel(rtStream_t stream, \
        int32_t gridX, int32_t gridY, int32_t gridZ, \
        void* workspace_addr, \
        void* syncBlockLock, \
        void* arg0, void* arg1, void* arg2, int32_t arg3)
/* end of add_kernel */

/* fused_gdn_gating_kernel */
#define FUSED_GDN_GATING_ARG_LIST(OP) \
  OP(void*, g)                        \
  OP(void*, beta_output)              \
  OP(void*, A_log)                    \
  OP(void*, a)                        \
  OP(void*, b) \        
    OP(void*, dt_bias) \ 
    OP(int32_t, seq_len)

REG_KERNEL_ARGS(fused_gdn_gating_head8_kernel, FUSED_GDN_GATING_ARG_LIST)
REG_KERNEL_LAUNCHER(fused_gdn_gating_head8_kernel, FUSED_GDN_GATING_ARG_LIST)
}  // namespace TritonLauncher
}  // namespace xllm::kernel::npu
