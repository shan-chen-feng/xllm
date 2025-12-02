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

#include <glog/logging.h>

#include <string>
#include <vector>

#include "experiment/runtime/runtime/rt.h"
#include "kernel_loader.h"

#define RT_ERROR_INVALID_VALUE -1
#define _DECLARE_STRUCT_FIELD(type, name) \
  type name __attribute__((aligned(sizeof(type))));
#define _DECLARE_PARAM(type, name) , type name
#define _DECLARE_ARG_VALUE(type, name) , name

// REG_KERNEL_ARGS(add_kernel, ARG_LIST_MACRO)
#define REG_KERNEL_ARGS(kernel_name, ARG_LIST_MACRO)     \
  struct __attribute__((packed)) _##kernel_name##_Args { \
    void* ffts_addr __attribute__((aligned(8)));         \
    void* syncBlockLock __attribute__((aligned(8)));     \
    void* workspace_addr __attribute__((aligned(8)));    \
    ARG_LIST_MACRO(_DECLARE_STRUCT_FIELD)                \
    int32_t gridX __attribute__((aligned(4)));           \
    int32_t gridY __attribute__((aligned(4)));           \
    int32_t gridZ __attribute__((aligned(4)));           \
  };

// REG_KERNEL_LAUNCHER(add_kernel, ARG_LIST_MACRO)
#define REG_KERNEL_LAUNCHER(kernel_name, ARG_LIST_MACRO)                       \
  static rtError_t kernel_name(                                                \
      rtStream_t stream,                                                       \
      int32_t gridX,                                                           \
      int32_t gridY,                                                           \
      int32_t gridZ,                                                           \
      void* workspace_addr,                                                    \
      void* syncBlockLock ARG_LIST_MACRO(_DECLARE_PARAM)) {                    \
    auto kernelHandle =                                                        \
        xllm::kernel::npu::KernelLoader::get_kernel(#kernel_name);             \
    if (!kernelHandle.is_valid()) {                                            \
      LOG(ERROR) << "Kernel '" << #kernel_name << "' is not registered";       \
      return RT_ERROR_INVALID_VALUE;                                           \
    }                                                                          \
                                                                               \
    int64_t workspaceSize = -1;                                                \
    int64_t lockInitValue = -1;                                                \
    int64_t lockNum = -1;                                                      \
    xllm::kernel::npu::KernelLoader::get_kernel_workspace_config(              \
        #kernel_name, workspaceSize, lockInitValue, lockNum);                  \
                                                                               \
    uint32_t blockNum = gridX * gridY * gridZ;                                 \
                                                                               \
    void* actual_workspace_addr = workspace_addr;                              \
    if (workspace_addr == nullptr && workspaceSize > 0) {                      \
      uint64_t totalWorkSpaceSize = workspaceSize * blockNum;                  \
      rtError_t ret =                                                          \
          rtMalloc(reinterpret_cast<void**>(&actual_workspace_addr),           \
                   totalWorkSpaceSize,                                         \
                   RT_MEMORY_HBM,                                              \
                   0);                                                         \
      if (ret != RT_ERROR_NONE) {                                              \
        LOG(ERROR) << "Failed to allocate workspace for kernel '"              \
                   << #kernel_name << "': error=" << ret;                      \
        return ret;                                                            \
      }                                                                        \
    }                                                                          \
                                                                               \
    void* actual_syncBlockLock = syncBlockLock;                                \
    if (syncBlockLock == nullptr && lockNum > 0) {                             \
      uint64_t syncBlockLockSize = lockNum * sizeof(int64_t);                  \
      rtError_t ret =                                                          \
          rtMalloc(reinterpret_cast<void**>(&actual_syncBlockLock),            \
                   syncBlockLockSize,                                          \
                   RT_MEMORY_HBM,                                              \
                   0);                                                         \
      if (ret != RT_ERROR_NONE) {                                              \
        LOG(ERROR) << "Failed to allocate syncBlockLock for kernel '"          \
                   << #kernel_name << "': error=" << ret;                      \
        if (actual_workspace_addr != workspace_addr) {                         \
          rtFree(actual_workspace_addr);                                       \
        }                                                                      \
        return ret;                                                            \
      }                                                                        \
      std::vector<int64_t> lockInitData(lockNum, lockInitValue);               \
      ret = rtMemcpy(actual_syncBlockLock,                                     \
                     syncBlockLockSize,                                        \
                     reinterpret_cast<void*>(lockInitData.data()),             \
                     syncBlockLockSize,                                        \
                     RT_MEMCPY_HOST_TO_DEVICE);                                \
      if (ret != RT_ERROR_NONE) {                                              \
        LOG(ERROR) << "Failed to initialize syncBlockLock for kernel '"        \
                   << #kernel_name << "': error=" << ret;                      \
        rtFree(actual_syncBlockLock);                                          \
        if (actual_workspace_addr != workspace_addr) {                         \
          rtFree(actual_workspace_addr);                                       \
        }                                                                      \
        return ret;                                                            \
      }                                                                        \
    }                                                                          \
                                                                               \
    void* ffts_addr = nullptr;                                                 \
    uint32_t ffts_len = 0;                                                     \
    rtError_t ret = rtGetC2cCtrlAddr((uint64_t*)&ffts_addr, &ffts_len);        \
    if (ret != RT_ERROR_NONE) {                                                \
      LOG(ERROR) << "rtGetC2cCtrlAddr failed: " << ret;                        \
      if (actual_syncBlockLock != syncBlockLock) {                             \
        rtFree(actual_syncBlockLock);                                          \
      }                                                                        \
      if (actual_workspace_addr != workspace_addr) {                           \
        rtFree(actual_workspace_addr);                                         \
      }                                                                        \
      return ret;                                                              \
    }                                                                          \
                                                                               \
    _##kernel_name##_Args args = {                                             \
        ffts_addr,                                                             \
        actual_syncBlockLock,                                                  \
        actual_workspace_addr ARG_LIST_MACRO(_DECLARE_ARG_VALUE),              \
        gridX,                                                                 \
        gridY,                                                                 \
        gridZ};                                                                \
                                                                               \
    ret = rtKernelLaunch(kernelHandle.get(),                                   \
                         blockNum,                                             \
                         static_cast<void*>(&args),                            \
                         sizeof(_##kernel_name##_Args),                        \
                         nullptr,                                              \
                         stream);                                              \
                                                                               \
    if (ret != RT_ERROR_NONE) {                                                \
      LOG(ERROR) << "rtKernelLaunch failed for '" << #kernel_name              \
                 << "': " << ret;                                              \
      if (actual_syncBlockLock != syncBlockLock) {                             \
        rtFree(actual_syncBlockLock);                                          \
      }                                                                        \
      if (actual_workspace_addr != workspace_addr) {                           \
        rtFree(actual_workspace_addr);                                         \
      }                                                                        \
      return ret;                                                              \
    }                                                                          \
                                                                               \
    return RT_ERROR_NONE;                                                      \
  }

