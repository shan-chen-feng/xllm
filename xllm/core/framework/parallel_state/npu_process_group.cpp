/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "npu_process_group.h"

#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <unistd.h>

#include <c10d/ProcessGroup.hpp>
#include <c10d/TCPStore.hpp>
#include <torch_npu/csrc/distributed/ProcessGroupHCCL.hpp>

#include "hccl/hccl.h"
namespace {
inline bool is_npu(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return false;
  }
  return tensor.device().is_privateuseone();
}

torch::Tensor flatten_for_scatter_gather(std::vector<torch::Tensor>& tensors) {
  auto& t = tensors[0];
  std::vector<int64_t> sizes{static_cast<int64_t>(tensors.size())};
  sizes.insert(sizes.end(), t.sizes().begin(), t.sizes().end());
  return torch::empty(sizes, t.options());
}

HcclDataType to_hccl_data_type(const torch::Tensor& input) {
  const auto type = input.scalar_type();
  switch (type) {
    case torch::kFloat:
      return HCCL_DATA_TYPE_FP32;
    case torch::kHalf:
      return HCCL_DATA_TYPE_FP16;
    case torch::kDouble:
      return HCCL_DATA_TYPE_FP64;
    case torch::kLong:
      return HCCL_DATA_TYPE_INT64;
    case torch::kInt:
      return HCCL_DATA_TYPE_INT32;
    case torch::kChar:
      return HCCL_DATA_TYPE_INT8;
    case torch::kByte:
      return HCCL_DATA_TYPE_UINT8;
    case torch::kBool:
      return HCCL_DATA_TYPE_UINT8;
    case torch::kBFloat16:
      return HCCL_DATA_TYPE_BFP16;
    default:
      LOG(FATAL) << "Unconvertible HCCL type " << type;
  }
}

void check_input(torch::Tensor input) {
  CHECK(is_npu(input)) << "input should be npu tensor";
  CHECK(input.is_contiguous()) << "input should be contiguous";
  CHECK(!input.is_sparse()) << "input have to be npu dense tensor";
}
}  // namespace

namespace xllm {

int32_t ProcessGroupImpl::group_id_ = 0;

ProcessGroupImpl::ProcessGroupImpl(int32_t global_rank,
                                   int32_t world_size,
                                   int32_t rank_size,
                                   int32_t port,
                                   bool trans,
                                   const std::string& host,
                                   const std::string& group_name,
                                   const torch::Device& device)
    : ProcessGroup(global_rank, world_size, device),
      comm_stream_(c10_npu::getNPUStreamFromPool(device.index())) {
  c10::intrusive_ptr<c10d_npu::ProcessGroupHCCL::Options> hccl_pg_options =
      c10d_npu::ProcessGroupHCCL::Options::create();
#if TORCH_VERSION_MAJOR > 2 || \
    (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 7)
  LOG(INFO) << "inside set group name";
  hccl_pg_options->group_name = group_name;
#endif
  group_size_ = rank_size;
  int32_t rank = global_rank;
  if (world_size != rank_size) {
    auto [local_rank, group_ranks] =
        get_group_rank(world_size, global_rank, rank_size, trans);
    std::vector<uint32_t> uint32_ranks;
    for (auto rank : group_ranks) {
      uint32_ranks.push_back(static_cast<uint32_t>(rank));
    }
    hccl_pg_options->global_ranks_in_group = uint32_ranks;
    rank = local_rank;
  }
  hccl_pg_options->group_id = std::to_string(group_id_++);
  LOG(INFO) << "create tcp store";
  auto store = create_tcp_store(host, port, rank, world_size);
  LOG(INFO) << "finish tcp store";
  pg_ = std::make_unique<c10d_npu::ProcessGroupHCCL>(
      store, rank, rank_size, hccl_pg_options);
  LOG(INFO) << "finish hhh";
}

ProcessGroupImpl::ProcessGroupImpl(int32_t global_rank,
                                   int32_t local_rank,
                                   const std::vector<int32_t>& group_ranks,
                                   int32_t world_size,
                                   int32_t rank_size,
                                   int32_t port,
                                   const std::string& host,
                                   const std::string& group_name,
                                   const torch::Device& device)
    : ProcessGroup(global_rank, world_size, device),
      comm_stream_(c10_npu::getNPUStreamFromPool(device.index())) {
  c10::intrusive_ptr<c10d_npu::ProcessGroupHCCL::Options> hccl_pg_options =
      c10d_npu::ProcessGroupHCCL::Options::create();
#if TORCH_VERSION_MAJOR > 2 || \
    (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 7)
  LOG(INFO) << "inside set group name";
  hccl_pg_options->group_name = group_name;
#endif
  group_size_ = rank_size;
  for (auto rank : group_ranks) {
    rank_per_group_.push_back(static_cast<uint32_t>(rank));
  }
  LOG(INFO) << "port is " << port;
  LOG(INFO) << "host is " << host;
  LOG(INFO) << "local_rank is " << local_rank;
  LOG(INFO) << "group name is " << group_name;
  LOG(INFO) << rank_per_group_[0] << rank_per_group_[1];
  LOG(INFO) << "rank_size is " << rank_size;
  LOG(INFO) << "world_size is " << world_size;
  hccl_pg_options->group_id = std::to_string(group_id_++);
  hccl_pg_options->global_ranks_in_group = rank_per_group_;
  auto store = create_tcp_store(host, port, local_rank);
  pg_ = std::make_unique<c10d_npu::ProcessGroupHCCL>(
      store, local_rank, rank_size, hccl_pg_options);
}

// Destructor.
ProcessGroupImpl::~ProcessGroupImpl() {
  if (pg_) {
    pg_->shutdown();
  } else {
    HCCLCHECK(HcclCommDestroy(comm_));
  }
  c10_npu::NPUCachingAllocator::emptyCache();
}

ProcessGroupImpl::ProcessGroupImpl(int rank,
                                   int world_size,
                                   const torch::Device& device,
                                   HcclComm comm)
    : ProcessGroup(rank, world_size, device),
      comm_(comm),
      comm_stream_(c10_npu::getNPUStreamFromPool(device.index())) {}
/*
std::vector<uint32_t> ProcessGroupImpl::get_rank_per_group(
    const std::string& group_type) {
  auto cfg_parallel_info = dit_mapping_npu_->get_parallel_info(group_type);
  auto group_id = cfg_parallel_info.current_group_id();

  auto& rank_per_group_int = cfg_parallel_info.rank_per_group()[group_id];
  std::vector<uint32_t> rank_per_group_uint(rank_per_group_int.size());
  std::transform(rank_per_group_int.begin(),
                 rank_per_group_int.end(),
                 rank_per_group_uint.begin(),
                 [](int rank) { return static_cast<uint32_t>(rank); });
  return rank_per_group_uint;
}
*/
}  // namespace xllm
