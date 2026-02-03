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

#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUEvent.h>

#include <c10d/ProcessGroup.hpp>
#include <c10d/TCPStore.hpp>
#include <torch_npu/csrc/distributed/ProcessGroupHCCL.hpp>

namespace {
inline bool is_npu(const at::Tensor& tensor) {
  if (!tensor.defined()) {
    return false;
  }
  return tensor.device().is_privateuseone();
}
inline bool is_npu(const at::TensorOptions& options) {
  return options.device().is_privateuseone();
}
inline bool is_npu(const at::Device& device) {
  return device.is_privateuseone();
}
at::Tensor flatten_for_scatter_gather(std::vector<at::Tensor>& tensors) {
  auto& t = tensors[0];
  std::vector<int64_t> sizes{static_cast<int64_t>(tensors.size())};
  sizes.insert(sizes.end(), t.sizes().begin(), t.sizes().end());
  return at::empty(sizes, t.options());
}
#if defined(USE_NPU)
HcclDataType to_hccl_data_type(const torch::Tensor& input) {
  const auto type = input.scalar_type();
  switch (type) {
    case at::kFloat:
      return HCCL_DATA_TYPE_FP32;
    case at::kHalf:
      return HCCL_DATA_TYPE_FP16;
    case at::kDouble:
      return HCCL_DATA_TYPE_FP64;
    case at::kLong:
      return HCCL_DATA_TYPE_INT64;
    case at::kInt:
      return HCCL_DATA_TYPE_INT32;
    case at::kChar:
      return HCCL_DATA_TYPE_INT8;
    case at::kByte:
      return HCCL_DATA_TYPE_UINT8;
    case at::kBool:
      return HCCL_DATA_TYPE_UINT8;
    case at::kBFloat16:
      return HCCL_DATA_TYPE_BFP16;
    default:
      TORCH_CHECK(false, "Unconvertible HCCL type ", type);
  }
}
#endif
void check_input(torch::Tensor input) {
  CHECK(is_npu(input)) << "input should be npu tensor";
  CHECK(input.is_contiguous()) << "input should be contiguous";
  CHECK(!input.is_sparse()) << "input have to be npu dense tensor";
}
}  // namespace

namespace xllm {

ProcessGroupImpl::ProcessGroupImpl(int32_t global_rank,
                                   int32_t world_size,
                                   int32_t rank_size,
                                   int32_t port,
                                   bool trans,
                                   const std::string& host,
                                   const std::string& group_name,
                                   const torch::Device& device)
    : ProcessGroup(device),
      comm_stream_(c10_npu::getNPUStreamFromPool(device.index())) {
  c10::intrusive_ptr<c10d_npu::ProcessGroupHCCL::Options> hccl_pg_options =
      c10d_npu::ProcessGroupHCCL::Options::create();
#if TORCH_VERSION_MAJOR > 2 || \
    (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 7)
  hccl_pg_options->group_name = group_name;
#endif
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

  auto store = create_tcp_store(host, port, rank);
  pg_ = std::make_unique<c10d_npu::ProcessGroupHCCL>(
      store, rank, rank_size, hccl_pg_options);
}

// Destructor.
ProcessGroupImpl::~ProcessGroupImpl() {
  if (pg_) {
    pg_->shutdown();
  } else {
    HCCLCHECK(HcclCommDestroy(comm_));
  }
}

ProcessGroupImpl::ProcessGroupImpl(int rank,
                                   int world_size,
                                   const torch::Device& device,
                                   HcclComm comm)
    : ProcessGroup(device),
      comm_(comm),
      comm_stream_(c10_npu::getNPUStreamFromPool(device.index())) {}

void ProcessGroupImpl::alltoall_single(
    torch::Tensor send,
    torch::Tensor recv,
    const std::vector<int64_t>& send_splits,
    const std::vector<int64_t>& recv_splits,
    bool is_sync,
    std::shared_ptr<c10_npu::NPUEvent>* out_done) {
#if !defined(USE_NPU)
  LOG(FATAL) << "alltoall_single only supported with USE_NPU";
#else
  check_input(send);
  check_input(recv);
  CHECK(send.device() == device() && recv.device() == device())
      << "send/recv must be on the same device as the process group";
  const int P = world_size();
  CHECK((int)send_splits.size() == P && (int)recv_splits.size() == P)
      << "split sizes length must equal world_size";

  std::vector<uint64_t> sc(P), rc(P), sdisp(P), rdisp(P);
  uint64_t acc = 0;
  for (int i = 0; i < P; ++i) {
    sc[i] = static_cast<uint64_t>(send_splits[i]);
    sdisp[i] = acc;
    acc += sc[i];
  }
  acc = 0;
  for (int i = 0; i < P; ++i) {
    rc[i] = static_cast<uint64_t>(recv_splits[i]);
    rdisp[i] = acc;
    acc += rc[i];
  }

  auto dtype = to_hccl_data_type(send);
  auto compute_stream = c10_npu::getCurrentNPUStream();
  c10_npu::NPUEvent ready;
  ready.record(compute_stream);

  torch::DeviceGuard guard(device());
  // const auto prev_stream = c10_npu::getCurrentNPUStream();
  // c10_npu::setCurrentNPUStream(comm_stream_);
  ready.block(comm_stream_);  // compute -> comm
  c10_npu::NPUCachingAllocator::recordStream(send.storage().data_ptr(),
                                             comm_stream_);
  c10_npu::NPUCachingAllocator::recordStream(recv.storage().data_ptr(),
                                             comm_stream_);
  HCCLCHECK(HcclAlltoAllV(
      /*sendBuf=*/send.data_ptr(),
      /*sendCounts=*/sc.data(),
      /*sdispls=*/sdisp.data(),
      /*sendType=*/dtype,
      /*recvBuf=*/recv.data_ptr(),
      /*recvCounts=*/rc.data(),
      /*rdispls=*/rdisp.data(),
      /*recvType=*/dtype,
      /*comm=*/comm_,
      /*stream=*/comm_stream_.stream()));

  if (is_sync) {
    c10_npu::NPUEvent ev;
    ev.record(comm_stream_);
    ev.synchronize();
  } else {
    auto done = std::make_shared<c10_npu::NPUEvent>();
    done->record(comm_stream_);

    if (out_done) {
      *out_done = std::move(done);
    } else {
      done->block(compute_stream);
    }
  }
  // c10_npu::setCurrentNPUStream(prev_stream);
#endif
}

void ProcessGroupImpl::flush_comm_to_current() {
#if defined(USE_NPU)
  auto cur = c10_npu::getCurrentNPUStream();
  c10_npu::NPUEvent fence;
  fence.record(comm_stream_);  // 通信流 -> 事件
  fence.block(cur);            // 事件 -> 当前计算流
#endif
}

void ProcessGroupImpl::alltoall_equal(
    torch::Tensor send,
    torch::Tensor recv,
    bool is_sync,
    std::shared_ptr<c10_npu::NPUEvent>* out_done) {
#if !defined(USE_NPU)
  LOG(FATAL) << "alltoall_equal only supported with USE_NPU";
#else
  check_input(send);
  check_input(recv);
  const int P = world_size();
  CHECK(send.numel() % P == 0 && recv.numel() % P == 0)
      << "send/recv numel must be divisible by world_size";
  const int64_t per_rank_send = send.numel() / P;
  const int64_t per_rank_recv = recv.numel() / P;
  std::vector<int64_t> in_splits(P, per_rank_send);
  std::vector<int64_t> out_splits(P, per_rank_recv);
  alltoall_single(send, recv, in_splits, out_splits, is_sync, out_done);
#endif
}

}  // namespace xllm