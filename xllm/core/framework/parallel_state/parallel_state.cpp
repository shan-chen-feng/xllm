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

#include "parallel_state.h"

#include "core/util/utils.h"
#include "runtime/options.h"
#include "util/net.h"

#if defined(USE_NPU)
#include "hccl/hccl.h"
#include "npu_process_group.h"
#endif

namespace xllm {
namespace parallel_state {

std::optional<ParallelArgs> get_dp_attn_parallel_args(
    const ParallelArgs& parallel_args) {
  if (parallel_args.dp_size() <= 1) {
    return std::nullopt;
  }

  // tp=1 in each dp group
  if (parallel_args.dp_size() == parallel_args.world_size()) {
    return ParallelArgs(0,  // local rank
                        1,  // world_size
                        nullptr,
                        nullptr,
                        parallel_args.dp_size());
  }

  return ParallelArgs(parallel_args.dp_local_process_group_->rank(),
                      parallel_args.dp_local_process_group_->world_size(),
                      parallel_args.dp_local_process_group_,
                      nullptr,
                      parallel_args.dp_size());
}

torch::Tensor gather(const torch::Tensor& input,
                     ProcessGroup* process_group,
                     int32_t dim) {
  if (!process_group) {
    return input;
  }
  const int32_t world_size = process_group->world_size();
  if (world_size == 1) {
    return input;
  }

  const int32_t rank = process_group->rank();
  std::vector<torch::Tensor> tensors(world_size);
  for (int64_t i = 0; i < world_size; ++i) {
    tensors[i] = torch::empty_like(input);
  }
  // blocking call
  process_group->allgather(input, tensors);
  return torch::cat(tensors, /*dim=*/dim).contiguous();
}

torch::Tensor gather(const torch::Tensor& input,
                     ProcessGroup* process_group,
                     const std::vector<int32_t>& token_num_list) {
  if (!process_group) return input;
  const int32_t world_size = process_group->world_size();
  const int32_t rank = process_group->rank();
  if (world_size == 1) return input;

  CHECK_EQ(token_num_list.size(), world_size)
      << "token_num_list size " << token_num_list.size()
      << " does not match world_size " << world_size;

  const bool num_tokens_equal =
      std::all_of(token_num_list.begin(),
                  token_num_list.end(),
                  [first_token_num = token_num_list[0]](int64_t num) {
                    return num == first_token_num;
                  });
  if (num_tokens_equal) {
    return gather(input, process_group, 0);
  }

  int32_t max_num_tokens = xllm::util::max(token_num_list);
  auto options = input.options();
  int32_t num_padding = max_num_tokens - token_num_list[rank];
  torch::Tensor padded_input = input;
  // pad local input if needed
  if (num_padding > 0) {
    std::vector<int64_t> pad = {0, 0, 0, num_padding};
    // Explicitly calling kConstant and value of 0 ensures consistency across
    // platforms and versions.
    padded_input = torch::nn::functional::pad(
        input, torch::nn::functional::PadFuncOptions(pad));
  }

  // perform allgather
  std::vector<torch::Tensor> gathered_tensors(world_size);
  for (size_t i = 0; i < world_size; ++i)
    gathered_tensors[i] = torch::empty_like(padded_input);
  process_group->allgather(padded_input, gathered_tensors);

  // zero-copy assembly (performance critical)
  // directly allocate output tensor of the final size, avoiding huge
  // intermediate tensors produced by cat.
  int64_t total_tokens = xllm::util::sum(token_num_list);
  auto out_shape = input.sizes().vec();
  out_shape[0] = total_tokens;
  torch::Tensor output = torch::empty(out_shape, options);

  int64_t offset = 0;
  for (size_t i = 0; i < world_size; ++i) {
    int32_t valid_tokens = token_num_list[i];
    if (valid_tokens > 0) {
      // copy only the valid part of gathered_tensors[i] into the designated
      // position in output. This avoids the inefficient process of cat followed
      // by slice.
      output.slice(0, offset, offset + valid_tokens)
          .copy_(gathered_tensors[i].slice(0, 0, valid_tokens));
      offset += valid_tokens;
    }
  }

  return output;
}

torch::Tensor all_gather_interleaved(const torch::Tensor& input,
                                     ProcessGroup* process_group) {
  if (!process_group) {
    return input;
  }
  const int32_t world_size = process_group->world_size();
  const int32_t rank = process_group->rank();
  if (world_size == 1) {
    return input;
  }

  std::vector<torch::Tensor> gathered_tensors(world_size);
  for (int64_t i = 0; i < world_size; ++i) {
    gathered_tensors[i] = torch::empty_like(input);
  }
  process_group->allgather(input, gathered_tensors);

  int32_t dim = -1;
  size_t num_chunks = 3;
  std::vector<torch::Tensor> ordered_tensors;
  int64_t shard_size = input.size(dim) / num_chunks;
  for (size_t i = 0; i < num_chunks; ++i) {
    for (size_t j = 0; j < world_size; ++j) {
      auto shard_tensor =
          gathered_tensors[j].slice(dim, shard_size * i, shard_size * (i + 1));
      ordered_tensors.push_back(shard_tensor);
    }
  }
  return torch::cat(ordered_tensors, dim).contiguous();
}

torch::Tensor reduce(torch::Tensor& input, ProcessGroup* process_group) {
  if (!process_group) {
    return input;
  }
  const int32_t world_size = process_group->world_size();
  if (world_size == 1) {
    return input;
  }
  process_group->allreduce(input);
  return input;
}

torch::Tensor reduce_scatter(const torch::Tensor& input,
                             ProcessGroup* process_group) {
  // currently only support scatter_dim == 0
  if (!process_group) return input;
  const int32_t world_size = process_group->world_size();
  if (world_size == 1) return input;

  const int32_t rank = process_group->rank();
  const int64_t original_dim_size = input.size(0);

  // check if padding is needed
  // round up to the nearest multiple of world_size: (N + W - 1) / W * W or N +
  // (W - N%W)%W
  int64_t remainder = original_dim_size % world_size;
  int64_t target_size = (remainder == 0)
                            ? original_dim_size
                            : (original_dim_size + world_size - remainder);
  int64_t num_padding = target_size - original_dim_size;
  torch::Tensor padded_input = input;
  if (num_padding > 0) {
    std::vector<int64_t> pad = {0, 0, 0, num_padding};
    // Explicitly calling kConstant and value of 0 ensures consistency across
    // platforms and versions.
    padded_input = torch::nn::functional::pad(
        input, torch::nn::functional::PadFuncOptions(pad));
  }

  // prepare output tensor
  // at this point, padded_input size along dim 0 is divisible by world_size
  const int64_t padded_dim_size = padded_input.size(0);
  const int64_t chunk_size = padded_dim_size / world_size;

  auto output_shape = padded_input.sizes().vec();
  output_shape[0] = chunk_size;
  torch::Tensor output = torch::empty(output_shape, padded_input.options());

  // perform reduce scatter operation
  process_group->reduce_scatter(padded_input, output);

  // remove padding
  if (num_padding > 0) {
    int64_t global_start = rank * chunk_size;
    int64_t global_end = global_start + chunk_size;

    if (global_start >= original_dim_size) {
      return output.slice(0, 0, 0);
    } else if (global_end > original_dim_size) {
      return output.slice(0, 0, original_dim_size - global_start);
    }
  }

  return output;
}

torch::Tensor scatter(torch::Tensor input,
                      ProcessGroup* process_group,
                      int dim) {
  if (!process_group) {
    return input;
  }
  const int32_t world_size = process_group->world_size();
  if (world_size == 1) {
    return input;
  }

  // get the size for last dimension
  const int32_t dim_size = input.size(dim);
  CHECK(dim_size % world_size == 0)
      << "dim_size " << dim_size << " cannot be divided by world_size "
      << world_size;

  // torch::split does not create contiguous tensors by default.
  const auto tensor_list = input.split(dim_size / world_size, dim);
  const int32_t rank = process_group->rank();
  return tensor_list[rank];
}

std::vector<std::unique_ptr<ProcessGroup>> create_npu_process_groups(
    const std::vector<torch::Device>& devices) {
#if defined(USE_NPU)
  CHECK(!devices.empty()) << "devices should not be empty";

  std::vector<int> device_idxs;
  device_idxs.reserve(devices.size());
  for (const auto& device : devices) {
    device_idxs.push_back(device.index());
  }

  std::vector<HcclComm> comms(devices.size());
  const int32_t world_size = static_cast<int32_t>(devices.size());
  // HCCLCHECK(HcclCommInitAll(world_size, device_idxs.data(),comms.data()));

  std::vector<std::unique_ptr<ProcessGroup>> process_groups;
  process_groups.reserve(devices.size());
  for (int32_t i = 0; i < world_size; ++i) {
    process_groups.emplace_back(std::make_unique<ProcessGroupImpl>(
        /*rank=*/i, world_size, devices[i], comms[i]));
  }

  return process_groups;
#else
  LOG(FATAL) << "non-NPU device is not supported";
#endif
}

std::vector<std::unique_ptr<ProcessGroup>> create_local_process_groups(
    const std::vector<torch::Device>& devices,
    const runtime::Options& options) {
  CHECK(!devices.empty()) << "devices should not be empty";
  const int32_t world_size = static_cast<int32_t>(devices.size());

  std::vector<std::unique_ptr<ProcessGroup>> process_groups;
  process_groups.reserve(devices.size());

#if defined(USE_NPU)
  std::vector<HcclComm> comms(devices.size());
  for (int32_t i = 0; i < world_size; ++i) {
    process_groups.emplace_back(std::make_unique<ProcessGroupImpl>(
        /*rank=*/i, world_size, devices[i], comms[i]));
  }
#elif defined(USE_CUDA) || defined(USE_MLU) || defined(USE_ILU)
  // For GPU: use create_process_group with localhost
  // Parse port from options.master_node_addr() to support multiple instances
  std::string host;
  int port;

  // Parse port from options.master_node_addr()
  // Note: master_node_addr always has a default value (127.0.0.1:19888)
  net::parse_host_port_from_addr(
      options.master_node_addr().value(), host, port);

  // Override host to localhost for local communication
  host = "127.0.0.1";

  for (int32_t i = 0; i < world_size; ++i) {
    process_groups.emplace_back(create_process_group(
        /*rank=*/i,
        /*world_size=*/world_size,
        /*rank_size=*/world_size,
        /*port=*/port,
        /*trans=*/false,
        host,
        /*group_name=*/"local_tp_group",
        devices[i]));
  }
#else
  LOG(FATAL) << "Unsupported device type for create_local_process_groups";
#endif

  return process_groups;
}

/**
 * Supports two modes:
 * 1. scatter_idx=2, gather_idx=1: Ulysses parallel "split heads" scenario
 *    - Splits tensor along head dimension (dim2)
 *    - Reconstructs sequence dimension (dim1) to full length
 *    - Input: (bs, seqlen/P, hc, hs) -> Output: (bs, seqlen, hc/P, hs)
 *
 * 2. scatter_idx=1, gather_idx=2: Ulysses parallel "merge heads" scenario
 *    - Splits tensor along sequence dimension (dim1)
 *    - Reconstructs head dimension (dim2) to full head count
 *    - Input: (bs, seqlen, hc/P, hs) -> Output: (bs, seqlen/P, hc, hs)
 */
torch::Tensor all_to_all_4D(const torch::Tensor& input,
                            int32_t scatter_idx,
                            int32_t gather_idx,
                            ProcessGroup* process_group,
                            bool use_sync) {
  // Check input dimensions
  CHECK_EQ(input.dim(), 4) << "input must be 4D tensor, got " << input.dim()
                           << " and shape " << input.sizes();

  if (!process_group) {
    return input;
  }

  const int32_t world_size = process_group->world_size();
  if (world_size == 1) {
    return input;
  }

  // Branch 1: scatter_idx=2 and gather_idx=1
  // (Ulysses parallel "split heads" scenario)
  if (scatter_idx == 2 && gather_idx == 1) {
    // input: tensor sharded along dim 1 (bs, seqlen/P, hc, hs)
    // output: (bs, seqlen, hc/P, hs)
    auto sizes = input.sizes().vec();
    const int64_t bs = sizes[0];
    const int64_t shard_seqlen = sizes[1];
    const int64_t hc = sizes[2];
    const int64_t hs = sizes[3];
    const int64_t seqlen = shard_seqlen * world_size;
    CHECK(hc % world_size == 0)
        << "hc " << hc << " not divisible by world_size " << world_size;

    // Transpose groups of heads with the seq-len parallel dimension
    // (bs, seqlen/P, hc, hs) -> reshape -> (bs, seq_len/P, P, hc/P, hs)
    // -> transpose(0,2) -> (P, seq_len/P, bs, hc/P, hs)
    torch::Tensor input_t =
        input.reshape({bs, shard_seqlen, world_size, shard_hc, hs})
            .transpose(0, 2)
            .contiguous();

    torch::Tensor output = torch::empty_like(input_t);

    // Perform all-to-all communication
    // (P, seq_len/P, bs, hc/P, hs) scatter seqlen -> all2all ->
    // (P, seq_len/P, bs, hc/P, hs) scatter head
    process_group->alltoall(input_t, output);

    // Synchronize to ensure data consistency
    // This is critical for all-to-all operations to prevent race conditions
    if (use_sync) {
#if defined(USE_NPU)
      torch::npu::synchronize();
#elif defined(USE_CUDA) || defined(USE_MLU) || defined(USE_ILU)
      torch::cuda::synchronize();
#endif
    }

    // If scattering the seq-dim, transpose the heads back to original dimension
    output = output.reshape({seqlen, bs, shard_hc, hs});

    // (seq_len, bs, hc/P, hs) -> transpose(0,1) -> (bs, seq_len, hc/P, hs)
    output =
        output.transpose(0, 1).contiguous().reshape({bs, seqlen, shard_hc, hs});

    return output;
  }

  // Branch 2: scatter_idx=1 and gather_idx=2
  // (Ulysses parallel "merge heads" scenario)
  if (scatter_idx == 1 && gather_idx == 2) {
    // input: tensor sharded along dim 1 (bs, seqlen, hc/P, hs)
    // output: (bs, seqlen/P, hc, hs)
    auto sizes = input.sizes().vec();
    const int64_t bs = sizes[0];
    const int64_t seqlen = sizes[1];
    const int64_t shard_hc = sizes[2];
    const int64_t hs = sizes[3];
    const int64_t hc = shard_hc * world_size;
    const int64_t shard_seqlen = seqlen / world_size;
    CHECK(seqlen % world_size == 0)
        << "seqlen " << seqlen << " not divisible by world_size " << world_size;

    // Transpose groups of heads with the seq-len parallel dimension
    // (bs, seqlen, hc/P, hs) -> reshape -> (bs, P, seq_len/P, hc/P, hs)
    // -> transpose(0,3) -> (hc/P, P, seqlen/P, bs, hs)
    // -> transpose(0,1) -> (P, hc/P, seqlen/P, bs, hs)
    torch::Tensor input_t =
        input.reshape({bs, world_size, shard_seqlen, shard_hc, hs})
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape({world_size, shard_hc, shard_seqlen, bs, hs});

    torch::Tensor output = torch::empty_like(input_t);

    // Perform all-to-all communication
    // (P, bs x hc/P, seqlen/P, hs) scatter seqlen -> all2all ->
    // (P, bs x seq_len/P, hc/P, hs) scatter head
    process_group->alltoall(input_t, output);

    // Synchronize to ensure data consistency
    // This is critical for all-to-all operations to prevent race conditions
    if (use_sync) {
#if defined(USE_NPU)
      torch::npu::synchronize();
#elif defined(USE_CUDA) || defined(USE_MLU) || defined(USE_ILU)
      torch::cuda::synchronize();
#endif
    }

    // If scattering the seq-dim, transpose the heads back to original dimension
    output = output.reshape({hc, shard_seqlen, bs, hs});

    // (hc, seqlen/N, bs, hs) -> transpose(0,2) -> (bs, seqlen/N, hc, hs)
    output =
        output.transpose(0, 2).contiguous().reshape({bs, shard_seqlen, hc, hs});

    return output;
  }

  CHECK(false) << "scatter_idx must be 1 or 2 and gather_idx must be 1 or 2";
  return input;
}

}  // namespace parallel_state
}  // namespace xllm
