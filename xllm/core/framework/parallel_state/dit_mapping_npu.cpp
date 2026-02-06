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

#include "dit_mapping_npu.h"

#include <glog/logging.h>

namespace xllm {

DiTMappingNPU::DiTMappingNPU(const int32_t world_size,
                             const int32_t rank,
                             const Options& options)
    : rank_(rank), options_(options), world_size_(world_size) {
  tp_.backend("hccl");
  sp_.backend("hccl");
  cfg_.backend("hccl");
  parse_parallel_info();
  validate();
  rank_generator_ =
      std::make_unique<RankGenerator>(tp_.group_size(),
                                      sp_.group_size(),
                                      cfg_.group_size(),
                                      /*group_order=*/"tp-sp-cfg");
  get_group_by_type(tp_, "tp");
  get_group_by_type(sp_, "sp");
  get_group_by_type(cfg_, "cfg");
}

void DiTMappingNPU::parse_parallel_info() {
  if (options_.cfg_size() != -1) {
    cfg_.group_size(options_.cfg_size());
  }
  if (options_.tp_size() != -1) {
    tp_.group_size(options_.tp_size());
  }
  if (options_.sp_size() != -1) {
    sp_.group_size(options_.sp_size());
  }
}

void DiTMappingNPU::validate() {
  CHECK(cfg_.group_size() * tp_.group_size() * sp_.group_size() == world_size_)
      << "World size must equal to cfg_size * tp_size * sp_size. "
         "cfg_size is " +
             std::to_string(cfg_.group_size()) +
             ". "
             "tp_size is " +
             std::to_string(tp_.group_size()) +
             ". "
             "sp_size is " +
             std::to_string(sp_.group_size()) +
             ". "
             "world_size is " +
             std::to_string(world_size_) +
             ". "
             "Please check `cfg`, `tp`, `sp` and `world_size`.";

  CHECK(cfg_.group_size() <= 2) << "cfg_size must less than 2 "
                                   "cfg_size is " +
                                       std::to_string(cfg_.group_size()) +
                                       ". Please check `cfg` .";
}

void DiTMappingNPU::get_group_by_type(ParallelInfo& parallel_info,
                                      const std::string& group_type) {
  auto rank_per_group = rank_generator_->get_ranks(group_type);
  parallel_info.rank_per_group(rank_per_group);
  auto [current_group_id, local_rank] =
      get_current_group_id(rank_per_group, rank_);
  CHECK(current_group_id >= 0 && local_rank >= 0)
      << "Failed to get current group id : " << current_group_id
      << " local_rank " << local_rank;
  parallel_info.current_group_id(current_group_id);
  parallel_info.rank(local_rank);
}

std::tuple<int32_t, int32_t> DiTMappingNPU::get_current_group_id(
    const std::vector<std::vector<int32_t>>& rank_per_group,
    int32_t target_rank_id) {
  for (int32_t idx = 0; idx < rank_per_group.size(); ++idx) {
    const auto& group = rank_per_group[idx];
    auto it = std::find(group.begin(), group.end(), target_rank_id);
    if (it != group.end()) {
      return std::make_tuple(idx, std::distance(group.begin(), it));
    }
  }
  return std::make_tuple(-1, -1);
}

nlohmann::json DiTMappingNPU::to_json() {
  nlohmann::json data;

  data["SpSize"] = options_.sp_size();
  data["TpSize"] = options_.tp_size();
  data["CfgSize"] = options_.cfg_size();
  data["worldSize"] = world_size_;
  data["rank"] = rank_;
  data["sp"] = sp_.to_json();
  data["tp"] = tp_.to_json();
  data["cfg"] = cfg_.to_json();
  return data;
}

}  // namespace xllm
