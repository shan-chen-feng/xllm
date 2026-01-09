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

#include "mposition.h"

#include <absl/strings/match.h>

#include "framework/model/model_args.h"
#include "framework/request/sequence.h"

namespace xllm {

namespace {
std::vector<std::tuple<std::string, int, int>> groupByTokenType(
    const std::vector<std::string>& token_types) {
  std::vector<std::tuple<std::string, int, int>> groups;
  if (token_types.empty()) return groups;

  std::string current_key = token_types[0];
  int start = 0;

  for (int i = 1; i < token_types.size(); ++i) {
    if (token_types[i] != current_key) {
      groups.emplace_back(current_key, start, i);
      current_key = token_types[i];
      start = i;
    }
  }
  groups.emplace_back(current_key, start, static_cast<int>(token_types.size()));
  return groups;
}
}  // namespace

torch::Tensor MPositionHelper::get_positions() {
  // if (seq_.is_chunked_prefill_stage()) {
  if (seq_.kv_state().kv_cache_tokens_num() < seq_.num_prompt_tokens()) {
    auto& mm_data = seq_.get_mm_data();

    torch::Tensor image_grid_thw;
    if (auto res = mm_data.get<torch::Tensor>("image_grid_thw"))
      image_grid_thw = res.value();

    torch::Tensor video_grid_thw;
    if (auto res = mm_data.get<torch::Tensor>("video_grid_thw"))
      video_grid_thw = res.value();

    torch::Tensor second_per_grid_ts;
    if (auto res = mm_data.get<torch::Tensor>("second_per_grid_ts"))
      second_per_grid_ts = res.value();

    torch::Tensor feat_length;
    if (auto res = mm_data.get<torch::Tensor>("feat_length"))
      feat_length = res.value();

    std::tuple<torch::Tensor, int> res;
    if (absl::StartsWith(args_.model_type(), "glm4v")) {
      res = get_positions_glm(image_grid_thw, video_grid_thw);
    } else if (absl::StartsWith(args_.model_type(), "qwen3_omni_moe")) {
      res = get_positions_omni(image_grid_thw, video_grid_thw, feat_length);
    } else {
      res = get_positions_p(image_grid_thw, video_grid_thw, second_per_grid_ts);
    }
    seq_.set_mrope_position_delta(std::get<1>(res));
    return std::get<0>(res);
  } else {
    return get_positions_d();
  }
}

#include <torch/torch.h>

#include <vector>

torch::Tensor get_llm_pos_ids_for_vision(int start_idx,
                                         int vision_idx,
                                         int spatial_merge_size,
                                         const torch::Tensor& t_index,
                                         const torch::Tensor& grid_hs,
                                         const torch::Tensor& grid_ws) {
  int llm_grid_h = grid_hs[vision_idx].item<int>() / spatial_merge_size;
  int llm_grid_w = grid_ws[vision_idx].item<int>() / spatial_merge_size;

  auto h_index = torch::arange(llm_grid_h, torch::kInt32)
                     .view({1, -1, 1})
                     .expand({t_index.size(0), -1, llm_grid_w})
                     .flatten();

  auto w_index = torch::arange(llm_grid_w, torch::kInt32)
                     .view({1, 1, -1})
                     .expand({t_index.size(0), llm_grid_h, -1})
                     .flatten();

  auto t_index_tensor = t_index.to(grid_hs.device())
                            .view({-1, 1})
                            .expand({-1, llm_grid_h * llm_grid_w})
                            .to(torch::kInt32)
                            .flatten();
  torch::TensorList tensor_list({t_index_tensor, h_index, w_index});
  return torch::stack(tensor_list) + start_idx;
}

std::tuple<torch::Tensor, int> MPositionHelper::get_positions_omni(
    torch::Tensor image_grid_thw,
    torch::Tensor video_grid_thw,
    torch::Tensor audio_seqlens) {
  bool use_audio_in_video = args_.mm_use_audio_in_video();
  auto input_tokens = seq_.tokens();
  auto spatial_merge_size = args_.mm_spatial_merge_size();
  auto image_token_id = args_.image_token_id();
  auto video_token_id = args_.video_token_id();
  auto audio_token_id = args_.audio_token_id();
  auto vision_start_token_id = args_.vision_start_token_id();
  auto video_start_token_id = args_.video_start_token_id();
  auto video_end_token_id = args_.video_end_token_id();
  auto audio_start_token_id = args_.audio_start_token_id();
  auto position_id_per_seconds = args_.mm_position_id_per_seconds();

  auto input_ids = torch::tensor(std::vector<int>(input_tokens));
  LOG(INFO) << "input_ids is";
  input_ids.print();
  // std::cout << input_ids;

  LOG(INFO) << "image thw and video thw";
  image_grid_thw.print();
  LOG(INFO) << "video thw";
  video_grid_thw.print();

  int seq_len = input_ids.size(0);
  LOG(INFO) << "after size";

  if (seq_len > 0 && (image_grid_thw.defined() || video_grid_thw.defined())) {
    torch::Tensor second_per_grids;
    if (video_grid_thw.defined()) {
      double fps = args_.mm_fps();
      int temporal_patch_size = args_.mm_temporal_patch_size();
      double second_per_grid_ts_value = temporal_patch_size / fps;
      second_per_grids =
          torch::ones({video_grid_thw.size(0)},
                      torch::TensorOptions().dtype(torch::kFloat32)) *
          second_per_grid_ts_value;
    }
    auto vision_start_mask = input_ids == vision_start_token_id;
    torch::Tensor vision_start_indices =
        torch::nonzero(vision_start_mask).squeeze(1);

    LOG(INFO) << "vision_start_indices is";
    std::cout << vision_start_indices;

    torch::Tensor vision_tokens;
    if (vision_start_indices.defined()) {
      vision_tokens = input_ids.index({vision_start_indices + 1});
    } else {
      vision_tokens =
          torch::empty({0}, torch::TensorOptions().dtype(torch::kLong));
    }

    int audio_nums = (input_ids == audio_start_token_id).sum().item<int>();
    int image_nums = (vision_tokens == image_token_id).sum().item<int>();
    int video_nums;

    if (use_audio_in_video) {
      video_nums = (vision_tokens == audio_start_token_id).sum().item<int>();
    } else {
      video_nums = (vision_tokens == video_token_id).sum().item<int>();
    }

    std::vector<torch::Tensor> llm_pos_ids_list;
    int st = 0;
    int image_idx = 0;
    int video_idx = 0;
    int audio_idx = 0;
    int remain_images = image_nums;
    int remain_videos = video_nums;
    int remain_audios = audio_nums;

    int multimodal_nums;
    if (use_audio_in_video) {
      multimodal_nums = image_nums + audio_nums;
    } else {
      multimodal_nums = image_nums + video_nums + audio_nums;
    }

    auto input_tokens_vec = std::vector<int>(input_tokens);
    LOG(INFO) << 1;
    for (int i = 0; i < multimodal_nums; ++i) {
      int st_idx = llm_pos_ids_list.empty()
                       ? 0
                       : llm_pos_ids_list.back().max().item<int>() + 1;

      int ed_vision_start = seq_len + 1;
      int ed_audio_start = seq_len + 1;
      LOG(INFO) << ed_vision_start;
      LOG(INFO) << ed_audio_start;
      if (remain_videos > 0 || remain_images > 0) {
        auto it = std::find(input_tokens_vec.begin() + st,
                            input_tokens_vec.end(),
                            vision_start_token_id);
        if (it != input_tokens_vec.end()) {
          ed_vision_start = std::distance(input_tokens_vec.begin(), it);
        }
      }

      LOG(INFO) << 2;
      if (audio_token_id == *std::find(input_tokens_vec.begin() + st,
                                       input_tokens_vec.end(),
                                       audio_token_id) &&
          remain_audios > 0) {
        auto it = std::find(input_tokens_vec.begin() + st,
                            input_tokens_vec.end(),
                            audio_start_token_id);
        if (it != input_tokens_vec.end()) {
          ed_audio_start = std::distance(input_tokens_vec.begin(), it);
        }
      }
      LOG(INFO) << 3;
      LOG(INFO) << ed_vision_start;
      LOG(INFO) << ed_audio_start;
      int min_ed = std::min(ed_vision_start, ed_audio_start);
      int text_len = min_ed - st;
      LOG(INFO) << "text_len";
      LOG(INFO) << min_ed;
      LOG(INFO) << st;

      if (text_len != 0) {
        auto text_pos_ids = torch::arange(text_len, torch::kInt32)
                                .view({1, -1})
                                .expand({3, -1}) +
                            st_idx;
        llm_pos_ids_list.push_back(text_pos_ids);
        st_idx += text_len;
      }
      LOG(INFO) << "3.1";
      int bos_len, eos_len;
      if (min_ed == ed_vision_start && ed_vision_start + 1 == ed_audio_start) {
        bos_len = 2;
        eos_len = 2;
      } else {
        bos_len = 1;
        eos_len = 1;
      }

      auto bos_pos_ids =
          torch::arange(bos_len, torch::kInt32).view({1, -1}).expand({3, -1}) +
          st_idx;
      LOG(INFO) << "3.2";
      llm_pos_ids_list.push_back(bos_pos_ids);
      st_idx += bos_len;
      LOG(INFO) << 4;
      if (min_ed == ed_audio_start) {
        auto audio_len =
            get_feat_extract_output_lengths(audio_seqlens[audio_idx]);
        auto audio_pos_ids = torch::arange(audio_len.item<int>(), torch::kInt32)
                                 .view({1, -1})
                                 .expand({3, -1}) +
                             st_idx;
        llm_pos_ids_list.push_back(audio_pos_ids);

        st += text_len + bos_len + audio_len.item<int>() + eos_len;
        audio_idx++;
        remain_audios--;
        LOG(INFO) << 5;
      } else if (min_ed == ed_vision_start &&
                 input_tokens_vec[ed_vision_start + 1] == image_token_id) {
        int grid_t = image_grid_thw[image_idx][0].item<int>();
        auto grid_hs = image_grid_thw.index({torch::indexing::Slice(), 1});
        auto grid_ws = image_grid_thw.index({torch::indexing::Slice(), 2});

        auto t_index =
            torch::arange(grid_t, torch::kInt32) * position_id_per_seconds;
        LOG(INFO) << 6;
        auto image_pos_ids = get_llm_pos_ids_for_vision(
            st_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws);

        int image_len = image_grid_thw[image_idx].prod().item<int>() /
                        (spatial_merge_size * spatial_merge_size);
        llm_pos_ids_list.push_back(image_pos_ids);

        st += text_len + bos_len + image_len + eos_len;
        image_idx++;
        remain_images--;
        LOG(INFO) << 7;
      } else if (min_ed == ed_vision_start &&
                 input_tokens_vec[ed_vision_start + 1] == video_token_id) {
        int grid_t = video_grid_thw[video_idx][0].item<int>();
        auto grid_hs = video_grid_thw.index({torch::indexing::Slice(), 1});
        auto grid_ws = video_grid_thw.index({torch::indexing::Slice(), 2});

        auto t_index = (torch::arange(grid_t, torch::kInt32) *
                        second_per_grids[video_idx].item<float>() *
                        position_id_per_seconds)
                           .to(torch::kInt32);
        LOG(INFO) << 8;
        auto video_pos_ids = get_llm_pos_ids_for_vision(
            st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws);

        int video_len = video_grid_thw[video_idx].prod().item<int>() /
                        (spatial_merge_size * spatial_merge_size);
        llm_pos_ids_list.push_back(video_pos_ids);

        st += text_len + bos_len + video_len + eos_len;
        video_idx++;
        remain_videos--;
        LOG(INFO) << 9;
      } else if (min_ed == ed_vision_start &&
                 ed_vision_start + 1 == ed_audio_start) {
        auto audio_len =
            get_feat_extract_output_lengths(audio_seqlens[audio_idx]);
        auto audio_pos_ids = torch::arange(audio_len.item<int>(), torch::kInt32)
                                 .view({1, -1})
                                 .expand({3, -1}) +
                             st_idx;

        int grid_t = video_grid_thw[video_idx][0].item<int>();
        auto grid_hs = video_grid_thw.index({torch::indexing::Slice(), 1});
        auto grid_ws = video_grid_thw.index({torch::indexing::Slice(), 2});
        LOG(INFO) << 10;
        auto t_index = (torch::arange(grid_t, torch::kInt32) *
                        second_per_grids[video_idx].item<float>() *
                        position_id_per_seconds)
                           .to(torch::kInt32);

        auto video_pos_ids = get_llm_pos_ids_for_vision(
            st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws);

        int video_idx_pos = 0;
        int audio_idx_pos = 0;

        while (video_idx_pos < video_pos_ids.size(1) &&
               audio_idx_pos < audio_pos_ids.size(1)) {
          if (video_pos_ids[0][video_idx_pos].item<float>() <=
              audio_pos_ids[0][audio_idx_pos].item<float>()) {
            llm_pos_ids_list.push_back(video_pos_ids.index(
                {torch::indexing::Slice(),
                 torch::indexing::Slice(video_idx_pos, video_idx_pos + 1)}));
            video_idx_pos++;
          } else {
            llm_pos_ids_list.push_back(audio_pos_ids.index(
                {torch::indexing::Slice(),
                 torch::indexing::Slice(audio_idx_pos, audio_idx_pos + 1)}));
            audio_idx_pos++;
          }
        }
        LOG(INFO) << 11;
        if (video_idx_pos < video_pos_ids.size(1)) {
          llm_pos_ids_list.push_back(video_pos_ids.index(
              {torch::indexing::Slice(),
               torch::indexing::Slice(video_idx_pos, video_pos_ids.size(1))}));
        }
        if (audio_idx_pos < audio_pos_ids.size(1)) {
          llm_pos_ids_list.push_back(audio_pos_ids.index(
              {torch::indexing::Slice(),
               torch::indexing::Slice(audio_idx_pos, audio_pos_ids.size(1))}));
        }
        LOG(INFO) << 12;

        int video_len = video_grid_thw[video_idx].prod().item<int>() /
                        (spatial_merge_size * spatial_merge_size);

        st += text_len + bos_len + audio_len.item<int>() + video_len + eos_len;
        audio_idx++;
        video_idx++;
        remain_videos--;
        remain_audios--;
      }
      LOG(INFO) << 13;
      st_idx = llm_pos_ids_list.empty()
                   ? 0
                   : llm_pos_ids_list.back().max().item<int>() + 1;

      auto eos_pos_ids =
          torch::arange(eos_len, torch::kInt32).view({1, -1}).expand({3, -1}) +
          st_idx;
      llm_pos_ids_list.push_back(eos_pos_ids);
    }
    LOG(INFO) << 14;
    if (st < input_tokens_vec.size()) {
      int st_idx = llm_pos_ids_list.empty()
                       ? 0
                       : llm_pos_ids_list.back().max().item<int>() + 1;

      int text_len = input_tokens.size() - st;
      LOG(INFO) << 15;
      auto text_pos_ids =
          torch::arange(text_len, torch::kInt32).view({1, -1}).expand({3, -1}) +
          st_idx;
      llm_pos_ids_list.push_back(text_pos_ids);
    }
    LOG(INFO) << 16;

    torch::Tensor llm_positions = torch::cat(llm_pos_ids_list, 1);
    LOG(INFO) << 17;
    int mrope_position_delta =
        llm_positions.max().item<int>() + 1 - input_tokens_vec.size();
    LOG(INFO) << 18;
    torch::save(llm_positions, "llm_positions.pt");
    LOG(INFO) << "mrope_position_delta is";
    LOG(INFO) << mrope_position_delta;
    return {llm_positions, mrope_position_delta};

  } else {
    LOG(INFO) << "inside else";
    auto text_pos_ids = torch::arange(input_ids.size(0), torch::kInt32)
                            .view({1, -1})
                            .expand({3, -1});
    auto max_position_ids =
        std::get<0>(text_pos_ids.max(0, false)).max(-1, true);
    std::cout << std::get<0>(max_position_ids);
    auto mrope_position_deltas =
        std::get<0>(max_position_ids).item<int>() + 1 - input_ids.size(0);

    return std::make_tuple(text_pos_ids, mrope_position_deltas);
  }
}

std::tuple<torch::Tensor, int> MPositionHelper::get_positions_glm(
    torch::Tensor image_grid_thw,
    torch::Tensor video_grid_thw) {
  auto input_tokens = seq_.tokens();
  auto spatial_merge_size = args_.mm_spatial_merge_size();
  auto image_token_id = args_.image_token_id();
  auto video_token_id = args_.video_token_id();
  auto video_start_token_id = args_.video_start_token_id();
  auto video_end_token_id = args_.video_end_token_id();

  auto dtype = torch::kInt32;

  std::vector<std::string> input_token_type;
  bool in_video = false;
  int num_tokens = input_tokens.size();

  for (int index = 0; index < num_tokens; ++index) {
    auto token = input_tokens[index];
    if (token == video_start_token_id) {
      in_video = true;
    } else if (token == video_end_token_id) {
      in_video = false;
    }

    if (token == image_token_id && !in_video) {
      input_token_type.push_back("image");
    } else if (token == image_token_id && in_video) {
      input_token_type.push_back("video");
    } else {
      input_token_type.push_back("text");
    }
  }
  auto input_type_group = groupByTokenType(input_token_type);
  int image_index = 0;
  int video_index = 0;
  int video_group_index = 0;

  std::vector<torch::Tensor> llm_pos_ids_list;
  int video_frame_num = 1;
  for (const auto& group : input_type_group) {
    const auto& modality_type = std::get<0>(group);
    int start_idx = std::get<1>(group);
    int end_idx = std::get<2>(group);
    int st_idx = 0;
    if (!llm_pos_ids_list.empty()) {
      st_idx = llm_pos_ids_list.back().max().item<int>() + 1;
    }

    if (modality_type == "image") {
      auto grid = image_grid_thw[image_index];
      int t = grid[0].item<int>();
      int h = grid[1].item<int>() / spatial_merge_size;
      int w = grid[2].item<int>() / spatial_merge_size;

      auto t_arange =
          torch::arange(t, dtype).view({-1, 1}).expand({-1, h * w}).flatten();
      auto h_arange =
          torch::arange(h, dtype).view({1, -1, 1}).expand({t, -1, w}).flatten();
      auto w_arange =
          torch::arange(w, dtype).view({1, 1, -1}).expand({t, h, -1}).flatten();

      auto pos = torch::stack({t_arange, h_arange, w_arange}) + st_idx;
      llm_pos_ids_list.push_back(pos);
      video_frame_num = 1;
      image_index++;
    } else if (modality_type == "video") {
      int t = video_frame_num;
      int h = video_grid_thw[video_index][1].item<int>() / spatial_merge_size;
      int w = video_grid_thw[video_index][2].item<int>() / spatial_merge_size;

      for (int t_idx = 0; t_idx < t; ++t_idx) {
        auto t_tensor = torch::full({1, h * w}, t_idx, dtype).flatten();
        auto h_tensor = torch::arange(h, dtype)
                            .view({1, -1, 1})
                            .expand({1, -1, w})
                            .flatten();
        auto w_tensor = torch::arange(w, dtype)
                            .view({1, 1, -1})
                            .expand({1, h, -1})
                            .flatten();

        auto pos = torch::stack({t_tensor, h_tensor, w_tensor}) + st_idx;
        llm_pos_ids_list.push_back(pos);
      }

      video_group_index++;
      if (video_group_index >= video_grid_thw[video_index][0].item<int>()) {
        video_index++;
        video_group_index = 0;
      }
      video_frame_num++;
    } else {  // text
      int text_len = end_idx - start_idx;
      auto arange =
          torch::arange(text_len, dtype).view({1, -1}).expand({3, -1}) + st_idx;
      llm_pos_ids_list.push_back(arange);
      video_frame_num = 1;
    }
  }

  torch::Tensor llm_positions =
      torch::cat(llm_pos_ids_list, /*dim=*/1).reshape({3, -1});
  llm_positions = llm_positions;
  int mrope_position_delta =
      (llm_positions.max().item<int>() + 1 - input_tokens.size());

  return std::make_pair(llm_positions, mrope_position_delta);
}

std::tuple<torch::Tensor, int> MPositionHelper::get_positions_p(
    torch::Tensor image_grid_thw,
    torch::Tensor video_grid_thw,
    torch::Tensor second_per_grid_ts) {
  auto image_token_id = args_.image_token_id();
  auto video_token_id = args_.video_token_id();
  auto vision_start_token_id = args_.vision_start_token_id();
  auto spatial_merge_size = args_.mm_spatial_merge_size();
  auto tokens_per_second = args_.mm_tokens_per_second();

  auto input_tokens = seq_.tokens();
  auto input_tokens_tensor = torch::tensor(std::vector<int>(input_tokens));
  auto vision_start_indices =
      torch::argwhere(input_tokens_tensor == vision_start_token_id).squeeze(1);

  auto vision_tokens = input_tokens_tensor.index({vision_start_indices + 1});
  int image_nums = torch::sum(vision_tokens == image_token_id).item<int>();
  int video_nums = torch::sum(vision_tokens == video_token_id).item<int>();

  std::vector<torch::Tensor> llm_pos_ids_list;
  int st = 0;
  int remain_images = image_nums, remain_videos = video_nums;
  int image_index = 0, video_index = 0;

  for (int i = 0; i < image_nums + video_nums; ++i) {
    float video_second_per_grid_t = 1.0f;
    int ed_image = input_tokens.size() + 1;
    int ed_video = input_tokens.size() + 1;

    if (remain_images > 0) {
      auto it = std::find(
          input_tokens.begin() + st, input_tokens.end(), image_token_id);
      if (it != input_tokens.end()) {
        ed_image = std::distance(input_tokens.begin(), it);
      }
    }

    if (remain_videos > 0) {
      auto it = std::find(
          input_tokens.begin() + st, input_tokens.end(), video_token_id);
      if (it != input_tokens.end()) {
        ed_video = std::distance(input_tokens.begin(), it);
      }
    }

    int t = 0, h = 0, w = 0;
    int ed = 0;
    if (ed_image < ed_video) {
      t = image_grid_thw[image_index][0].item<int>();
      h = image_grid_thw[image_index][1].item<int>();
      w = image_grid_thw[image_index][2].item<int>();
      image_index++;
      remain_images--;
      ed = ed_image;
    } else {
      t = video_grid_thw[video_index][0].item<int>();
      h = video_grid_thw[video_index][1].item<int>();
      w = video_grid_thw[video_index][2].item<int>();

      video_second_per_grid_t = second_per_grid_ts[video_index].item<float>();

      video_index++;
      remain_videos--;
      ed = ed_video;
    }

    int llm_grid_t = t;
    int llm_grid_h = h / spatial_merge_size;
    int llm_grid_w = w / spatial_merge_size;
    int text_len = ed - st;

    int st_idx = 0;
    if (!llm_pos_ids_list.empty()) {
      st_idx = llm_pos_ids_list.back().max().item<int>() + 1;
    }

    if (text_len > 0) {
      auto text_pos =
          torch::arange(text_len, torch::kInt32).view({1, -1}).expand({3, -1}) +
          st_idx;
      llm_pos_ids_list.push_back(text_pos);
    }

    auto t_index = (torch::arange(llm_grid_t, torch::kInt32)
                        .view({-1, 1})
                        .expand({-1, llm_grid_h * llm_grid_w}) *
                    video_second_per_grid_t * tokens_per_second)
                       .to(torch::kInt32)
                       .flatten();

    auto h_index = torch::arange(llm_grid_h, torch::kInt32)
                       .view({1, -1, 1})
                       .expand({llm_grid_t, -1, llm_grid_w})
                       .flatten();

    auto w_index = torch::arange(llm_grid_w, torch::kInt32)
                       .view({1, 1, -1})
                       .expand({llm_grid_t, llm_grid_h, -1})
                       .flatten();

    auto visual_pos =
        torch::stack({t_index, h_index, w_index}) + text_len + st_idx;
    llm_pos_ids_list.push_back(visual_pos);

    st = ed + llm_grid_t * llm_grid_h * llm_grid_w;
  }

  if (st < static_cast<int>(input_tokens.size())) {
    int st_idx = 0;
    if (!llm_pos_ids_list.empty()) {
      st_idx = llm_pos_ids_list.back().max().item<int>() + 1;
    }

    int text_len = input_tokens.size() - st;
    auto text_pos =
        torch::arange(text_len, torch::kInt32).view({1, -1}).expand({3, -1}) +
        st_idx;
    llm_pos_ids_list.push_back(text_pos);
  }

  auto llm_positions = torch::cat(llm_pos_ids_list, 1).reshape({3, -1});
  int mrope_position_delta =
      (llm_positions.max().item<int>() + 1 - input_tokens.size());
  return std::make_tuple(llm_positions, mrope_position_delta);
}

torch::Tensor MPositionHelper::get_positions_d() {
  auto mrope_position_delta = seq_.get_mrope_position_delta();
  auto num_tokens = seq_.num_tokens();
  return torch::arange(int(mrope_position_delta + num_tokens - 1),
                       int(mrope_position_delta + num_tokens),
                       torch::kInt32)
      .expand({3, -1});
}

}  // namespace xllm
