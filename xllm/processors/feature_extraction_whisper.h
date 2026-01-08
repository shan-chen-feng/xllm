#pragma once
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include "core/framework/model_context.h"
#include "core/util/audio_utils.h"
#include "feature_extractor.h"

namespace xllm {

class WhisperFeatureExtractor : public FeatureExtractor {
 public:
  WhisperFeatureExtractor(const ModelArgs& args);

  bool process(const MMInput& mm_inputs, MMData& mm_datas);

  bool process_audio(const MMInput& mm_inputs,
                     MMData& mm_datas,
                     std::vector<torch::Tensor> raw_speech,
                     bool audio_in_video);

  void set_options(torch::TensorOptions& options) { options_ = options; }

 private:
  torch::Tensor torch_extract_fbank_features_(
      const torch::Tensor& waveform,
      const torch::TensorOptions& options);

  void zero_mean_unit_var_norm_(BatchFeature& batch_feature,
                                double padding_value = 0.0);

  int64_t n_fft_;
  int64_t hop_length_;
  int64_t chunk_length_;
  int64_t n_samples_;      // chunk_length_ * sampling_rate_
  int64_t nb_max_frames_;  // n_samples_ // hop_length_
  int64_t sampling_rate_;
  double dither_;
  torch::TensorOptions options_;
  torch::Tensor mel_filters_;

  bool truncation_ = false;
  std::optional<int64_t> pad_to_multiple_of_ = std::nullopt;
  bool return_attention_mask_ = false;
  PaddingStrategy padding_ = PaddingStrategy::MAX_LENGTH;
  std::optional<int64_t> max_length_ = std::nullopt;
  bool do_normalize_ = false;
  bool return_token_timestamps_ = false;
  bool use_audio_in_video_ = false;
};

}  // namespace xllm
