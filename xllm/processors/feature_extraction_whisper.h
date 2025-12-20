#pragma once
#include "feature_extractor.h"
#include "audio_utils.h" 
#include "core/framework/model_context.h"
#include <cmath>
#include <glog/logging.h>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <stdexcept>

namespace xllm {

class WhisperFeatureExtractor : public FeatureExtractor {
public:

    WhisperFeatureExtractor(const ModelArgs& args);
    
    // TODO: need to modify with MMDATA & MMINPUT    
    BatchFeature  process(
        const std::vector<torch::Tensor>& raw_speech,
        bool truncation = true,
        std::optional<int64_t> pad_to_multiple_of = std::nullopt,
        std::optional<bool> return_tensors = std::nullopt,
        std::optional<bool> return_attention_mask = std::nullopt,
        PaddingStrategy padding = PaddingStrategy::MAX_LENGTH,
        std::optional<int64_t> max_length = std::nullopt,
        std::optional<int64_t> sampling_rate = std::nullopt,
        std::optional<bool> do_normalize = std::nullopt,
        bool return_token_timestamps = false
    );

    void set_options(torch::TensorOptions& options){
        options_ = options;
    }

private:
    int64_t n_fft_;
    int64_t hop_length_;
    int64_t chunk_length_;
    int64_t n_samples_;           // chunk_length_ * sampling_rate_
    int64_t nb_max_frames_;       // n_samples_ // hop_length_
    int64_t sampling_rate_;
    double dither_;
    torch::TensorOptions options_;
    torch::Tensor mel_filters_;
    
    torch::Tensor torch_extract_fbank_features_(
        const torch::Tensor& waveform,
        const torch::TensorOptions& options
    );
    
    void zero_mean_unit_var_norm_(
        BatchFeature& batch_feature,
        double padding_value = 0.0
    );
    


};

}
