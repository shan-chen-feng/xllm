#pragma once

#include "qwen3_omni_moe_thinker.h" 
#include "qwen3_omni_moe_talker.h"
#include "models/llm/npu/qwen3_omni_code2wav.h"

namespace xllm {

class Qwen3_Omni_Moe_InputProcessor : public InputProcessor {
  enum class TokenType {
    INVALID,
    IMAGE,
    VIDEO,
    AUDIO
  };
 
  struct FeatureIndexRef {
    int* index_ = 0;
    const std::string* token_ = nullptr;
    const torch::Tensor* feat_size_ = nullptr;
    
    FeatureIndexRef() = default;
     
    FeatureIndexRef(int* index,
                    const std::string* token,
                    const torch::Tensor* feat_size
                    ) : index_(index),
                        token_(token),
                        feat_size_(feat_size){}
  };

 public:
  Qwen3_Omni_Moe_InputProcessor(const ModelArgs& args) {
    merge_size_ = args.mm_image_merge_size();
    fps_ =  args.mm_fps();
    temporal_patch_size_  = args.mm_temporal_patch_size();
    use_audio_in_video_ = args.mm_use_audio_in_video();
    video_second_per_grid_ =  temporal_patch_size_ / fps_;
  }

  void process(std::string& prompt, const MMData& mm_data) override {
    prompt = "<|im_start|>user\please describe the audio<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant";
    prompt = "<|im_start|>user\nplease describe the audio<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|video_pad|><|vision_end|><|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n";
    LOG(INFO) << prompt;
    torch::Tensor image_grid_thw;
    if (auto res = mm_data.get<torch::Tensor>("image_grid_thw"))
      image_grid_thw = res.value();

    torch::Tensor video_grid_thw;
    if (auto res = mm_data.get<torch::Tensor>("video_grid_thw"))
      video_grid_thw = res.value();

    torch::Tensor feature_attention_mask;
    if (auto res = mm_data.get<torch::Tensor>("attention_mask"))
      feature_attention_mask = res.value();
    
    torch::Tensor feat_length;
    if (feature_attention_mask.defined())
      feat_length = get_feat_extract_output_lengths(feature_attention_mask.sum(-1));

    if (!image_grid_thw.defined() && !video_grid_thw.defined()) return;

    auto merge_length = merge_size_ * merge_size_;

    int total_audio_token = 0;
    if (feat_length.defined()) {
      auto count = feat_length.sizes()[0];
      for (int idx = 0; idx < count; ++idx)
        total_audio_token +=
            feat_length[idx].item<int>();
    }

    int total_image_token = 0;
    if (image_grid_thw.defined()) {
      auto count = image_grid_thw.sizes()[0];
      for (int idx = 0; idx < count; ++idx)
        total_image_token +=
            image_grid_thw[idx].prod().item<int>() / merge_length;
    }

    int total_video_token = 0;
    if (video_grid_thw.defined()) {
      auto count = video_grid_thw.sizes()[0];
      for (int idx = 0; idx < count; ++idx) {
        std::cout << video_grid_thw[idx];
        LOG(INFO) << video_grid_thw[idx].prod().item<int>();
        total_video_token +=
            video_grid_thw[idx].prod().item<int>() / merge_length;
        LOG(INFO) << merge_length;
        LOG(INFO) << total_video_token;}
    }

    size_t total_token_len = total_image_token * image_token_.size() +
                             total_video_token * video_token_.size() +
                             total_audio_token * audio_token_.size();
    std::string data;
    data.reserve(prompt.size() + total_token_len);

    int image_index = 0;
    int video_index = 0;
    int audio_index = 0;
  

    int* index = 0;
    const std::string* token = nullptr;
    const torch::Tensor* feat_size = nullptr;
    std::unordered_map<TokenType, FeatureIndexRef> feature_index_map = {
        {TokenType::AUDIO, FeatureIndexRef(&audio_index, &audio_token_, &feat_length)},
        {TokenType::IMAGE, FeatureIndexRef(&image_index, &image_token_, &image_grid_thw)},
        {TokenType::VIDEO, FeatureIndexRef(&video_index, &video_token_, &video_grid_thw)}
    };

    size_t begin = 0;
    auto pair = find_special_token(prompt, begin);
    FLAGS_max_log_size = 1000000;
    LOG(INFO) << pair.second;
    while (pair.second != std::string::npos) {
      data.append(prompt, begin, pair.second - begin);
      
      auto feature_index = feature_index_map[pair.first];
      feat_size = feature_index.feat_size_;
      token = feature_index.token_;
      index = feature_index.index_;
      LOG(INFO) << token;
      if (pair.first == TokenType::AUDIO) {
        // for audio
        auto token_num = (*feat_size)[(*index)].item<int>();
        while (token_num--) data.append(*token);
      } else if (pair.first == TokenType::VIDEO && use_audio_in_video_){
        // for audio in video
        auto audio_feature_index = feature_index_map[TokenType::AUDIO];
        auto audio_feat_size = audio_feature_index.feat_size_;
        auto audio_token = audio_feature_index.token_;
        auto audio_index = audio_feature_index.index_;

        auto audio_token_indices = torch::arange((*audio_feat_size)[(*audio_index)].item<int>());

        int T = feat_size[0].item<int>();
        int H = feat_size[1].item<int>();
        int W = feat_size[2].item<int>();
        
        int height = H / merge_size_;
        int width = W / merge_size_;
       
        auto video_token_indices_1d = torch::arange(T);
        auto video_token_indices = video_token_indices_1d.view({T, 1, 1});

        video_token_indices = video_token_indices.expand({T, height, width});

        video_token_indices = video_token_indices.reshape({-1});
        video_token_indices = video_token_indices * video_second_per_grid_ * position_id_per_seconds_;

        auto video_indices_vec = video_token_indices.accessor<double, 1>();
        auto audio_indices_vec = audio_token_indices.accessor<int, 1>();

        std::string placeholder_string = audio_bos_token_;

        size_t video_data_index = 0;
        size_t audio_data_index = 0;
        size_t video_len = video_indices_vec.size(0);
        size_t audio_len = audio_indices_vec.size(0);

        while (video_data_index < video_len && audio_data_index < audio_len) {
           if (video_indices_vec[video_data_index] <= audio_indices_vec[audio_data_index]) {
               placeholder_string.append(*token);                
               video_data_index++;
           } else {
               placeholder_string.append(*audio_token);
               audio_data_index++;
           }
        }

        if (video_data_index < video_len) {
            size_t remaining_video = video_len - video_data_index;
            for (size_t i = 0; i < remaining_video; ++i) {
                placeholder_string.append(*token);
            }
         }

        if (audio_data_index < audio_len) {
            size_t remaining_audio = audio_len - audio_data_index;
            for (size_t i = 0; i < remaining_audio; ++i) {
                placeholder_string.append(*audio_token);
            }
        }

        placeholder_string.append(audio_eos_token_);
        data.append(placeholder_string);

      } else {
        // for image and video
        auto token_num = (*feat_size)[(*index)].prod().item<int>() / merge_length;
        while (token_num--) data.append(*token);

      }
      ++(*index);
      begin = pair.second + token->size();
      pair = find_special_token(prompt, begin);
    }

    if (begin < prompt.size()) data.append(prompt, begin, std::string::npos);
    //LOG(INFO) << data;
    std::streambuf* buf = std::cout.rdbuf();
    
    char buffer[8192];
    std::cout.rdbuf()->pubsetbuf(buffer, sizeof(buffer));
    std::cout << data;
    prompt = std::move(data);
  }

 private:
  std::pair<TokenType, size_t> find_special_token(const std::string& prompt,
                                                 size_t begin) {
    struct TokenInfo {
      const std::string& token;
      TokenType type;
      size_t pos;
    };
    
    std::vector<TokenInfo> tokens = {
      {image_token_, TokenType::IMAGE, std::string::npos},
      {video_token_, TokenType::VIDEO, std::string::npos},
      {audio_token_, TokenType::AUDIO, std::string::npos}
    };
    
    for (auto& token_info : tokens) {
      token_info.pos = prompt.find(token_info.token, begin);
    }
    
    auto earliest = std::min_element(tokens.begin(), tokens.end(),
      [](const TokenInfo& a, const TokenInfo& b) {
        if (a.pos == std::string::npos) return false;
        if (b.pos == std::string::npos) return true;
        return a.pos < b.pos;
      });
    
    if (earliest == tokens.end() || earliest->pos == std::string::npos) {
      return {TokenType::INVALID, std::string::npos};
    }
    
    return {earliest->type, earliest->pos};
  }

 private:
  const std::string image_token_ = "<|image_pad|>";
  const std::string video_token_ = "<|video_pad|>";
  const std::string audio_token_ = "<|audio_pad|>";
  const std::string vision_bos_token_ = "<|vision_start|>";
  const std::string vision_end_token_ = "<|vision_end|>";
  const std::string audio_bos_token_ = "<|audio_start|>";
  const std::string audio_eos_token_ = "<|audio_end|>";

  
  int merge_size_ = 0; 
  int position_id_per_seconds_ = 13;
  double fps_ = 1.0;
  int temporal_patch_size_ = 2.0;
  bool use_audio_in_video_ = false;
  double video_second_per_grid_ = 0.0;
};

class Qwen3_Omni_Moe_ForConditionalGenerationImpl : public torch::nn::Module {
 public:
  Qwen3_Omni_Moe_ForConditionalGenerationImpl(const ModelContext& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()) {
    thinker_ = register_module("thinker", Qwen3_Omni_Moe_Thinker_ForConditionalGeneration(context));
    //talker_ = register_module("talker", Qwen3_Omni_MoeTalkerForConditionalGeneration(context));
  }


  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelInputParams& input_params) {
    LOG(INFO) << "inside forwards +++++++=";
    torch::NoGradGuard no_grad;
    auto emb = thinker_(tokens, positions, kv_caches, input_params);

    return emb;
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    return thinker_->logits(hidden_states, seleted_idxes);
  }

  void load_model(std::shared_ptr<ModelLoader> loader) {
    LOG(INFO) << "start load thinker model";
    //talker_->load_model(loader);
    thinker_->load_model(std::move(loader));
  }

  layer::LmHead get_lm_head() { return thinker_->get_lm_head(); }
  void set_lm_head(layer::LmHead& head) { thinker_->set_lm_head(head); }
  

  layer::WordEmbedding get_word_embedding() {
    return thinker_->get_word_embedding();
  }

  void set_word_embedding(layer::WordEmbedding& word_embedding) {
    thinker_->set_word_embedding(word_embedding);
  }

 private:
  ModelArgs model_args_;
  torch::TensorOptions options_;
  Qwen3_Omni_Moe_Thinker_ForConditionalGeneration thinker_{nullptr};
  Qwen3_Omni_MoeTalkerForConditionalGeneration talker_{nullptr};
  
};
TORCH_MODULE(Qwen3_Omni_Moe_ForConditionalGeneration);


REGISTER_INPUT_PROCESSOR(qwen3_omni_moe, Qwen3_Omni_Moe_InputProcessor);
REGISTER_CAUSAL_VLM_MODEL(qwen3_omni_moe, Qwen3_Omni_Moe_ForConditionalGeneration);
REGISTER_IMAGE_PROCESSOR(qwen3_omni_moe, Qwen2VLImageProcessor);
REGISTER_FEATURE_EXTRACTOR(qwen3_omni_moe, WhisperFeatureExtractor);

REGISTER_MODEL_ARGS(qwen3_omni_moe, [&] {
  LOAD_ARG_OR(model_type, "model_type", "qwen3_omni_moe");
  
  // feature extractor processor parameters 
  // ***
  // TODO: these parameters should be placed in hf_model_loader.cpp
  LOAD_ARG_OR(has_feature_extractor, "has_feature_extractor", true);
  LOAD_ARG_OR(mm_audio_truncation, "truncation", false);
  LOAD_ARG_OR(mm_audio_padding_strategy, "padding_strategy", 1);
  LOAD_ARG_OR(mm_audio_max_length, "max_length", -1);
  LOAD_ARG_OR(mm_audio_pad_to_multiple_of, "pad_to_multiple_of", -1);
  LOAD_ARG_OR(mm_audio_do_normalize, "do_normalize", false);
  LOAD_ARG_OR(mm_audio_return_token_timestamps, "return_token_timestamps", false);  
  LOAD_ARG_OR(mm_audio_return_attention_mask, "return_attention_mask", true);
  LOAD_ARG_OR(mm_use_audio_in_video, "use_audio_in_video", false);
  LOAD_ARG_OR(mm_position_id_per_seconds, "position_id_per_seconds", 13);
  LOAD_ARG_OR(mm_fps, "fps", 1.0);
  LOAD_ARG_OR(mm_temporal_patch_size, "temporal_patch_size", 2);
  /*
  *******
  rename the parms,
  or the thinker will core at moe decoder
  *******
  // Talker basic
  LOAD_ARG_WITH_PREFIX_JSON("talker_config.text_config", [&]{
      LOAD_ARG_OR_PREFIX(n_layers, "num_hidden_layers", 20);
      LOAD_ARG_PREFIX(rope_scaling_mrope_section, "rope_scaling.mrope_section");
      LOAD_ARG_OR_PREFIX(max_position_embeddings, "max_position_embeddings",65536);
      LOAD_ARG_OR_PREFIX(rope_theta, "rope_theta",1000000);
      // Attention
      LOAD_ARG_OR_PREFIX(head_dim, "head_dim", 128);
      LOAD_ARG_OR_PREFIX(n_kv_heads, "num_key_value_heads", 2);
      LOAD_ARG_OR_PREFIX(n_heads, "num_attention_heads", 16);
      // MLP
      LOAD_ARG_OR_PREFIX(n_shared_experts, "num_shared_experts", 1);
      LOAD_ARG_OR_PREFIX(num_experts, "num_experts", 128);
      LOAD_ARG_OR_PREFIX(num_experts_per_tok, "num_experts_per_tok", 6);
      // code 
      LOAD_ARG_OR_PREFIX(talker_text_hidden_size, "hidden_size", 6);
      LOAD_ARG_OR_PREFIX(talker_text_vocab_size, "vocab_size", 6);
      // resize MLP
      LOAD_ARG_OR_PREFIX(talker_text_intermediate_size, "intermediate_size", 2048);
      LOAD_ARG_OR_PREFIX(talker_text_hidden_size, "hidden_size", 1024);
   });

  LOAD_ARG_OR(thinker_hidden_size, "talker_config.thinker_hidden_size", 2048);
  */
  // thinker config 
  LOAD_ARG_WITH_PREFIX_JSON("thinker_config", [&]{
      LOAD_ARG_OR_PREFIX(
          vision_start_token_id, "vision_start_token_id", 151652);
      LOAD_ARG_OR_PREFIX(
          vision_end_token_id, "vision_end_token_id", 151653);
      LOAD_ARG_OR_PREFIX(vision_token_id, "vision_token_id", 151654);
      LOAD_ARG_OR_PREFIX(image_token_id, "image_token_id", 151655);
      LOAD_ARG_OR_PREFIX(video_token_id, "video_token_id", 151656);
      LOAD_ARG_OR_PREFIX(audio_token_id, "audio_token_id", 151675);
      LOAD_ARG_OR_PREFIX(dtype, "dtype", "bfloat16"); 
   });
  
  // thinker.text_config
  LOAD_ARG_WITH_PREFIX_JSON("thinker_config.text_config", [&]{
      LOAD_ARG_OR_PREFIX(attention_bias, "attention_bias", false);
      LOAD_ARG_OR_PREFIX(attention_dropout, "attention_dropout", 0.0);
      LOAD_ARG_OR_PREFIX(decoder_sparse_step, "decoder_sparse_step", 1);

      LOAD_ARG_OR_PREFIX(bos_token_id, "bos_token_id", 151643);
      LOAD_ARG_OR_PREFIX(eos_token_id, "eos_token_id", 151645);
      LOAD_ARG_OR_PREFIX(hidden_act, "hidden_act", "silu");
      LOAD_ARG_OR_PREFIX(hidden_size, "hidden_size", 2048);
      LOAD_ARG_OR_PREFIX(intermediate_size, "intermediate_size", 768);
      LOAD_ARG_OR_PREFIX(
          max_position_embeddings, "max_position_embeddings", 65536);
      LOAD_ARG_OR_PREFIX(max_window_layers, "max_window_layers", 28);
      LOAD_ARG_OR_PREFIX(n_heads, "num_attention_heads", 32);
      LOAD_ARG_OR_PREFIX(n_layers, "num_hidden_layers", 48);
      LOAD_ARG_OR_PREFIX(n_kv_heads, "num_key_value_heads", 4);
      LOAD_ARG_OR_PREFIX(rms_norm_eps, "rms_norm_eps", 1e-06);
      LOAD_ARG_OR_PREFIX(sliding_window, "sliding_window", 32768);
      LOAD_ARG_OR_PREFIX(tie_word_embeddings, "tie_word_embeddings", false);
      LOAD_ARG_PREFIX(rope_scaling_mrope_section,
               "rope_scaling.mrope_section");
      LOAD_ARG_OR_PREFIX(initializer_range, "initializer_range", 0.02);
      LOAD_ARG_OR_PREFIX(use_sliding_window, "use_sliding_window", false);
      LOAD_ARG_OR_PREFIX(moe_intermediate_size, "moe_intermediate_size", 768);
      LOAD_ARG_OR_PREFIX(norm_topk_prob, "norm_topk_prob", true);
      LOAD_ARG_OR_PREFIX(num_experts, "num_experts", 128);
      LOAD_ARG_OR_PREFIX(num_experts_per_tok, "num_experts_per_tok", 8);
      LOAD_ARG_OR_FUNC_PREFIX(head_dim, "head_dim", [&] {
        return args->hidden_size() / args->n_heads();
      });
      LOAD_ARG_OR_PREFIX(
         rope_scaling_rope_type, "rope_scaling.type", "mrope");
      LOAD_ARG_PREFIX(rope_scaling_mrope_section,
           "rope_scaling.mrope_section");
      LOAD_ARG_OR_PREFIX(rope_theta, "rope_theta", 1000000.0f);
      LOAD_ARG_OR_PREFIX(vocab_size, "vocab_size", 152064);
   });
  
  if (args->rope_scaling_rope_type() == "default")
      args->rope_scaling_rope_type() = "mrope";
  LOG(INFO) << args->rope_scaling_rope_type();
  LOG(INFO) << args->rope_scaling_mrope_section().size();
  // thinker.vision_config
  LOAD_ARG_WITH_PREFIX_JSON("thinker_config.vision_config", [&]{
      LOAD_ARG_OR_PREFIX(mm_num_hidden_layers, "depth", 27);
      LOAD_ARG_OR_PREFIX(mm_hidden_act, "hidden_act", "gelu_pytorch_tanh");
      LOAD_ARG_OR_PREFIX(mm_hidden_size, "hidden_size", 1152);
      LOAD_ARG_OR_PREFIX(mm_intermediate_size, "intermediate_size", 4304);
      LOAD_ARG_OR_PREFIX(mm_num_attention_heads, "num_heads", 16);
      LOAD_ARG_OR_PREFIX(mm_num_channels, "in_channels", 3);
      LOAD_ARG_OR_PREFIX(mm_projection_dim, "out_hidden_size", 2048);
      LOAD_ARG_OR_PREFIX(mm_patch_size, "patch_size", 16);
      LOAD_ARG_OR_PREFIX(mm_num_position_embeddings,
                "num_position_embeddings",
                2304);
      LOAD_ARG_OR_PREFIX(mm_spatial_merge_size, "spatial_merge_size", 2);
      LOAD_ARG_PREFIX(mm_deepstack_visual_indexes,
               "deepstack_visual_indexes");
      LOAD_ARG_OR_PREFIX(mm_temporal_patch_size, "temporal_patch_size", 2);
      LOAD_ARG_OR_PREFIX(mm_image_size, "image_size", 768);
  

      LOAD_ARG_OR_FUNC_PREFIX(mm_head_dim, "head_dim", [&] {
        return args->mm_hidden_size() / args->mm_num_attention_heads();
      });
  });
  
  // thinker.audio_config
  LOAD_ARG_WITH_PREFIX_JSON("thinker_config.audio_config", [&]{
      LOAD_ARG_OR_PREFIX(mm_audio_num_attention_heads, "encoder_attention_heads", 20);
      LOAD_ARG_OR_PREFIX(mm_audio_hidden_size, "d_model", 1280);
      LOAD_ARG_OR_PREFIX(mm_audio_downsample_hidden_size, "downsample_hidden_size", 480);
      LOAD_ARG_OR_PREFIX(mm_audio_d_model, "d_model", 1280);
      LOAD_ARG_OR_PREFIX(mm_audio_num_mel_bins, "num_mel_bins", 128);
      LOAD_ARG_OR_PREFIX(mm_audio_max_source_positions, "max_source_positions", 1500);
      LOAD_ARG_OR_PREFIX(mm_audio_scale_embedding, "scale_embedding", false);
      LOAD_ARG_OR_PREFIX(mm_audio_n_window, "n_window", 50);
      LOAD_ARG_OR_PREFIX(mm_audio_n_window_infer, "n_window_infer", 800);
      LOAD_ARG_OR_PREFIX(mm_audio_conv_chunksize, "conv_chunksize", 500);
      LOAD_ARG_OR_PREFIX(mm_audio_encoder_layers, "encoder_layers", 32);
      LOAD_ARG_OR_PREFIX(mm_audio_output_dim, "output_dim", 2048);
  });
  LOG(INFO) << std::to_string(args->mm_audio_encoder_layers());

  // code2wav_config
  LOAD_ARG_WITH_PREFIX_JSON("code2wav_config", [&]{
      LOAD_ARG_OR(code2wav_config_codebook_size, "codebook_size", 2048);
      LOAD_ARG_OR(code2wav_config_hidden_size, "hidden_size", 1024);
      LOAD_ARG_OR(code2wav_config_max_position_embeddings, "max_position_embeddings", 8000);
      LOAD_ARG_OR(code2wav_config_rope_theta, "rope_theta", 10000.0);
      LOAD_ARG_OR(code2wav_config_num_attention_heads, "num_attention_heads", 16);
      LOAD_ARG_OR(code2wav_config_num_key_value_heads, "num_key_value_heads", 16);
      LOAD_ARG_OR(code2wav_config_attention_bias, "attention_bias", false);
      LOAD_ARG_OR(code2wav_config_sliding_window, "sliding_window", 72);
      LOAD_ARG_OR(code2wav_config_intermediate_size, "intermediate_size", 3072);
      LOAD_ARG_OR(code2wav_config_hidden_act, "hidden_act", std::string("silu"));
      LOAD_ARG_OR(code2wav_config_layer_scale_initial_scale, "layer_scale_initial_scale", 0.01);
      LOAD_ARG_OR(code2wav_config_rms_norm_eps, "rms_norm_eps", 1e-5);
      LOAD_ARG_OR(code2wav_config_num_hidden_layers, "num_hidden_layers", 8);
      LOAD_ARG_OR(code2wav_config_num_quantizers, "num_quantizers", 16);
      LOAD_ARG_OR(code2wav_config_decoder_dim, "decoder_dim", 1536);
      LOAD_ARG_OR(code2wav_config_attention_dropout, "attention_dropout", 0.0);
      LOAD_ARG_OR(code2wav_config_upsampling_ratios_vec, "upsampling_ratios_vec", (std::vector<int>{2, 2}));
      LOAD_ARG_OR(code2wav_config_upsample_rates_vec, "upsample_rates_vec", (std::vector<int>{8, 5, 4, 3}));
  });
});

}
