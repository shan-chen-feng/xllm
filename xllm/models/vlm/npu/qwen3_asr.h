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

#pragma once

#include "models/llm/npu/qwen3_moe.h"
#include "qwen3_omni_moe.h"

namespace xllm {

class Qwen3_ASRForConditionalGenerationImpl : public torch::nn::Module {
 public:
  Qwen3_ASRForConditionalGenerationImpl(const ModelContext& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()) {
    audio_tower_ = register_module(
        "audio_tower", Qwen3OmniMoe_Thinker_AudioTransformer(context));
    language_model_ =
        register_module("language_model", QWen3ForCausalLM(context));
  }

  void prepare_encoder_input(
      const ModelInputParams& input_params,
      std::optional<Qwen3_OmniAudioInputs>& audio_inputs) {
    const auto& mm_data = input_params.mm_data;
    torch::Tensor input_features;
    if (const auto& res = mm_data.get<torch::Tensor>("input_features")) {
      input_features = res.value();
    }

    torch::Tensor feat_length;
    if (const auto& res = mm_data.get<torch::Tensor>("feat_length")) {
      feat_length = res.value();
    }

    torch::Tensor feat_origin_lens;
    if (const auto& res = mm_data.get<torch::Tensor>("feat_origin_lens")) {
      feat_origin_lens = res.value();
    }

    if (input_features.defined() && feat_length.defined() &&
        feat_origin_lens.defined()) {
      audio_inputs =
          Qwen3_OmniAudioInputs{input_features, feat_length, feat_origin_lens};
    }
  }

  MMDict get_multimodal_embeddings(const ModelInputParams& input_params) {
    std::optional<Qwen3_OmniAudioInputs> audio_input;
    prepare_encoder_input(input_params, audio_input);
    MMDict multimodal_embeds;
    if (audio_input) {
      auto feat_origin_lens =
          audio_input->feat_origin_lens.to(options_.device(), torch::kLong);

      auto input_features =
          audio_input->input_features.permute({1, 0}).to(options_);

      auto audio_embeds =
          audio_tower_->forward(input_features, input_params, feat_origin_lens);

      auto audio_tokens =
          audio_input->feat_length.cpu().contiguous().to(torch::kLong);

      std::vector<int64_t> feature_lens_vec(
          audio_tokens.data_ptr<int64_t>(),
          audio_tokens.data_ptr<int64_t>() + audio_tokens.numel());

      multimodal_embeds["audio|embedding"] =
          audio_embeds.split(feature_lens_vec, 0 /*dim*/);
    }
    return multimodal_embeds;
  }

  torch::Tensor generate_multimodal_mask(torch::Tensor input_ids) {
    return torch::Tensor();
  }

  torch::Tensor generate_multimodal_mask_with_audio(torch::Tensor input_ids) {
    auto special_token_ids =
        torch::tensor({model_args_.audio_token_id()},
                      input_ids.options().dtype(torch::kInt64));
    auto is_multimodal = torch::isin(input_ids, special_token_ids);
    return is_multimodal;
  }

  std::vector<torch::Tensor> get_deep_stacks(
      const ModelInputParams& input_params) {
    return {};
  }

  torch::Tensor merge_multimodal_embeddings(
      torch::Tensor inputs_embeds,
      const torch::Tensor& multimodal_embeds,
      const torch::Tensor& is_multimodal) {
    inputs_embeds.index_put_({is_multimodal}, multimodal_embeds);
    return inputs_embeds;
  }

  torch::Tensor get_input_embeddings(const torch::Tensor input_ids,
                                     const ModelInputParams& input_params) {
    const auto& mm_data = input_params.mm_data;
    torch::Tensor multimodal_embeds;
    if (const auto& emb = mm_data.get<torch::Tensor>("embedding")) {
      multimodal_embeds = emb.value();
    }
    auto inputs_embeds = language_model_->get_input_embeddings(input_ids);
    if (!multimodal_embeds.defined()) {
      return inputs_embeds;
    }
    auto is_multimodal_with_audio =
        generate_multimodal_mask_with_audio(input_ids);
    inputs_embeds = merge_multimodal_embeddings(
        inputs_embeds, multimodal_embeds, is_multimodal_with_audio);
    return inputs_embeds;
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    input_params.deep_stacks = std::move(get_deep_stacks(input_params));
    return language_model_(tokens, positions, kv_caches, input_params);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    return language_model_->logits(hidden_states, seleted_idxes);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      if (state_dict->get_tensor("thinker.lm_head.weight").defined()) {
        state_dict->rename_prefix_inplace("thinker.lm_head.", "lm_head.");
      }
      audio_tower_->load_state_dict(
          state_dict->get_dict_with_prefix("thinker.audio_tower."));
    }
    // verify
    audio_tower_->verify_loaded_weights("thinker.audio_tower.");
    audio_tower_->merge_loaded_weights();
    audio_tower_->to(options_.device(),
                     torch::typeMetaToScalarType(options_.dtype()));

    if (!model_args_.image_embedding_mode()) {
      language_model_->load_model(std::move(loader), "thinker.model.");
    }
  }

  layer::NpuLmHead get_npu_lm_head() {
    return language_model_->get_npu_lm_head();
  }

  void set_npu_lm_head(layer::NpuLmHead& head) {
    language_model_->set_npu_lm_head(head);
  }

  layer::NpuWordEmbedding get_npu_word_embedding() {
    return language_model_->get_npu_word_embedding();
  }

  void set_npu_word_embedding(layer::NpuWordEmbedding& npu_word_embedding) {
    language_model_->set_npu_word_embedding(npu_word_embedding);
  }

 private:
  ModelArgs model_args_;
  torch::TensorOptions options_;
  Qwen3OmniMoe_Thinker_AudioTransformer audio_tower_{nullptr};
  QWen3ForCausalLM language_model_{nullptr};
};
TORCH_MODULE(Qwen3_ASRForConditionalGeneration);

REGISTER_INPUT_PROCESSOR(qwen3_asr, Qwen3OmniMoe_InputProcessor);
REGISTER_CAUSAL_VLM_MODEL(qwen3_asr, Qwen3_ASRForConditionalGeneration);
REGISTER_FEATURE_EXTRACTOR(qwen3_asr, WhisperFeatureExtractor);

REGISTER_MODEL_ARGS(qwen3_asr, [&] {
  LOAD_ARG_OR(model_type, "model_type", "qwen3_asr");

  // feature extractor default config
  LOAD_ARG_OR(has_feature_extractor, "has_feature_extractor", true);
  LOAD_ARG_OR(mm_audio_truncation, "truncation", false);
  // padding_strategy default to be PADDING_STRATEGT::LONGEST
  LOAD_ARG_OR(mm_audio_padding_strategy, "padding_strategy", 1);
  LOAD_ARG_OR(mm_audio_max_length, "max_length", -1);
  LOAD_ARG_OR(mm_audio_pad_to_multiple_of, "pad_to_multiple_of", -1);
  LOAD_ARG_OR(mm_audio_do_normalize, "do_normalize", false);
  LOAD_ARG_OR(
      mm_audio_return_token_timestamps, "return_token_timestamps", false);
  LOAD_ARG_OR(mm_audio_return_attention_mask, "return_attention_mask", true);

  // thinker config
  LOAD_ARG_WITH_PREFIX_JSON("thinker_config", [&] {
    LOAD_ARG_OR_PREFIX(audio_token_id, "audio_token_id", 151675);
    LOAD_ARG_OR_PREFIX(audio_start_token_id, "audio_start_token_id", 151669);
    LOAD_ARG_OR_PREFIX(audio_end_token_id, "audio_end_token_id", 151670);
    LOAD_ARG_OR_PREFIX(dtype, "dtype", "bfloat16");
  });

  // thinker.text_config
  LOAD_ARG_WITH_PREFIX_JSON("thinker_config.text_config", [&] {
    LOAD_ARG_OR_PREFIX(attention_bias, "attention_bias", false);
    LOAD_ARG_OR_PREFIX(attention_dropout, "attention_dropout", 0.0);

    LOAD_ARG_OR_PREFIX(bos_token_id, "bos_token_id", 151643);
    LOAD_ARG_OR_PREFIX(eos_token_id, "eos_token_id", 151645);
    LOAD_ARG_OR_PREFIX(hidden_act, "hidden_act", "silu");
    LOAD_ARG_OR_PREFIX(hidden_size, "hidden_size", 2048);
    LOAD_ARG_OR_PREFIX(intermediate_size, "intermediate_size", 6144);
    LOAD_ARG_OR_PREFIX(
        max_position_embeddings, "max_position_embeddings", 65536);
    LOAD_ARG_OR_PREFIX(max_window_layers, "max_window_layers", 28);
    LOAD_ARG_OR_PREFIX(n_heads, "num_attention_heads", 16);
    LOAD_ARG_OR_PREFIX(n_layers, "num_hidden_layers", 28);
    LOAD_ARG_OR_PREFIX(n_kv_heads, "num_key_value_heads", 8);
    LOAD_ARG_OR_PREFIX(rms_norm_eps, "rms_norm_eps", 1e-06);
    LOAD_ARG_OR_PREFIX(sliding_window, "sliding_window", 32768);
    LOAD_ARG_OR_PREFIX(tie_word_embeddings, "tie_word_embeddings", true);
    LOAD_ARG_PREFIX(rope_scaling_mrope_section, "rope_scaling.mrope_section");
    LOAD_ARG_OR_PREFIX(initializer_range, "initializer_range", 0.02);
    LOAD_ARG_OR_PREFIX(use_sliding_window, "use_sliding_window", false);
    LOAD_ARG_OR_FUNC_PREFIX(head_dim, "head_dim", [&] {
      return args->hidden_size() / args->n_heads();
    });
    LOAD_ARG_OR_PREFIX(rope_scaling_rope_type, "rope_scaling.type", "mrope");
    LOAD_ARG_PREFIX(rope_scaling_mrope_section, "rope_scaling.mrope_section");
    LOAD_ARG_OR_PREFIX(rope_theta, "rope_theta", 1000000.0f);
    LOAD_ARG_OR_PREFIX(vocab_size, "vocab_size", 151936);
  });

  if (args->rope_scaling_rope_type() == "default") {
    args->rope_scaling_rope_type() = "mrope";
  }

  // thinker.audio_config
  LOAD_ARG_WITH_PREFIX_JSON("thinker_config.audio_config", [&] {
    LOAD_ARG_OR_PREFIX(
        mm_audio_num_attention_heads, "encoder_attention_heads", 16);
    LOAD_ARG_OR_PREFIX(mm_audio_hidden_size, "d_model", 1024);
    LOAD_ARG_OR_PREFIX(
        mm_audio_downsample_hidden_size, "downsample_hidden_size", 480);
    LOAD_ARG_OR_PREFIX(mm_audio_d_model, "d_model", 1024);
    LOAD_ARG_OR_PREFIX(mm_audio_num_mel_bins, "num_mel_bins", 128);
    LOAD_ARG_OR_PREFIX(
        mm_audio_max_source_positions, "max_source_positions", 1500);
    LOAD_ARG_OR_PREFIX(mm_audio_scale_embedding, "scale_embedding", false);
    LOAD_ARG_OR_PREFIX(mm_audio_n_window, "n_window", 50);
    LOAD_ARG_OR_PREFIX(mm_audio_n_window_infer, "n_window_infer", 800);
    LOAD_ARG_OR_PREFIX(mm_audio_conv_chunksize, "conv_chunksize", 500);
    LOAD_ARG_OR_PREFIX(mm_audio_encoder_layers, "encoder_layers", 24);
    LOAD_ARG_OR_PREFIX(mm_audio_output_dim, "output_dim", 2048);
  });
});

}  // namespace xllm
