#pragma once

#if defined(USE_NPU)
#include <atb/atb_infer.h>
#include "xllm_kernels/core/include/atb_speed/log.h"
#endif

#include <c10/core/ScalarType.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <boost/algorithm/string.hpp> //TODO: @pxy what are they
#include <unordered_map>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model_context.h"
#include "models/llm/npu/qwen3_moe.h"
#include "models/model_registry.h"

//debug
#include "core/util/tensor_helper.h"

// #include "core/layers/lm_head.h"
// /*
// #if defined(USE_NPU)
// #include "core/layers/npu/npu_rms_norm_impl.h"
// #endif
// */

namespace xllm {
/*
multi-model input,
input_ids
input_embeds
attention-masks
xxx


*/
class Qwen3_Omni_Moe_ThinkerForConditionalGeneration : public torch::nn::Module {
  /*
  language_model_ =
  register_module("language_model", Glm4ForCausalLM(context));
  auto inputs_embeds = language_model_->get_input_embeddings(input_ids);
  auto emb = language_model_(tokens, positions, kv_caches, input_params);
  language_model_->logits
  */

};


class Qwen3_Omni_Moe_ModelImpl: public torch::nn::Module{};

class Qwen3_Omni_MoeTalkerResizeMLPImpl : public torch::nn::Module {
  public:
    Qwen3_Omni_MoeTalkerResizeMLPImpl(const ModelContext& context){
      auto model_args = context.get_model_args();
      linear_fc1 = register_module(
        "linear_fc1",
        torch::nn::Linear(torch::nn::LinearOptions(
            model_args.thinker_hidden_size(), 
            model_args.talker_text_intermediate_size()
        ).bias(true))
      );
      linear_fc2 = register_module(
        "linear_fc2",
        torch::nn::Linear(torch::nn::LinearOptions(
            model_args.talker_text_intermediate_size(), 
            model_args.talker_text_hidden_size()
        ).bias(true))
      );
    }

    torch::Tensor forward(const torch::Tensor& x){
      return linear_fc2(torch::silu(linear_fc1(x)));
    }

  private:
    torch::nn::Linear linear_fc1 = nullptr;
    torch::nn::Linear linear_fc2 = nullptr;
    
};
TORCH_MODULE(Qwen3_Omni_MoeTalkerResizeMLP);

class Qwen3_Omni_MoeThinkerForConditionalGenerationImpl: public torch::nn::Module{};
class Qwen3_Omni_MoeTalkerForConditionalGenerationImpl: public torch::nn::Module{
  public:
    /*
    From Transformers：
    // 1. 初始化
      self.model = Qwen3OmniMoeTalkerModel._from_config(config.text_config)
      self.text_projection = Qwen3OmniMoeTalkerResizeMLP(config)
      self.hidden_projection = Qwen3OmniMoeTalkerResizeMLP(config)
      self.codec_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
      self.code_predictor = Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration._from_config(
          config=config.code_predictor_config
      )
      self.rope_deltas = None

      self.vocab_size = config.text_config.vocab_size
      self.router_aux_loss_coef = config.text_config.router_aux_loss_coef
      self.num_experts = config.text_config.num_experts
      self.num_experts_per_tok = config.text_config.num_experts_per_tok
      self.spatial_merge_size = self.config.spatial_merge_size
    // 2. post_init(加载权重)
    */

    Qwen3_Omni_MoeTalkerForConditionalGenerationImpl(const ModelContext& context){
      LOG(INFO) << "Talker starts to init";
      auto model_args = context.get_model_args();
      auto options = context.get_tensor_options();
      model_ = register_module("model", Qwen3MoeModel(context)); //TODO: what is the string name for
      codec_head_ = register_module("codec_head", torch::nn::Linear(
         torch::nn::LinearOptions(model_args.talker_text_hidden_size(), model_args.talker_text_vocab_size()).bias(false))
            ); // TODO: @pxy maybe refactor as CausalLLM models'lm head
      codec_head_->weight.set_data(codec_head_->weight.to(options));
      LOG(INFO) << "Talker finished initing";
      text_projection_ = register_module("text_projection", Qwen3_Omni_MoeTalkerResizeMLP(context)); //TODO: torch组图？
      hidden_projection_ = register_module("hidden_projection_", Qwen3_Omni_MoeTalkerResizeMLP(context));
      // code_predictor_ = register_module(); TODO: @pxy
    }

    torch::Tensor forward(const torch::Tensor& tokens, // no need
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
      LOG(INFO) << "Talker starts to forward";
    /* 
    Xllm
      tokens
      positions TODO: @pxy is this rope?
      kv_caches: 
      input_params: 

    Transformers:
      input:
      inputs_embeds=talker_input_embed,
      trailing_text_hidden=trailing_text_hidden,
      tts_pad_embed=tts_pad_embed,
      talker_input_ids=talker_input_id,  # Not use input_ids to prevent repetation penalty out of bound
      **talker_kwargs,

      output:
      return Qwen3OmniMoeTalkerOutputWithPast(
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=(
            outputs.hidden_states,
            residual_codes,
        ),  # TODO: hack here to take residual codes out, need refactor.
        generation_step=generation_step + 1,
        )
      */
        print_tensor(hidden_states, "Talker input: hidden_states", 10);
        auto hidden_states = model_(tokens, positions, kv_caches, input_params);
        LOG(INFO) << "Talker finished forwarding";
      }

    torch::Tensor logits(const torch::Tensor& hidden_states, const torch::Tensor& selected_idxes) {
      return codec_head_(hidden_states); // TODO: its fake
    }     
    // ModelInputParams prepare_talker_input(){
    //   /*
    //   input: thinker_result
    //   immediate: thinker_embed, thinker_hidden, im_start_indexes, multimodal_mask, talker_special_tokens,
    //   tts_bos_embed, tts_eos_embed, tts_pad_embed
    //   output: talker_input_embed, talker_input_id
    //   */
    //   ModelInputParams input_params; 
    //   return input_params;
    // }

    void load_model(std::shared_ptr<ModelLoader> loader) { //TODO: @pxy
      for (const auto& state_dict : loader->get_state_dicts()) {
        model_->load_state_dict(state_dict->get_dict_with_prefix("talker.model"));
        auto weight = state_dict->get_tensor("codec_head.weight");
        if (weight.defined()) codec_head_->weight.data().copy_(weight);
      }
    }

    // TODO: @pxy what for
    virtual void prepare_expert_weight(int32_t layer_id,
                                    const std::vector<int32_t>& expert_ids) {
      return;
    }

    // TODO: @pxy what for
    virtual void update_expert_weight(int32_t layer_id) { return; }
  private:
    Qwen3MoeModel model_ = nullptr;
    Qwen3_Omni_MoeTalkerResizeMLP text_projection_ = nullptr;
    Qwen3_Omni_MoeTalkerResizeMLP hidden_projection_ = nullptr;
    torch::nn::Linear codec_head_ = nullptr;
    // Qwen3_Omni_MoeTalkerCodePredictorModelForConditionalGeneration code_predictor_;
};
TORCH_MODULE(Qwen3_Omni_MoeTalkerForConditionalGeneration);
REGISTER_CAUSAL_MODEL(qwen3_omni_moe_audio, Qwen3_Omni_MoeTalkerForConditionalGeneration);  // TODO

// class Qwen3_Omni_MoeForConditionalGenerationImpl : public torch::nn::Module {
//   public:
//     thinker_model_: qwen3 moe + deepstack_video
//     talker_model_: qwen3 moe + deepstack_video + shared moe expert
//     Qwen3_VLMoeForConditionalGenerationImpl(const ModelContext& context)
//       : model_args_(context.get_model_args()),
//         options_(context.get_tensor_options()) {
//       thinker_model_ = register_module("thinker_model", Qwen3MoeForCausalLM(context));
//     }

//     torch::Tensor forward(const torch::Tensor& tokens,
//                       const torch::Tensor& positions,
//                       std::vector<KVCache>& kv_caches,
//                       const ModelInputParams& input_params) {
//     }

//   private:
//     Qwen3MoeForCausalLM thinker_model_;
//     Qwen3MoeForCausalLM talker_model_;
// };

/* TODO:
class QWen2ForCausalLMImpl : public LlmForCausalLMImplBase<QWen2Model> {
 public:
  QWen2ForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<QWen2Model>(context) {}
};
TORCH_MODULE(QWen2ForCausalLM);
*/
// loader
REGISTER_MODEL_ARGS(qwen3_omni_moe_audio, [&] {
  //TODO: @pxy hack for run omni only
  LOAD_ARG_OR(model_type, "model_type", "qwen3moe_talker");
  // LOAD_ARG_OR(dtype, "", "bfloat16"); // TODO
  // LOAD_ARG_OR(rms_norm_eps, "", ); // TODO

  // Talker basic
  LOAD_ARG_OR(n_layers, "talker_config.text_config.num_hidden_layers", 20);
  LOAD_ARG(rope_scaling_mrope_section, "talker_config.text_config.rope_scaling.mrope_section");
  LOAD_ARG_OR(max_position_embeddings, "talker_config.text_config.max_position_embeddings",65536);
  LOAD_ARG_OR(rope_theta, "talker_config.text_config.rope_theta",1000000);
  // Attention
  LOAD_ARG_OR(head_dim, "talker_config.text_config.head_dim", 128);
  LOAD_ARG_OR(n_kv_heads, "talker_config.text_config.num_key_value_heads", 2);
  LOAD_ARG_OR(n_heads, "talker_config.text_config.num_attention_heads", 16);
  // MLP
  LOAD_ARG_OR(n_shared_experts, "talker_config.text_config.num_shared_experts", 1);
  LOAD_ARG_OR(num_experts, "talker_config.text_config.num_experts", 128);
  LOAD_ARG_OR(num_experts_per_tok, "talker_config.text_config.num_experts_per_tok", 6);
  // code C
  LOAD_ARG_OR(talker_text_hidden_size, "talker_config.text_config.hidden_size", 6);
  LOAD_ARG_OR(talker_text_vocab_size, "talker_config.text_config.vocab_size", 6);
  // resize MLP
  LOAD_ARG_OR(thinker_hidden_size, "talker_config.thinker_hidden_size", 2048);
  LOAD_ARG_OR(talker_text_intermediate_size, "talker_config.text_config.intermediate_size", 2048);
  LOAD_ARG_OR(talker_text_hidden_size, "talker_config.text_config.hidden_size", 1024);
  
  /*
      "attention_bias": false,
      "attention_dropout": 0,
      "decoder_sparse_step": 1,
      "head_dim": 128,
      "hidden_act": "silu",
      "hidden_size": 1024,
      "initializer_range": 0.02,
      "intermediate_size": 2048,
      "max_position_embeddings": 65536,
      "mlp_only_layers": [],
      "moe_intermediate_size": 384,
      "norm_topk_prob": true,
      "num_attention_heads": 16,
      "num_experts": 128,
      "num_experts_per_tok": 6,
      "num_hidden_layers": 20,
      "num_key_value_heads": 2,
      "rms_norm_eps": 1e-06,
      "rope_scaling": {
        "interleaved": true,
        "mrope_section": [
          24,
          20,
          20
        ],
        "rope_type": "default",
        "type": "default"
      },
      "rope_theta": 1000000,
      "router_aux_loss_coef": 0.001,
      "shared_expert_intermediate_size": 768,
      "sliding_window": null,
      "use_cache": true,
      "use_sliding_window": false,
      "vocab_size": 3072
  */
});


