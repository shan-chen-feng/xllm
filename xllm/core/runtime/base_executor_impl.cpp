/* Copyright 2025-2026 The xLLM Authors.

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

#include "base_executor_impl.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <iomanip>
#include <sstream>

#include "common/metrics.h"

namespace xllm {

namespace {
// DIAGNOSTIC: mirror of the graph executor's block dump so eager (no-graph)
// draft inference can be compared element-wise against the graph replay. Prints
// abs-sum + first up-to-5x5 block. Remove once localized.
void log_hidden_block(const char* tag,
                      const char* which,
                      const torch::Tensor& t) {
  if (!t.defined() || t.numel() == 0) {
    LOG(INFO) << tag << " " << which << " <undefined/empty>";
    return;
  }
  torch::Tensor m = t.dim() == 1 ? t.unsqueeze(0) : t;
  const int64_t rows = std::min<int64_t>(5, m.size(0));
  const int64_t cols = std::min<int64_t>(5, m.size(-1));
  torch::Tensor blk = m.slice(0, 0, rows)
                          .slice(1, 0, cols)
                          .to(torch::kCPU)
                          .to(torch::kFloat32)
                          .contiguous();
  const double abs_sum = m.to(torch::kFloat32).abs().sum().item<double>();
  std::ostringstream os;
  os.setf(std::ios::fixed);
  os << std::setprecision(6);
  os << tag << " " << which << " shape=[" << m.size(0) << "," << m.size(-1)
     << "] abs_sum=" << std::setprecision(9) << abs_sum << std::setprecision(6)
     << " block=";
  const float* p = blk.const_data_ptr<float>();
  for (int64_t r = 0; r < rows; ++r) {
    os << "[";
    for (int64_t c = 0; c < cols; ++c) {
      os << p[r * cols + c] << (c + 1 < cols ? " " : "");
    }
    os << "]";
  }
  LOG(INFO) << os.str();
}
}  // namespace

BaseExecutorImpl::BaseExecutorImpl(CausalLM* model,
                                   const ModelArgs& args,
                                   const torch::Device& device,
                                   const runtime::Options& options)
    : model_(model), args_(args), device_(device), options_(options) {}

ForwardInput BaseExecutorImpl::prepare_inputs(Batch& batch) {
  return batch.prepare_forward_input(
      options_.num_decoding_tokens(), 0, args_, options_.cp_size());
}

ModelOutput BaseExecutorImpl::run(const torch::Tensor& tokens,
                                  const torch::Tensor& positions,
                                  std::vector<KVCache>& kv_caches,
                                  const ModelInputParams& params) {
  COUNTER_INC(num_model_execution_total_eager);

  // DIAGNOSTIC: log the same input/output as the graph replay path so single
  // buffer graph-vs-eager can be compared for identical inputs. Gated to the
  // draft engine to match the graph path (which only graph-captures the draft
  // in this eagle3 config) and to keep noise down. Remove once localized.
  const bool dbg = options_.is_draft_engine();
  if (dbg) {
    const int64_t nt = tokens.defined() ? tokens.size(0) : 0;
    LOG(INFO) << std::setprecision(12) << "[EAGER_INFER_INPUTS]"
              << " model=" << static_cast<const void*>(model_)
              << " num_tokens=" << nt << " tok_sum="
              << (tokens.defined() && tokens.numel() > 0
                      ? tokens.to(torch::kLong).sum().item<int64_t>()
                      : 0)
              << " pos_sum="
              << (positions.defined() && positions.numel() > 0
                      ? positions.to(torch::kLong).sum().item<int64_t>()
                      : 0);
    log_hidden_block(
        "[EAGER_INFER]", "in_emb", params.embedding.input_embedding);
  }

  ModelOutput out = model_->forward(tokens, positions, kv_caches, params);

  if (dbg) {
    log_hidden_block("[EAGER_INFER]", "out_hidden", out.hidden_states);
    log_hidden_block("[EAGER_INFER]", "out_aux", out.aux_hidden_states);
  }
  return out;
}

}  // namespace xllm
