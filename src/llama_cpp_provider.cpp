#include "llama_cpp_provider.hpp"

#include <llama.h>

#include <algorithm>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace runtime {
namespace {

static std::string BasenameNoExt(const std::string& path) {
  std::filesystem::path p(path);
  auto stem = p.stem().string();
  if (!stem.empty()) return stem;
  return p.filename().string();
}

static llama_token SampleGreedy(const float* logits, int n_vocab) {
  int best = 0;
  float best_v = logits[0];
  for (int i = 1; i < n_vocab; i++) {
    if (logits[i] > best_v) {
      best_v = logits[i];
      best = i;
    }
  }
  return static_cast<llama_token>(best);
}

static std::string TokenToPiece(const llama_model* model, llama_token tok) {
  std::string out;
  out.resize(64);
  int n = llama_token_to_piece(model, tok, out.data(), static_cast<int>(out.size()), 0, false);
  if (n < 0) {
    out.resize(static_cast<size_t>(-n));
    n = llama_token_to_piece(model, tok, out.data(), static_cast<int>(out.size()), 0, false);
  }
  if (n > 0) out.resize(static_cast<size_t>(n));
  return out;
}

}  // namespace

LlamaCppProvider::LlamaCppProvider(std::string model_path) : model_path_(std::move(model_path)) {
  model_id_ = BasenameNoExt(model_path_);
}

LlamaCppProvider::~LlamaCppProvider() {
  std::lock_guard<std::mutex> lock(mu_);
  if (ctx_) {
    llama_free(ctx_);
    ctx_ = nullptr;
  }
  if (model_) {
    llama_free_model(model_);
    model_ = nullptr;
  }
}

std::string LlamaCppProvider::Name() const {
  return "llama_cpp";
}

std::vector<ModelInfo> LlamaCppProvider::ListModels(std::string* err) {
  if (model_path_.empty()) {
    if (err) *err = "llama_cpp: missing model path";
    return {};
  }
  ModelInfo m;
  m.id = model_id_.empty() ? "default" : model_id_;
  m.owned_by = "llama_cpp";
  return {m};
}

std::optional<std::vector<double>> LlamaCppProvider::Embeddings(const std::string&, const std::string&, std::string* err) {
  if (err) *err = "llama_cpp: embeddings not supported";
  return std::nullopt;
}

bool LlamaCppProvider::EnsureLoaded(std::string* err) {
  if (model_) return true;
  if (model_path_.empty()) {
    if (err) *err = "llama_cpp: missing model path";
    return false;
  }
  llama_backend_init();

  llama_model_params mparams = llama_model_default_params();
  model_ = llama_load_model_from_file(model_path_.c_str(), mparams);
  if (!model_) {
    if (err) *err = "llama_cpp: failed to load model";
    return false;
  }

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 4096;
  cparams.n_threads = std::max(1u, std::thread::hardware_concurrency());
  ctx_ = llama_new_context_with_model(model_, cparams);
  if (!ctx_) {
    if (err) *err = "llama_cpp: failed to create context";
    return false;
  }
  return true;
}

std::string LlamaCppProvider::BuildPrompt(const std::vector<ChatMessage>& messages) const {
  std::string p;
  for (const auto& m : messages) {
    std::string role = m.role;
    std::transform(role.begin(), role.end(), role.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    p += role;
    p += ": ";
    p += m.content;
    p += "\n";
  }
  p += "ASSISTANT: ";
  return p;
}

std::optional<ChatResponse> LlamaCppProvider::ChatOnce(const ChatRequest& req, std::string* err) {
  std::string out_text;
  const bool ok = ChatStream(
      req,
      [&](const std::string& delta) {
        out_text += delta;
      },
      []() {},
      err);
  if (!ok) return std::nullopt;
  ChatResponse r;
  r.model = req.model;
  r.content = std::move(out_text);
  r.done = true;
  return r;
}

bool LlamaCppProvider::ChatStream(const ChatRequest& req,
                                  const std::function<void(const std::string&)>& on_delta,
                                  const std::function<void()>& on_done,
                                  std::string* err) {
  std::lock_guard<std::mutex> lock(mu_);
  if (!EnsureLoaded(err)) return false;

  llama_kv_cache_clear(ctx_);

  std::string prompt = BuildPrompt(req.messages);

  std::vector<llama_token> prompt_tokens;
  prompt_tokens.resize(prompt.size() + 8);
  int n_prompt = llama_tokenize(model_, prompt.c_str(), static_cast<int>(prompt.size()), prompt_tokens.data(),
                                static_cast<int>(prompt_tokens.size()), true, true);
  if (n_prompt < 0) {
    if (err) *err = "llama_cpp: tokenize failed";
    return false;
  }
  prompt_tokens.resize(static_cast<size_t>(n_prompt));

  llama_batch batch = llama_batch_init(static_cast<int>(prompt_tokens.size()), 0, 1);
  for (size_t i = 0; i < prompt_tokens.size(); i++) {
    batch.token[batch.n_tokens] = prompt_tokens[i];
    batch.pos[batch.n_tokens] = static_cast<llama_pos>(i);
    batch.n_seq_id[batch.n_tokens] = 1;
    batch.seq_id[batch.n_tokens][0] = 0;
    batch.logits[batch.n_tokens] = (i + 1 == prompt_tokens.size());
    batch.n_tokens++;
  }
  if (llama_decode(ctx_, batch) != 0) {
    llama_batch_free(batch);
    if (err) *err = "llama_cpp: decode failed";
    return false;
  }
  llama_batch_free(batch);

  const int n_vocab = llama_n_vocab(model_);
  const llama_token eos = llama_token_eos(model_);
  llama_pos n_past = static_cast<llama_pos>(prompt_tokens.size());

  const int max_new_tokens = 256;
  for (int i = 0; i < max_new_tokens; i++) {
    const float* logits = llama_get_logits(ctx_);
    llama_token next = SampleGreedy(logits, n_vocab);
    if (next == eos) break;

    auto piece = TokenToPiece(model_, next);
    if (!piece.empty()) on_delta(piece);

    llama_batch b = llama_batch_init(1, 0, 1);
    b.token[0] = next;
    b.pos[0] = n_past;
    b.n_seq_id[0] = 1;
    b.seq_id[0][0] = 0;
    b.logits[0] = true;
    b.n_tokens = 1;
    n_past++;
    if (llama_decode(ctx_, b) != 0) {
      llama_batch_free(b);
      if (err) *err = "llama_cpp: decode failed";
      return false;
    }
    llama_batch_free(b);
  }

  on_done();
  return true;
}

}  // namespace runtime
