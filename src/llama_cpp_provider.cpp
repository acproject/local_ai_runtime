#include "llama_cpp_provider.hpp"

#include <llama.h>

#include <algorithm>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace runtime {
namespace {

static std::string BasenameNoExt(const std::string& path) {
  std::filesystem::path p(path);
  auto stem = p.stem().string();
  if (!stem.empty()) return stem;
  return p.filename().string();
}

static bool IsFirstShardFile(const std::filesystem::path& p) {
  const std::string stem = p.stem().string();
  return stem.find("-00001-of-") != std::string::npos;
}

static bool PreferModelFile(const std::filesystem::path& cand, const std::filesystem::path& cur) {
  const bool c1 = IsFirstShardFile(cand);
  const bool c2 = IsFirstShardFile(cur);
  if (c1 != c2) return c1;
  return cand.filename().string() < cur.filename().string();
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

LlamaCppProvider::LlamaCppProvider(std::string model_path) : model_root_(std::move(model_path)) {
  if (model_root_.empty()) {
    std::filesystem::path fallback = std::filesystem::path("models");
    std::error_code ec;
    if (std::filesystem::exists(fallback, ec) && std::filesystem::is_directory(fallback, ec)) {
      model_root_ = fallback.string();
    }
  }
  BuildModelIndex();
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
  if (model_ids_.empty()) {
    if (err) *err = "llama_cpp: missing model path";
    return {};
  }
  std::vector<ModelInfo> out;
  out.reserve(model_ids_.size());
  for (const auto& id : model_ids_) {
    ModelInfo m;
    m.id = id;
    m.owned_by = "llama_cpp";
    out.push_back(std::move(m));
  }
  return out;
}

std::optional<std::vector<double>> LlamaCppProvider::Embeddings(const std::string&, const std::string&, std::string* err) {
  if (err) *err = "llama_cpp: embeddings not supported";
  return std::nullopt;
}

void LlamaCppProvider::BuildModelIndex() {
  model_paths_by_id_.clear();
  model_ids_.clear();
  root_is_dir_ = false;

  if (model_root_.empty()) return;

  std::filesystem::path root(model_root_);
  std::error_code ec;
  if (!std::filesystem::exists(root, ec)) return;

  if (std::filesystem::is_directory(root, ec)) {
    root_is_dir_ = true;
    for (std::filesystem::recursive_directory_iterator it(root, ec), end; it != end; it.increment(ec)) {
      if (ec) break;
      if (!it->is_regular_file(ec)) continue;
      auto p = it->path();
      auto ext = p.extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(),
                     [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      if (ext != ".gguf") continue;

      auto rel_dir = std::filesystem::relative(p.parent_path(), root, ec);
      if (ec) continue;

      std::string id;
      if (rel_dir.empty() || rel_dir == ".") {
        id = root.filename().string();
        if (id.empty()) id = BasenameNoExt(p.string());
      } else {
        id = rel_dir.generic_string();
      }
      if (id.empty()) continue;

      auto existing = model_paths_by_id_.find(id);
      if (existing == model_paths_by_id_.end()) {
        model_paths_by_id_[id] = p.string();
      } else {
        std::filesystem::path cur(existing->second);
        if (PreferModelFile(p, cur)) existing->second = p.string();
      }
    }
  } else if (std::filesystem::is_regular_file(root, ec)) {
    root_is_dir_ = false;
    std::string id = BasenameNoExt(root.string());
    if (!id.empty()) model_paths_by_id_[id] = root.string();
  }

  model_ids_.reserve(model_paths_by_id_.size());
  for (const auto& [id, _] : model_paths_by_id_) model_ids_.push_back(id);
  std::sort(model_ids_.begin(), model_ids_.end());
}

std::optional<std::string> LlamaCppProvider::ResolveModelPath(const std::string& requested_model, std::string* err) const {
  if (model_ids_.empty()) {
    if (err) *err = "llama_cpp: missing model path";
    return std::nullopt;
  }

  if (requested_model == "any" && model_ids_.size() == 1) {
    const auto& only_id = model_ids_[0];
    auto it = model_paths_by_id_.find(only_id);
    if (it == model_paths_by_id_.end()) {
      if (err) *err = "llama_cpp: missing model path";
      return std::nullopt;
    }
    return it->second;
  }

  if (!root_is_dir_) {
    const auto& only_id = model_ids_[0];
    if (!requested_model.empty() && requested_model != only_id) {
      if (err) *err = "llama_cpp: unknown model";
      return std::nullopt;
    }
    auto it = model_paths_by_id_.find(only_id);
    if (it == model_paths_by_id_.end()) {
      if (err) *err = "llama_cpp: missing model path";
      return std::nullopt;
    }
    return it->second;
  }

  auto it = model_paths_by_id_.find(requested_model);
  if (it == model_paths_by_id_.end()) {
    if (err) *err = "llama_cpp: unknown model";
    return std::nullopt;
  }
  return it->second;
}

bool LlamaCppProvider::EnsureLoaded(const std::string& model_path, std::string* err) {
  if (model_ && ctx_ && loaded_model_path_ == model_path) return true;

  if (ctx_) {
    llama_free(ctx_);
    ctx_ = nullptr;
  }
  if (model_) {
    llama_free_model(model_);
    model_ = nullptr;
  }
  loaded_model_path_.clear();

  if (model_path.empty()) {
    if (err) *err = "llama_cpp: missing model path";
    return false;
  }

  static std::once_flag init_once;
  std::call_once(init_once, [] { llama_backend_init(); });

  llama_model_params mparams = llama_model_default_params();
  model_ = llama_load_model_from_file(model_path.c_str(), mparams);
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
  loaded_model_path_ = model_path;
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
  auto model_path = ResolveModelPath(req.model, err);
  if (!model_path) return false;
  if (!EnsureLoaded(*model_path, err)) return false;

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
