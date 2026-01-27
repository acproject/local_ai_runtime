#include "llama_cpp_provider.hpp"

#include <llama.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <cstring>
#include <filesystem>
#include <iostream>
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

static std::string TokenToPiece(const llama_vocab* vocab, llama_token tok) {
  std::string out;
  out.resize(64);
  int n = llama_token_to_piece(vocab, tok, out.data(), static_cast<int>(out.size()), 0, false);
  if (n < 0) {
    out.resize(static_cast<size_t>(-n));
    n = llama_token_to_piece(vocab, tok, out.data(), static_cast<int>(out.size()), 0, false);
  }
  if (n > 0) {
    out.resize(static_cast<size_t>(n));
    out.erase(std::remove(out.begin(), out.end(), '\0'), out.end());
  }
  return out;
}

static std::mutex g_log_mu;
static std::deque<std::string> g_llama_log_lines;

static std::string TrimWs(std::string s) {
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '\t')) s.pop_back();
  size_t i = 0;
  while (i < s.size() && (s[i] == '\n' || s[i] == '\r' || s[i] == ' ' || s[i] == '\t')) i++;
  if (i > 0) s.erase(0, i);
  return s;
}

static std::string GetEnvStr(const char* name) {
#if defined(_WIN32)
  char* buf = nullptr;
  size_t len = 0;
  if (_dupenv_s(&buf, &len, name) != 0 || !buf) return {};
  std::string out(buf);
  std::free(buf);
  return out;
#else
  const char* v = std::getenv(name);
  if (!v) return {};
  return std::string(v);
#endif
}

static std::string ToLowerAscii(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

static bool TryParseInt32(const std::string& s, int32_t* out) {
  if (!out) return false;
  if (s.empty()) return false;
  char* end = nullptr;
  long v = std::strtol(s.c_str(), &end, 10);
  if (end == s.c_str() || (end && *end != '\0')) return false;
  if (v < INT32_MIN || v > INT32_MAX) return false;
  *out = static_cast<int32_t>(v);
  return true;
}

static bool TryParseUint32(const std::string& s, uint32_t* out) {
  if (!out) return false;
  if (s.empty()) return false;
  char* end = nullptr;
  unsigned long v = std::strtoul(s.c_str(), &end, 10);
  if (end == s.c_str() || (end && *end != '\0')) return false;
  if (v > UINT32_MAX) return false;
  *out = static_cast<uint32_t>(v);
  return true;
}

static bool TryParseFloat(const std::string& s, float* out) {
  if (!out) return false;
  if (s.empty()) return false;
  char* end = nullptr;
  float v = std::strtof(s.c_str(), &end);
  if (end == s.c_str() || (end && *end != '\0')) return false;
  *out = v;
  return true;
}

static bool TryParseBool(const std::string& s, bool* out) {
  if (!out) return false;
  const std::string v = ToLowerAscii(TrimWs(s));
  if (v == "1" || v == "true" || v == "yes" || v == "y" || v == "on") {
    *out = true;
    return true;
  }
  if (v == "0" || v == "false" || v == "no" || v == "n" || v == "off") {
    *out = false;
    return true;
  }
  return false;
}

static std::optional<llama_split_mode> ParseSplitMode(const std::string& s) {
  const std::string v = ToLowerAscii(TrimWs(s));
  if (v == "none" || v == "single") return LLAMA_SPLIT_MODE_NONE;
  if (v == "layer" || v == "layers") return LLAMA_SPLIT_MODE_LAYER;
  if (v == "row" || v == "rows") return LLAMA_SPLIT_MODE_ROW;
  return std::nullopt;
}

static std::optional<llama_flash_attn_type> ParseFlashAttnType(const std::string& s) {
  const std::string v = ToLowerAscii(TrimWs(s));
  if (v == "auto") return LLAMA_FLASH_ATTN_TYPE_AUTO;
  if (v == "enabled" || v == "enable" || v == "1" || v == "true" || v == "on") return LLAMA_FLASH_ATTN_TYPE_ENABLED;
  if (v == "disabled" || v == "disable" || v == "0" || v == "false" || v == "off") return LLAMA_FLASH_ATTN_TYPE_DISABLED;
  return std::nullopt;
}

struct LlamaRuntimeConfig {
  int32_t n_gpu_layers = 0;
  std::optional<llama_split_mode> split_mode;
  std::optional<int32_t> main_gpu;
  std::optional<bool> offload_kqv;
  std::optional<llama_flash_attn_type> flash_attn;
  std::optional<uint32_t> n_ctx;
  std::optional<uint32_t> n_batch;
  std::optional<uint32_t> n_ubatch;
  std::optional<int32_t> n_threads;
  std::optional<int32_t> n_threads_batch;
  std::optional<bool> unload_after_chat;
  std::optional<int32_t> max_new_tokens;
  std::optional<float> temperature;
  std::optional<float> top_p;
  std::optional<int32_t> seed;
  std::optional<int32_t> penalty_last_n;
  std::optional<float> penalty_repeat;
  bool requested = false;
};

static LlamaRuntimeConfig LoadLlamaRuntimeConfigFromEnv() {
  LlamaRuntimeConfig cfg;

  {
    const std::string v = GetEnvStr("LLAMA_CPP_N_GPU_LAYERS");
    if (!v.empty()) {
      int32_t n = 0;
      if (TryParseInt32(TrimWs(v), &n)) cfg.n_gpu_layers = n;
    } else {
      const std::string v2 = GetEnvStr("LLAMA_CPP_GPU_LAYERS");
      if (!v2.empty()) {
        int32_t n = 0;
        if (TryParseInt32(TrimWs(v2), &n)) cfg.n_gpu_layers = n;
      }
    }
  }

  {
    const std::string v = GetEnvStr("LLAMA_CPP_MAIN_GPU");
    int32_t n = 0;
    if (!v.empty() && TryParseInt32(TrimWs(v), &n)) cfg.main_gpu = n;
  }

  {
    const std::string v = GetEnvStr("LLAMA_CPP_SPLIT_MODE");
    if (!v.empty()) cfg.split_mode = ParseSplitMode(v);
  }

  {
    const std::string v = GetEnvStr("LLAMA_CPP_OFFLOAD_KQV");
    bool b = false;
    if (!v.empty() && TryParseBool(v, &b)) cfg.offload_kqv = b;
  }

  {
    const std::string v = GetEnvStr("LLAMA_CPP_FLASH_ATTN");
    if (!v.empty()) cfg.flash_attn = ParseFlashAttnType(v);
  }

  {
    const std::string v = GetEnvStr("LLAMA_CPP_N_CTX");
    uint32_t n = 0;
    if (!v.empty() && TryParseUint32(TrimWs(v), &n) && n > 0) cfg.n_ctx = n;
  }

  {
    const std::string v = GetEnvStr("LLAMA_CPP_N_BATCH");
    uint32_t n = 0;
    if (!v.empty() && TryParseUint32(TrimWs(v), &n) && n > 0) cfg.n_batch = n;
  }

  {
    const std::string v = GetEnvStr("LLAMA_CPP_N_UBATCH");
    uint32_t n = 0;
    if (!v.empty() && TryParseUint32(TrimWs(v), &n) && n > 0) cfg.n_ubatch = n;
  }

  {
    const std::string v = GetEnvStr("LLAMA_CPP_N_THREADS");
    int32_t n = 0;
    if (!v.empty() && TryParseInt32(TrimWs(v), &n) && n > 0) cfg.n_threads = n;
  }

  {
    const std::string v = GetEnvStr("LLAMA_CPP_N_THREADS_BATCH");
    int32_t n = 0;
    if (!v.empty() && TryParseInt32(TrimWs(v), &n) && n > 0) cfg.n_threads_batch = n;
  }

  {
    const std::string v = GetEnvStr("LLAMA_CPP_UNLOAD_AFTER_CHAT");
    bool b = false;
    if (!v.empty() && TryParseBool(v, &b)) cfg.unload_after_chat = b;
  }

  {
    const std::string v = GetEnvStr("LLAMA_CPP_MAX_NEW_TOKENS");
    const std::string v2 = v.empty() ? GetEnvStr("LLAMA_CPP_MAX_TOKENS") : std::string();
    const std::string raw = !v.empty() ? v : v2;
    int32_t n = 0;
    if (!raw.empty() && TryParseInt32(TrimWs(raw), &n) && n > 0) cfg.max_new_tokens = n;
  }

  {
    const std::string v = GetEnvStr("LLAMA_CPP_TEMPERATURE");
    float f = 0.0f;
    if (!v.empty() && TryParseFloat(TrimWs(v), &f) && f >= 0.0f) cfg.temperature = f;
  }

  {
    const std::string v = GetEnvStr("LLAMA_CPP_TOP_P");
    float f = 0.0f;
    if (!v.empty() && TryParseFloat(TrimWs(v), &f) && f >= 0.0f && f <= 1.0f) cfg.top_p = f;
  }

  {
    const std::string v = GetEnvStr("LLAMA_CPP_SEED");
    int32_t n = 0;
    if (!v.empty() && TryParseInt32(TrimWs(v), &n)) cfg.seed = n;
  }

  {
    const std::string v = GetEnvStr("LLAMA_CPP_PENALTY_LAST_N");
    int32_t n = 0;
    if (!v.empty() && TryParseInt32(TrimWs(v), &n)) cfg.penalty_last_n = n;
  }

  {
    const std::string v = GetEnvStr("LLAMA_CPP_REPEAT_PENALTY");
    float f = 0.0f;
    if (!v.empty() && TryParseFloat(TrimWs(v), &f) && f > 0.0f) cfg.penalty_repeat = f;
  }

  cfg.requested = (cfg.n_gpu_layers != 0) || (cfg.offload_kqv.has_value() && cfg.offload_kqv.value());
  return cfg;
}

static void LlamaLogCallback(enum ggml_log_level level, const char* text, void*) {
  std::string line = text ? text : "";
  {
    std::lock_guard<std::mutex> lock(g_log_mu);
    g_llama_log_lines.push_back(line);
    while (g_llama_log_lines.size() > 200) g_llama_log_lines.pop_front();
  }
  if (level != GGML_LOG_LEVEL_NONE && text) {
    std::cerr << text;
  }
}

static std::string LastLlamaLogLine() {
  std::lock_guard<std::mutex> lock(g_log_mu);
  for (auto it = g_llama_log_lines.rbegin(); it != g_llama_log_lines.rend(); ++it) {
    auto t = TrimWs(*it);
    if (!t.empty()) return t;
  }
  return {};
}

static bool RecentLlamaLogsContain(const std::string& needle) {
  std::lock_guard<std::mutex> lock(g_log_mu);
  int checked = 0;
  for (auto it = g_llama_log_lines.rbegin(); it != g_llama_log_lines.rend() && checked < 200; ++it, ++checked) {
    if (it->find(needle) != std::string::npos) return true;
  }
  return false;
}

static std::string LastLlamaLogContaining(const std::string& needle) {
  std::lock_guard<std::mutex> lock(g_log_mu);
  int checked = 0;
  for (auto it = g_llama_log_lines.rbegin(); it != g_llama_log_lines.rend() && checked < 400; ++it, ++checked) {
    if (it->find(needle) != std::string::npos) return TrimWs(*it);
  }
  return {};
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
    llama_model_free(model_);
    model_ = nullptr;
  }
}

void LlamaCppProvider::Start() {
  std::lock_guard<std::mutex> lock(mu_);
  if (model_ids_.empty()) BuildModelIndex();
}

void LlamaCppProvider::Stop() {
  std::lock_guard<std::mutex> lock(mu_);
  if (ctx_) {
    llama_free(ctx_);
    ctx_ = nullptr;
  }
  if (model_) {
    llama_model_free(model_);
    model_ = nullptr;
  }
  loaded_model_path_.clear();
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
    llama_model_free(model_);
    model_ = nullptr;
  }
  loaded_model_path_.clear();

  if (model_path.empty()) {
    if (err) *err = "llama_cpp: missing model path";
    return false;
  }

  {
    std::error_code ec;
    if (!std::filesystem::exists(std::filesystem::path(model_path), ec)) {
      if (err) *err = "llama_cpp: model file not found";
      return false;
    }
  }

  static std::once_flag init_once;
  std::call_once(init_once, [] {
    llama_backend_init();
    llama_log_set(LlamaLogCallback, nullptr);
  });

  const auto cfg = LoadLlamaRuntimeConfigFromEnv();
  if (cfg.requested && !llama_supports_gpu_offload()) {
    if (err) *err = "llama_cpp: gpu offload requested but not supported in this build";
    return false;
  }

  static llama_model_kv_override deepseek2_overrides[2]{};
  deepseek2_overrides[0].tag = LLAMA_KV_OVERRIDE_TYPE_FLOAT;
  strncpy_s(deepseek2_overrides[0].key, sizeof(deepseek2_overrides[0].key), "deepseek2.rope.scaling.yarn_log_multiplier", _TRUNCATE);
  deepseek2_overrides[0].val_f64 = 0.0;
  deepseek2_overrides[1].key[0] = 0;

  static llama_model_kv_override glm4_tokpre_overrides[2]{};
  glm4_tokpre_overrides[0].tag = LLAMA_KV_OVERRIDE_TYPE_STR;
  strncpy_s(glm4_tokpre_overrides[0].key, sizeof(glm4_tokpre_overrides[0].key), "tokenizer.ggml.pre", _TRUNCATE);
  strncpy_s(glm4_tokpre_overrides[0].val_str, sizeof(glm4_tokpre_overrides[0].val_str), "chatglm-bpe", _TRUNCATE);
  glm4_tokpre_overrides[1].key[0] = 0;

  static llama_model_kv_override deepseek2_and_glm4_overrides[3]{};
  deepseek2_and_glm4_overrides[0] = deepseek2_overrides[0];
  deepseek2_and_glm4_overrides[1] = glm4_tokpre_overrides[0];
  deepseek2_and_glm4_overrides[2].key[0] = 0;

  auto try_load = [&](llama_model_params& p) -> llama_model* {
    return llama_model_load_from_file(model_path.c_str(), p);
  };

  auto pick_overrides = [&](bool force_yarn, bool force_glm4_pre) -> const llama_model_kv_override* {
    if (force_yarn && force_glm4_pre) return deepseek2_and_glm4_overrides;
    if (force_yarn) return deepseek2_overrides;
    if (force_glm4_pre) return glm4_tokpre_overrides;
    return nullptr;
  };

  auto try_load_with = [&](const llama_model_kv_override* overrides,
                           const std::optional<int32_t>& gpu_layers_override) -> llama_model* {
    llama_model_params p = llama_model_default_params();
    p.kv_overrides = overrides;
    int32_t n_gpu_layers = cfg.n_gpu_layers;
    if (gpu_layers_override.has_value()) n_gpu_layers = gpu_layers_override.value();
    if (n_gpu_layers != 0) p.n_gpu_layers = n_gpu_layers;
    if (cfg.split_mode.has_value()) p.split_mode = cfg.split_mode.value();
    if (cfg.main_gpu.has_value()) p.main_gpu = cfg.main_gpu.value();
    p.use_mmap = true;
    auto* m = try_load(p);
    if (!m) {
      p.use_mmap = false;
      m = try_load(p);
    }
    return m;
  };

  bool force_yarn = false;
  bool force_glm4_pre = false;
  const llama_model_kv_override* overrides = nullptr;

  for (int attempt = 0; attempt < 4 && !model_; attempt++) {
    model_ = try_load_with(overrides, std::nullopt);
    if (model_) break;

    force_yarn = force_yarn || RecentLlamaLogsContain("deepseek2.rope.scaling.yarn_log_multiplier");
    force_glm4_pre = force_glm4_pre || RecentLlamaLogsContain("unknown pre-tokenizer type: 'glm4'");
    force_glm4_pre = force_glm4_pre || (RecentLlamaLogsContain("unknown pre-tokenizer type") && RecentLlamaLogsContain("glm4"));

    const auto* next_overrides = pick_overrides(force_yarn, force_glm4_pre);
    if (next_overrides == overrides) break;
    overrides = next_overrides;
  }
  if (!model_) {
    const bool cuda_oom = RecentLlamaLogsContain("cudaMalloc failed") ||
                          RecentLlamaLogsContain("unable to allocate CUDA") ||
                          RecentLlamaLogsContain("CUDA out of memory");
    if (cuda_oom && cfg.n_gpu_layers != 0) {
      std::cout << "[provider] llama_cpp cuda oom, fallback to cpu\n";
      model_ = try_load_with(overrides, 0);
    }
  }
  if (!model_) {
    if (err) {
      auto root = LastLlamaLogContaining("llama_model_load: error loading model:");
      if (root.empty()) root = LastLlamaLogContaining("error loading model");
      if (root.empty()) root = LastLlamaLogLine();
      if (!root.empty()) {
        *err = "llama_cpp: failed to load model: " + root;
      } else {
        *err = "llama_cpp: failed to load model";
      }
    }
    return false;
  }

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = cfg.n_ctx.has_value() ? cfg.n_ctx.value() : 4096;
  cparams.n_threads = cfg.n_threads.has_value() ? cfg.n_threads.value() : std::max(1u, std::thread::hardware_concurrency());
  cparams.n_threads_batch = cfg.n_threads_batch.has_value() ? cfg.n_threads_batch.value() : cparams.n_threads;
  const bool gpu_defaults = llama_supports_gpu_offload();
  const uint32_t default_batch = gpu_defaults ? std::min<uint32_t>(512, cparams.n_ctx) : std::min<uint32_t>(2048, cparams.n_ctx);
  cparams.n_batch = cfg.n_batch.has_value() ? cfg.n_batch.value() : default_batch;
  const uint32_t default_ubatch = gpu_defaults ? std::min<uint32_t>(256, cparams.n_batch) : cparams.n_batch;
  cparams.n_ubatch = cfg.n_ubatch.has_value() ? cfg.n_ubatch.value() : default_ubatch;
  if (cfg.offload_kqv.has_value()) cparams.offload_kqv = cfg.offload_kqv.value();
  if (cfg.flash_attn.has_value()) {
    cparams.flash_attn_type = cfg.flash_attn.value();
  } else if (gpu_defaults) {
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
  }
  ctx_ = llama_init_from_model(model_, cparams);
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
  std::string finish_reason = "stop";
  const bool ok = ChatStream(
      req,
      [&](const std::string& delta) {
        out_text += delta;
      },
      [&](const std::string& fr) {
        finish_reason = fr;
      },
      err);
  if (!ok) return std::nullopt;
  auto rtrim = [&](std::string& s) {
    while (!s.empty()) {
      const char c = s.back();
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        s.pop_back();
        continue;
      }
      break;
    }
  };
  rtrim(out_text);
  out_text.erase(std::remove(out_text.begin(), out_text.end(), '\0'), out_text.end());
  for (const std::string& stop : {std::string("\nUser:"), std::string("\nUser"), std::string("\nUSER:"), std::string("\nUSER")}) {
    if (out_text.size() >= stop.size() && out_text.compare(out_text.size() - stop.size(), stop.size(), stop) == 0) {
      out_text.erase(out_text.size() - stop.size());
      rtrim(out_text);
    }
  }
  ChatResponse r;
  r.model = req.model;
  r.content = std::move(out_text);
  r.done = true;
  r.finish_reason = std::move(finish_reason);
  return r;
}

bool LlamaCppProvider::ChatStream(const ChatRequest& req,
                                  const std::function<void(const std::string&)>& on_delta,
                                  const std::function<void(const std::string& finish_reason)>& on_done,
                                  std::string* err) {
  std::lock_guard<std::mutex> lock(mu_);
  auto model_path = ResolveModelPath(req.model, err);
  if (!model_path) return false;
  if (!EnsureLoaded(*model_path, err)) return false;

  llama_memory_clear(llama_get_memory(ctx_), false);

  std::string prompt;
  const char* tmpl = llama_model_chat_template(model_, nullptr);
  if (tmpl && tmpl[0] != '\0') {
    std::vector<llama_chat_message> chat;
    chat.reserve(req.messages.size());
    size_t approx_len = 0;
    for (const auto& m : req.messages) {
      llama_chat_message cm{};
      cm.role = m.role.c_str();
      cm.content = m.content.c_str();
      chat.push_back(cm);
      approx_len += m.role.size() + m.content.size() + 16;
    }
    std::string buf;
    buf.resize(std::max<size_t>(256, approx_len * 2 + 64));
    int n = llama_chat_apply_template(tmpl, chat.data(), chat.size(), true, buf.data(), static_cast<int>(buf.size()));
    if (n > static_cast<int>(buf.size())) {
      buf.resize(static_cast<size_t>(n));
      n = llama_chat_apply_template(tmpl, chat.data(), chat.size(), true, buf.data(), static_cast<int>(buf.size()));
    }
    if (n > 0) {
      buf.resize(static_cast<size_t>(n));
      prompt = std::move(buf);
    } else {
      prompt = BuildPrompt(req.messages);
    }
  } else {
    prompt = BuildPrompt(req.messages);
  }

  std::vector<llama_token> prompt_tokens;
  const llama_vocab* vocab = llama_model_get_vocab(model_);
  int n_prompt_guess = llama_tokenize(vocab, prompt.c_str(), static_cast<int>(prompt.size()), nullptr, 0, true, true);
  if (n_prompt_guess == 0) {
    if (err) *err = "llama_cpp: tokenize failed";
    return false;
  }
  const int n_prompt = n_prompt_guess < 0 ? -n_prompt_guess : n_prompt_guess;
  if (n_prompt <= 0) {
    if (err) *err = "llama_cpp: tokenize failed";
    return false;
  }
  prompt_tokens.resize(static_cast<size_t>(n_prompt));
  const int n_tok = llama_tokenize(vocab, prompt.c_str(), static_cast<int>(prompt.size()), prompt_tokens.data(),
                                   static_cast<int>(prompt_tokens.size()), true, true);
  if (n_tok <= 0) {
    if (err) *err = "llama_cpp: tokenize failed";
    return false;
  }
  prompt_tokens.resize(static_cast<size_t>(n_tok));

  const uint32_t n_ctx = llama_n_ctx(ctx_);
  const auto gen_cfg = LoadLlamaRuntimeConfigFromEnv();
  int max_new_tokens_requested = gen_cfg.max_new_tokens.has_value() ? gen_cfg.max_new_tokens.value() : 2048;
  if (req.max_tokens.has_value() && req.max_tokens.value() > 0) {
    max_new_tokens_requested = req.max_tokens.value();
  }
  const int penalty_last_n = gen_cfg.penalty_last_n.has_value() ? gen_cfg.penalty_last_n.value() : 64;
  const float penalty_repeat = gen_cfg.penalty_repeat.has_value() ? gen_cfg.penalty_repeat.value() : 1.1f;
  const float temperature = gen_cfg.temperature.has_value() ? gen_cfg.temperature.value() : 0.0f;
  const float top_p = gen_cfg.top_p.has_value() ? gen_cfg.top_p.value() : 0.0f;
  const int32_t seed = gen_cfg.seed.has_value() ? gen_cfg.seed.value() : static_cast<int32_t>(LLAMA_DEFAULT_SEED);

  if (n_ctx > 0) {
    size_t reserve = 0;
    if (max_new_tokens_requested > 0 && n_ctx > 1) {
      reserve = std::min(static_cast<size_t>(max_new_tokens_requested), static_cast<size_t>(n_ctx - 1));
    }
    size_t keep = static_cast<size_t>(n_ctx);
    if (reserve > 0) {
      keep = static_cast<size_t>(n_ctx) - reserve;
    }
    if (keep >= static_cast<size_t>(n_ctx)) keep = static_cast<size_t>(n_ctx > 1 ? n_ctx - 1 : 0);
    if (keep == 0 && !prompt_tokens.empty()) keep = 1;
    if (keep == 0) {
      if (err) *err = "llama_cpp: prompt too long";
      return false;
    }
    if (prompt_tokens.size() > keep) {
      const size_t drop = prompt_tokens.size() - keep;
      prompt_tokens.erase(prompt_tokens.begin(), prompt_tokens.begin() + static_cast<std::ptrdiff_t>(drop));
    }
  }

  int max_new_tokens = max_new_tokens_requested;
  if (n_ctx > 0) {
    const size_t used = prompt_tokens.size();
    const size_t avail = used < static_cast<size_t>(n_ctx) ? static_cast<size_t>(n_ctx) - used : 0;
    if (max_new_tokens < 0) max_new_tokens = 0;
    max_new_tokens = std::min(max_new_tokens, static_cast<int>(avail));
  }

  auto sparams = llama_sampler_chain_default_params();
  sparams.no_perf = true;
  llama_sampler* sampler = llama_sampler_chain_init(sparams);
  if (!sampler) {
    if (err) *err = "llama_cpp: failed to init sampler";
    return false;
  }
  llama_sampler_chain_add(sampler, llama_sampler_init_penalties(penalty_last_n, penalty_repeat, 0.0f, 0.0f));
  if (temperature > 0.0f) {
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
  }
  if (top_p > 0.0f && top_p < 1.0f) {
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
  }
  if (temperature <= 0.0f && !(top_p > 0.0f && top_p < 1.0f)) {
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
  } else {
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(static_cast<uint32_t>(seed)));
  }

  if (penalty_last_n != 0 && !prompt_tokens.empty()) {
    const size_t n_accept =
        (penalty_last_n < 0) ? prompt_tokens.size() : std::min(prompt_tokens.size(), static_cast<size_t>(penalty_last_n));
    for (size_t i = prompt_tokens.size() - n_accept; i < prompt_tokens.size(); i++) {
      llama_sampler_accept(sampler, prompt_tokens[i]);
    }
  }

  llama_batch last_batch{};
  const uint32_t n_batch_u = llama_n_batch(ctx_);
  const size_t n_batch = n_batch_u > 0 ? static_cast<size_t>(n_batch_u) : 512;
  if (llama_model_has_encoder(model_)) {
    for (size_t start = 0; start < prompt_tokens.size();) {
      const size_t chunk = std::min(n_batch, prompt_tokens.size() - start);
      llama_batch batch = llama_batch_get_one(prompt_tokens.data() + start, static_cast<int32_t>(chunk));
      batch.logits = nullptr;
      const int32_t rc = llama_encode(ctx_, batch);
      if (rc != 0) {
        llama_sampler_free(sampler);
        if (err) *err = "llama_cpp: encode failed (code " + std::to_string(rc) + ")";
        return false;
      }
      start += chunk;
    }

    llama_token decoder_start_token_id = llama_model_decoder_start_token(model_);
    if (decoder_start_token_id == LLAMA_TOKEN_NULL) decoder_start_token_id = llama_vocab_bos(vocab);
    static thread_local std::vector<llama_token> token_buf(1);
    token_buf[0] = decoder_start_token_id;
    last_batch = llama_batch_get_one(token_buf.data(), 1);
    last_batch.logits = nullptr;
    const int32_t rc = llama_decode(ctx_, last_batch);
    if (rc != 0) {
      llama_sampler_free(sampler);
      if (err) *err = "llama_cpp: decode failed (code " + std::to_string(rc) + ")";
      return false;
    }
  } else {
    for (size_t start = 0; start < prompt_tokens.size();) {
      const size_t chunk = std::min(n_batch, prompt_tokens.size() - start);
      llama_batch batch = llama_batch_get_one(prompt_tokens.data() + start, static_cast<int32_t>(chunk));
      batch.logits = nullptr;
      const int32_t rc = llama_decode(ctx_, batch);
      if (rc != 0) {
        llama_sampler_free(sampler);
        if (err) *err = "llama_cpp: decode failed (code " + std::to_string(rc) + ")";
        return false;
      }
      last_batch = batch;
      start += chunk;
    }
  }

  std::string out_acc;
  std::string finish_reason = "stop";
  std::vector<llama_token> gen_tokens;
  gen_tokens.reserve(static_cast<size_t>(std::max(0, max_new_tokens)));
  llama_token last_tok = LLAMA_TOKEN_NULL;
  int last_tok_run = 0;
  for (int i = 0; i < max_new_tokens; i++) {
    const int32_t sample_i = std::max<int32_t>(0, last_batch.n_tokens - 1);
    llama_token next = llama_sampler_sample(sampler, ctx_, sample_i);
    llama_sampler_accept(sampler, next);
    if (llama_vocab_is_eog(vocab, next)) break;

    gen_tokens.push_back(next);
    if (next == last_tok) {
      last_tok_run++;
    } else {
      last_tok = next;
      last_tok_run = 1;
    }
    if (last_tok_run >= 32) break;
    if (gen_tokens.size() >= 64) {
      for (size_t w : {static_cast<size_t>(4), static_cast<size_t>(8), static_cast<size_t>(16), static_cast<size_t>(32)}) {
        if (gen_tokens.size() < w * 2) continue;
        const size_t a0 = gen_tokens.size() - w;
        const size_t b0 = gen_tokens.size() - w * 2;
        bool same = true;
        for (size_t k = 0; k < w; k++) {
          if (gen_tokens[b0 + k] != gen_tokens[a0 + k]) {
            same = false;
            break;
          }
        }
        if (same) {
          i = max_new_tokens;
          break;
        }
      }
    }

    auto piece = TokenToPiece(vocab, next);
    if (!piece.empty()) {
      out_acc += piece;
      auto stop_at = [&](const std::string& s) -> bool {
        return !s.empty() && out_acc.size() >= s.size() && out_acc.compare(out_acc.size() - s.size(), s.size(), s) == 0;
      };
      if (stop_at("\nUser:") || stop_at("\nUSER:") || stop_at("\nAssistant:") || stop_at("\nASSISTANT:") || stop_at("USER:") ||
          stop_at("ASSISTANT:")) {
        break;
      }
      on_delta(piece);
    }

    static thread_local std::vector<llama_token> next_token_buf(1);
    next_token_buf[0] = next;
    last_batch = llama_batch_get_one(next_token_buf.data(), 1);
    last_batch.logits = nullptr;
    if (n_ctx > 0) {
      const size_t used = prompt_tokens.size() + gen_tokens.size();
      if (used >= static_cast<size_t>(n_ctx)) {
        finish_reason = "length";
        break;
      }
    }
    const int32_t rc = llama_decode(ctx_, last_batch);
    if (rc != 0) {
      llama_sampler_free(sampler);
      if (err) *err = "llama_cpp: decode failed (code " + std::to_string(rc) + ")";
      return false;
    }
  }

  if (finish_reason == "stop" && max_new_tokens > 0 && gen_tokens.size() >= static_cast<size_t>(max_new_tokens)) {
    finish_reason = "length";
  }

  llama_sampler_free(sampler);
  std::cout << "[llama_cpp] finish_reason=" << finish_reason << " prompt_tokens=" << prompt_tokens.size()
            << " gen_tokens=" << gen_tokens.size() << " n_ctx=" << n_ctx << " max_new_tokens=" << max_new_tokens << "\n";
  on_done(finish_reason);

  if (gen_cfg.unload_after_chat.has_value() && gen_cfg.unload_after_chat.value()) {
    if (ctx_) {
      llama_free(ctx_);
      ctx_ = nullptr;
    }
    if (model_) {
      llama_model_free(model_);
      model_ = nullptr;
    }
    loaded_model_path_.clear();
  }

  return true;
}

}  // namespace runtime
