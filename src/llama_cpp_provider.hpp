#pragma once

#include "providers/provider.hpp"

#include <chat.h>

#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct llama_context;
struct llama_model;

namespace runtime {

class LlamaCppProvider : public IProvider {
 public:
  explicit LlamaCppProvider(std::string model_path);
  ~LlamaCppProvider() override;

  void Start() override;
  void Stop() override;

  std::string Name() const override;
  std::vector<ModelInfo> ListModels(std::string* err) override;
  std::optional<std::vector<double>> Embeddings(const std::string& model, const std::string& input, std::string* err) override;

  std::optional<ChatResponse> ChatOnce(const ChatRequest& req, std::string* err) override;
  bool ChatStream(const ChatRequest& req,
                  const std::function<bool(const std::string&)>& on_delta,
                  const std::function<void(const std::string& finish_reason)>& on_done,
                  std::string* err) override;

 private:
  void BuildModelIndex();
  std::optional<std::string> ResolveModelPath(const std::string& requested_model, std::string* err) const;
  bool EnsureLoaded(const std::string& model_path, std::string* err);
  struct SessionState {
    std::mutex mu;
    llama_context* ctx = nullptr;
    std::vector<int32_t> tokens;
    uint32_t n_ctx = 0;
  };

  struct SessionEntry {
    std::shared_ptr<SessionState> state;
    std::list<std::string>::iterator lru_it;
  };

  bool EnsureContext(SessionState* s, std::string* err);
  std::string BuildPrompt(const std::vector<ChatMessage>& messages) const;

  std::string model_root_;
  bool root_is_dir_ = false;
  std::unordered_map<std::string, std::string> model_paths_by_id_;
  std::vector<std::string> model_ids_;

  mutable std::mutex mu_;
  mutable std::shared_mutex model_mu_;
  std::string loaded_model_path_;
  size_t max_sessions_ = 16;
  llama_model* model_ = nullptr;
  std::shared_ptr<SessionState> default_session_;
  std::list<std::string> lru_;
  std::unordered_map<std::string, SessionEntry> sessions_;

  // Chat template support (cached for performance)
  common_chat_templates_ptr chat_templates_;
  bool use_jinja_ = true;  // default to jinja parser
};

}  // namespace runtime

