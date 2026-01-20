#pragma once

#include "providers/provider.hpp"

#include <mutex>
#include <string>

struct llama_context;
struct llama_model;

namespace runtime {

class LlamaCppProvider : public IProvider {
 public:
  explicit LlamaCppProvider(std::string model_path);
  ~LlamaCppProvider() override;

  std::string Name() const override;
  std::vector<ModelInfo> ListModels(std::string* err) override;
  std::optional<std::vector<double>> Embeddings(const std::string& model, const std::string& input, std::string* err) override;

  std::optional<ChatResponse> ChatOnce(const ChatRequest& req, std::string* err) override;
  bool ChatStream(const ChatRequest& req,
                  const std::function<void(const std::string&)>& on_delta,
                  const std::function<void()>& on_done,
                  std::string* err) override;

 private:
  bool EnsureLoaded(std::string* err);
  std::string BuildPrompt(const std::vector<ChatMessage>& messages) const;

  std::string model_path_;
  std::string model_id_;

  mutable std::mutex mu_;
  llama_model* model_ = nullptr;
  llama_context* ctx_ = nullptr;
};

}  // namespace runtime

