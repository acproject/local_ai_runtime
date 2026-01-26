#pragma once

#include "config.hpp"
#include "providers/provider.hpp"

#include <nlohmann/json.hpp>
#include <mutex>

namespace runtime {

class OllamaProvider : public IProvider {
 public:
  explicit OllamaProvider(HttpEndpoint endpoint);

  void Start() override;
  void Stop() override;
  std::optional<nlohmann::json> GetPs(std::string* err);

  std::string Name() const override;
  std::vector<ModelInfo> ListModels(std::string* err) override;
  std::optional<std::vector<double>> Embeddings(const std::string& model, const std::string& input, std::string* err) override;

  std::optional<ChatResponse> ChatOnce(const ChatRequest& req, std::string* err) override;
  bool ChatStream(const ChatRequest& req,
                  const std::function<void(const std::string&)>& on_delta,
                  const std::function<void(const std::string& finish_reason)>& on_done,
                  std::string* err) override;

 private:
  HttpEndpoint endpoint_;
  std::mutex mu_;
  std::string last_model_;
};

}  // namespace runtime
