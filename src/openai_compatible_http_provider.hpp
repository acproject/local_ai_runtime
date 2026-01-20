#pragma once

#include "config.hpp"
#include "providers/provider.hpp"

#include <string>

namespace runtime {

class OpenAiCompatibleHttpProvider : public IProvider {
 public:
  OpenAiCompatibleHttpProvider(std::string name, HttpEndpoint endpoint);

  std::string Name() const override;
  std::vector<ModelInfo> ListModels(std::string* err) override;
  std::optional<std::vector<double>> Embeddings(const std::string& model, const std::string& input, std::string* err) override;
  std::optional<ChatResponse> ChatOnce(const ChatRequest& req, std::string* err) override;
  bool ChatStream(const ChatRequest& req,
                  const std::function<void(const std::string&)>& on_delta,
                  const std::function<void()>& on_done,
                  std::string* err) override;

 private:
  std::string name_;
  HttpEndpoint endpoint_;
};

}  // namespace runtime

