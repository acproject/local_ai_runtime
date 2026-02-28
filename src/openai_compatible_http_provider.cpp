#include "openai_compatible_http_provider.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

namespace runtime {
namespace {

static bool StartsWith(const std::string& s, const std::string& prefix) {
  return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

static bool EndsWith(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static std::unique_ptr<httplib::Client> MakeClient(const HttpEndpoint& ep) {
  auto cli = std::make_unique<httplib::Client>(ep.host, ep.port);
  cli->set_connection_timeout(5);
  cli->set_read_timeout(300);
  cli->set_write_timeout(30);
  const auto& auth = CurrentRequestAuthHeaders();
  if (!auth.empty()) {
    httplib::Headers headers;
    for (const auto& kv : auth) {
      headers.emplace(kv.first, kv.second);
    }
    cli->set_default_headers(std::move(headers));
  }
  return cli;
}

static std::string JoinPath(const std::string& base, const std::string& path) {
  if (base.empty()) return path;

  std::string b = base;
  std::string p = path;

  if (EndsWith(b, "/v1/")) b.pop_back();
  if (EndsWith(b, "/v1") && StartsWith(p, "/v1/")) p = p.substr(3);

  if (p.empty()) return b;
  if (b.back() == '/' && p.front() == '/') return b + p.substr(1);
  if (b.back() != '/' && p.front() != '/') return b + "/" + p;
  return b + p;
}

}  // namespace

OpenAiCompatibleHttpProvider::OpenAiCompatibleHttpProvider(std::string name, HttpEndpoint endpoint)
    : name_(std::move(name)), endpoint_(std::move(endpoint)) {}

void OpenAiCompatibleHttpProvider::Stop() {
  if (name_ != "vllm") return;

  const char* env_path = std::getenv("VLLM_STOP_PATH");
  if (!env_path || env_path[0] == '\0') env_path = std::getenv("VLLM_UNLOAD_PATH");
  if (!env_path || env_path[0] == '\0') return;

  std::string model;
  {
    std::lock_guard<std::mutex> lock(mu_);
    model = last_model_;
    last_model_.clear();
  }

  std::string path = env_path;
  if (!path.empty() && path.front() != '/') path = "/" + path;

  auto cli = MakeClient(endpoint_);
  nlohmann::json j = nlohmann::json::object();
  if (!model.empty()) j["model"] = model;
  auto res = cli->Post(JoinPath(endpoint_.base_path, path), j.dump(), "application/json");
  if (!res) {
    std::cout << "[vllm] stop request failed model=" << model << "\n";
    return;
  }
  std::cout << "[vllm] stop request status=" << res->status << " model=" << model << "\n";
}

std::string OpenAiCompatibleHttpProvider::Name() const {
  return name_;
}

std::vector<ModelInfo> OpenAiCompatibleHttpProvider::ListModels(std::string* err) {
  auto cli = MakeClient(endpoint_);
  auto res = cli->Get(JoinPath(endpoint_.base_path, "/v1/models"));
  if (!res) {
    if (err) *err = name_ + ": failed to connect";
    return {};
  }
  if (res->status < 200 || res->status >= 300) {
    if (err) *err = name_ + ": /v1/models http " + std::to_string(res->status);
    return {};
  }
  auto j = nlohmann::json::parse(res->body, nullptr, false);
  if (j.is_discarded() || !j.contains("data") || !j["data"].is_array()) {
    if (err) *err = name_ + ": invalid json from /v1/models";
    return {};
  }
  std::vector<ModelInfo> out;
  for (const auto& it : j["data"]) {
    if (!it.is_object()) continue;
    ModelInfo m;
    if (it.contains("id") && it["id"].is_string()) m.id = it["id"].get<std::string>();
    if (it.contains("owned_by") && it["owned_by"].is_string()) m.owned_by = it["owned_by"].get<std::string>();
    if (m.owned_by.empty()) m.owned_by = name_;
    if (!m.id.empty()) out.push_back(std::move(m));
  }
  return out;
}

std::optional<std::vector<double>> OpenAiCompatibleHttpProvider::Embeddings(const std::string& model,
                                                                            const std::string& input,
                                                                            std::string* err) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    last_model_ = model;
  }
  auto cli = MakeClient(endpoint_);
  nlohmann::json j;
  j["model"] = model;
  j["input"] = input;
  auto res = cli->Post(JoinPath(endpoint_.base_path, "/v1/embeddings"), j.dump(), "application/json");
  if (!res) {
    if (err) *err = name_ + ": failed to connect";
    return std::nullopt;
  }
  if (res->status < 200 || res->status >= 300) {
    if (err) *err = name_ + ": /v1/embeddings http " + std::to_string(res->status);
    return std::nullopt;
  }
  auto jr = nlohmann::json::parse(res->body, nullptr, false);
  if (jr.is_discarded() || !jr.contains("data") || !jr["data"].is_array() || jr["data"].empty() ||
      !jr["data"][0].is_object() || !jr["data"][0].contains("embedding") || !jr["data"][0]["embedding"].is_array()) {
    if (err) *err = name_ + ": invalid json from /v1/embeddings";
    return std::nullopt;
  }
  std::vector<double> vec;
  for (const auto& v : jr["data"][0]["embedding"]) {
    if (v.is_number_float() || v.is_number_integer()) vec.push_back(v.get<double>());
  }
  return vec;
}

std::optional<ChatResponse> OpenAiCompatibleHttpProvider::ChatOnce(const ChatRequest& req, std::string* err) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    last_model_ = req.model;
  }
  auto cli = MakeClient(endpoint_);
  nlohmann::json j;
  j["model"] = req.model;
  j["stream"] = false;
  if (req.max_tokens.has_value() && req.max_tokens.value() > 0) {
    j["max_tokens"] = req.max_tokens.value();
  }
  if (req.temperature.has_value()) j["temperature"] = req.temperature.value();
  if (req.top_p.has_value()) j["top_p"] = req.top_p.value();
  if (req.min_p.has_value()) j["min_p"] = req.min_p.value();
  j["messages"] = nlohmann::json::array();
  for (const auto& m : req.messages) {
    j["messages"].push_back({{"role", m.role}, {"content", m.content}});
  }
  auto res = cli->Post(JoinPath(endpoint_.base_path, "/v1/chat/completions"), j.dump(), "application/json");
  if (!res) {
    if (err) *err = name_ + ": failed to connect";
    return std::nullopt;
  }
  if (res->status < 200 || res->status >= 300) {
    if (err) *err = name_ + ": /v1/chat/completions http " + std::to_string(res->status);
    return std::nullopt;
  }
  auto jr = nlohmann::json::parse(res->body, nullptr, false);
  if (jr.is_discarded() || !jr.contains("choices") || !jr["choices"].is_array() || jr["choices"].empty() ||
      !jr["choices"][0].is_object() || !jr["choices"][0].contains("message") || !jr["choices"][0]["message"].is_object() ||
      !jr["choices"][0]["message"].contains("content") || !jr["choices"][0]["message"]["content"].is_string()) {
    if (err) *err = name_ + ": invalid json from /v1/chat/completions";
    return std::nullopt;
  }
  ChatResponse out;
  out.model = req.model;
  out.content = jr["choices"][0]["message"]["content"].get<std::string>();
  if (jr["choices"][0].contains("finish_reason") && jr["choices"][0]["finish_reason"].is_string()) {
    out.finish_reason = jr["choices"][0]["finish_reason"].get<std::string>();
  }
  out.done = true;
  return out;
}

bool OpenAiCompatibleHttpProvider::ChatStream(const ChatRequest& req,
                                              const std::function<bool(const std::string&)>& on_delta,
                                              const std::function<void(const std::string& finish_reason)>& on_done,
                                              std::string* err) {
  auto once = ChatOnce(req, err);
  if (!once) return false;
  constexpr size_t kChunkSize = 64;
  for (size_t i = 0; i < once->content.size(); i += kChunkSize) {
    if (!on_delta(once->content.substr(i, kChunkSize))) return false;
  }
  on_done(once->finish_reason);
  return true;
}

}  // namespace runtime

