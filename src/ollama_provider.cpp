#include "ollama_provider.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <iostream>
#include <string>

namespace runtime {
namespace {

static std::unique_ptr<httplib::Client> MakeClient(const HttpEndpoint& ep) {
  auto cli = std::make_unique<httplib::Client>(ep.host, ep.port);
  cli->set_connection_timeout(5);
  cli->set_read_timeout(300);
  cli->set_write_timeout(30);
  return cli;
}

static std::string JoinPath(const std::string& base, const std::string& path) {
  if (base.empty()) return path;
  if (base.back() == '/' && !path.empty() && path.front() == '/') return base + path.substr(1);
  if (base.back() != '/' && !path.empty() && path.front() != '/') return base + "/" + path;
  return base + path;
}

static void LogPs(httplib::Client& cli, const HttpEndpoint& endpoint, const std::string& tag) {
  auto res = cli.Get(JoinPath(endpoint.base_path, "/api/ps"));
  if (!res) {
    std::cout << "[ollama] " << tag << " ps=failed\n";
    return;
  }
  std::cout << "[ollama] " << tag << " ps_status=" << res->status << " body=" << res->body << "\n";
}

}  // namespace

OllamaProvider::OllamaProvider(HttpEndpoint endpoint) : endpoint_(std::move(endpoint)) {}

void OllamaProvider::Start() {
  auto cli = MakeClient(endpoint_);
  LogPs(*cli, endpoint_, "start");
}

void OllamaProvider::Stop() {
  std::string model;
  {
    std::lock_guard<std::mutex> lock(mu_);
    model = last_model_;
    last_model_.clear();
  }
  if (model.empty()) return;
  auto cli = MakeClient(endpoint_);
  nlohmann::json j;
  j["model"] = model;
  j["prompt"] = "";
  j["stream"] = false;
  j["keep_alive"] = 0;
  auto res = cli->Post(JoinPath(endpoint_.base_path, "/api/generate"), j.dump(), "application/json");
  if (!res) {
    std::cout << "[ollama] unload failed model=" << model << "\n";
    return;
  }
  std::cout << "[ollama] unload model=" << model << " status=" << res->status << "\n";
  LogPs(*cli, endpoint_, "stop");
}

std::string OllamaProvider::Name() const {
  return "ollama";
}

std::vector<ModelInfo> OllamaProvider::ListModels(std::string* err) {
  auto cli = MakeClient(endpoint_);
  auto res = cli->Get(JoinPath(endpoint_.base_path, "/api/tags"));
  if (!res) {
    if (err) *err = "ollama: failed to connect";
    return {};
  }
  if (res->status < 200 || res->status >= 300) {
    if (err) *err = "ollama: /api/tags http " + std::to_string(res->status);
    return {};
  }

  std::vector<ModelInfo> out;
  auto j = nlohmann::json::parse(res->body, nullptr, false);
  if (j.is_discarded()) {
    if (err) *err = "ollama: invalid json from /api/tags";
    return {};
  }
  if (!j.contains("models") || !j["models"].is_array()) return {};
  for (const auto& m : j["models"]) {
    if (!m.is_object()) continue;
    ModelInfo info;
    if (m.contains("name") && m["name"].is_string()) info.id = m["name"].get<std::string>();
    info.owned_by = "ollama";
    if (!info.id.empty()) out.push_back(std::move(info));
  }
  return out;
}

std::optional<std::vector<double>> OllamaProvider::Embeddings(const std::string& model,
                                                              const std::string& input,
                                                              std::string* err) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    last_model_ = model;
  }
  auto cli = MakeClient(endpoint_);
  nlohmann::json j;
  j["model"] = model;
  j["prompt"] = input;
  auto res = cli->Post(JoinPath(endpoint_.base_path, "/api/embeddings"), j.dump(), "application/json");
  if (!res) {
    if (err) *err = "ollama: failed to connect";
    return std::nullopt;
  }
  if (res->status < 200 || res->status >= 300) {
    if (err) *err = "ollama: /api/embeddings http " + std::to_string(res->status);
    return std::nullopt;
  }
  auto jr = nlohmann::json::parse(res->body, nullptr, false);
  if (jr.is_discarded() || !jr.contains("embedding") || !jr["embedding"].is_array()) {
    if (err) *err = "ollama: invalid json from /api/embeddings";
    return std::nullopt;
  }
  std::vector<double> vec;
  vec.reserve(jr["embedding"].size());
  for (const auto& v : jr["embedding"]) {
    if (v.is_number_float() || v.is_number_integer()) vec.push_back(v.get<double>());
  }
  return vec;
}

std::optional<ChatResponse> OllamaProvider::ChatOnce(const ChatRequest& req, std::string* err) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    last_model_ = req.model;
  }
  auto cli = MakeClient(endpoint_);
  nlohmann::json j;
  j["model"] = req.model;
  j["stream"] = false;
  j["messages"] = nlohmann::json::array();
  for (const auto& m : req.messages) {
    nlohmann::json jm;
    jm["role"] = m.role;
    jm["content"] = m.content;
    j["messages"].push_back(std::move(jm));
  }
  auto res = cli->Post(JoinPath(endpoint_.base_path, "/api/chat"), j.dump(), "application/json");
  if (!res) {
    if (err) *err = "ollama: failed to connect";
    return std::nullopt;
  }
  if (res->status < 200 || res->status >= 300) {
    if (err) *err = "ollama: /api/chat http " + std::to_string(res->status);
    return std::nullopt;
  }
  auto jr = nlohmann::json::parse(res->body, nullptr, false);
  if (jr.is_discarded() || !jr.contains("message") || !jr["message"].is_object()) {
    if (err) *err = "ollama: invalid json from /api/chat";
    return std::nullopt;
  }
  ChatResponse out;
  out.model = req.model;
  if (jr["message"].contains("content") && jr["message"]["content"].is_string()) {
    out.content = jr["message"]["content"].get<std::string>();
  }
  if (jr.contains("done") && jr["done"].is_boolean()) out.done = jr["done"].get<bool>();
  return out;
}

bool OllamaProvider::ChatStream(const ChatRequest& req,
                                const std::function<void(const std::string&)>& on_delta,
                                const std::function<void()>& on_done,
                                std::string* err) {
  auto once = ChatOnce(req, err);
  if (!once) return false;

  constexpr size_t kChunkSize = 64;
  for (size_t i = 0; i < once->content.size(); i += kChunkSize) {
    on_delta(once->content.substr(i, kChunkSize));
  }
  on_done();
  return true;
}

}  // namespace runtime
