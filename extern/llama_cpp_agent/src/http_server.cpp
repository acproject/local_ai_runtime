#include "llama_agent/http_server.hpp"
#include <format>
#include <iostream>

namespace llama_agent {

HttpServer::HttpServer(std::shared_ptr<AgentRuntime> agent,
                       const ServerConfig& config)
    : agent_(std::move(agent))
    , config_(config) {
    setupRoutes();
}

HttpServer::~HttpServer() {
    if (running_) {
        stop();
    }
}

void HttpServer::setupRoutes() {
    // Health check
    server_.Get("/health", [this](const httplib::Request& req, 
                                   httplib::Response& res) {
        handleHealth(req, res);
    });
    
    // Models endpoint
    server_.Get("/v1/models", [this](const httplib::Request& req,
                                      httplib::Response& res) {
        handleModels(req, res);
    });
    
    // Chat completions endpoint
    server_.Post("/v1/chat/completions", [this](const httplib::Request& req,
                                                 httplib::Response& res) {
        handleChatCompletions(req, res);
    });
}

void HttpServer::run() {
    running_ = true;
    server_.listen(config_.host.c_str(), config_.port);
}

void HttpServer::stop() {
    server_.stop();
    running_ = false;
}

bool HttpServer::isRunning() const {
    return running_;
}

void HttpServer::handleHealth(const httplib::Request& req,
                              httplib::Response& res) {
    nlohmann::json response = {
        {"status", "healthy"},
        {"version", "0.1.0"}
    };
    res.set_content(response.dump(), "application/json");
}

void HttpServer::handleModels(const httplib::Request& req,
                              httplib::Response& res) {
    nlohmann::json response = {
        {"object", "list"},
        {"data", nlohmann::json::array({
            {
                {"id", "llama-agent"},
                {"object", "model"},
                {"owned_by", "llama-cpp-agent"}
            }
        })}
    };
    res.set_content(response.dump(), "application/json");
}

void HttpServer::handleChatCompletions(const httplib::Request& req,
                                       httplib::Response& res) {
    try {
        auto body = parseRequest(req.body);
        
        if (!body.contains("messages")) {
            res.status = 400;
            res.set_content(R"({"error": "Missing messages field"})", 
                           "application/json");
            return;
        }
        
        // Extract the last user message
        std::string userMessage;
        for (const auto& msg : body["messages"]) {
            if (msg["role"] == "user") {
                userMessage = msg["content"].get<std::string>();
            }
        }
        
        if (userMessage.empty()) {
            res.status = 400;
            res.set_content(R"({"error": "No user message found"})",
                           "application/json");
            return;
        }
        
        // Process through agent
        auto result = agent_->processMessage(userMessage);
        if (!result) {
            res.status = 500;
            res.set_content(std::format(R"({{"error": "{}"}})", result.error()),
                           "application/json");
            return;
        }
        
        auto response = createChatResponse(*result, 
            body.value("model", "llama-agent"));
        res.set_content(response, "application/json");
        
    } catch (const std::exception& e) {
        res.status = 500;
        res.set_content(std::format(R"({{"error": "{}"}})", e.what()),
                       "application/json");
    }
}

nlohmann::json HttpServer::parseRequest(const std::string& body) {
    return nlohmann::json::parse(body);
}

std::string HttpServer::createChatResponse(const AgentResponse& response,
                                           const std::string& model) {
    nlohmann::json result = {
        {"id", "chatcmpl-" + std::to_string(std::time(nullptr))},
        {"object", "chat.completion"},
        {"created", std::time(nullptr)},
        {"model", model},
        {"choices", nlohmann::json::array({
            {
                {"index", 0},
                {"message", {
                    {"role", "assistant"},
                    {"content", response.content}
                }},
                {"finish_reason", "stop"}
            }
        })},
        {"usage", {
            {"prompt_tokens", 0},
            {"completion_tokens", 0},
            {"total_tokens", 0}
        }}
    };
    
    return result.dump();
}

} // namespace llama_agent
