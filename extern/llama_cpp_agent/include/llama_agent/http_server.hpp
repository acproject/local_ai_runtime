#pragma once

#include <string>
#include <memory>
#include <functional>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include "llama_agent/agent_runtime.hpp"

namespace llama_agent {

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    int threads = 4;
};

/**
 * @brief HTTP API Server
 * 
 * Provides OpenAI-compatible REST API endpoints for the agent.
 */
class HttpServer {
public:
    HttpServer(std::shared_ptr<AgentRuntime> agent,
               const ServerConfig& config);
    ~HttpServer();

    // Disable copy and move
    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;
    HttpServer(HttpServer&&) = delete;
    HttpServer& operator=(HttpServer&&) = delete;

    /**
     * @brief Start the server (blocking)
     */
    void run();

    /**
     * @brief Stop the server
     */
    void stop();

    /**
     * @brief Check if server is running
     */
    bool isRunning() const;

private:
    std::shared_ptr<AgentRuntime> agent_;
    ServerConfig config_;
    httplib::Server server_;
    bool running_ = false;

    void setupRoutes();
    
    // OpenAI-compatible endpoints
    void handleChatCompletions(const httplib::Request& req, 
                               httplib::Response& res);
    void handleModels(const httplib::Request& req,
                     httplib::Response& res);
    void handleHealth(const httplib::Request& req,
                     httplib::Response& res);
    
    // Helper methods
    nlohmann::json parseRequest(const std::string& body);
    std::string createChatResponse(const AgentResponse& response,
                                   const std::string& model);
};

} // namespace llama_agent
