#include <format>
#include <iostream>

#include "llama_agent/agent_runtime.hpp"
#include "llama_agent/llama_wrapper.hpp"
#include "llama_agent/tool_manager.hpp"
#include "llama_agent/http_server.hpp"

using namespace llama_agent;

int main(int argc, char* argv[]) {
    try {
        // Default configuration
        LlamaConfig llamaConfig;
        llamaConfig.modelPath = argc > 1 ? argv[1] : "model.gguf";
        llamaConfig.contextSize = 4096;
        llamaConfig.gpuLayers = 0;
        
        ServerConfig serverConfig;
        serverConfig.host = "0.0.0.0";
        serverConfig.port = 8080;
        
        std::cout << std::format("Loading model: {}\n", llamaConfig.modelPath);
        
        // Initialize components
        auto llama = std::make_unique<LlamaWrapper>(llamaConfig);
        auto tools = std::make_unique<ToolManager>();
        
        AgentConfig agentConfig;
        agentConfig.systemPrompt = "You are a helpful assistant with tool calling capabilities.";
        agentConfig.enableToolUse = true;
        
        auto agent = std::make_shared<AgentRuntime>(
            std::move(llama),
            std::move(tools),
            agentConfig
        );
        
        // Create and start HTTP server
        HttpServer server(agent, serverConfig);
        
        std::cout << std::format(
            "Server started on http://{}:{}\n",
            serverConfig.host,
            serverConfig.port
        );
        std::cout << "Press Ctrl+C to stop\n";
        
        server.run();
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Error: {}\n", e.what());
        return 1;
    }
    
    return 0;
}
