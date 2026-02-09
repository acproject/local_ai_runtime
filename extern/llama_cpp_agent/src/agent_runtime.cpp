#include "llama_agent/agent_runtime.hpp"
#include <format>
#include <chrono>

namespace llama_agent {

AgentRuntime::AgentRuntime(std::unique_ptr<LlamaWrapper> llama,
                           std::unique_ptr<ToolManager> tools,
                           const AgentConfig& config)
    : llama_(std::move(llama))
    , tools_(std::move(tools))
    , config_(config) {
    
    // Initialize conversation with system prompt
    if (!config_.systemPrompt.empty()) {
        conversation_.addSystemMessage(config_.systemPrompt);
    } else {
        conversation_.addSystemMessage(buildSystemPrompt());
    }
}

std::expected<AgentResponse, std::string> AgentRuntime::processMessage(
    const std::string& userMessage) {
    
    resetIteration();
    updateState(AgentState::Thinking);
    conversation_.addUserMessage(userMessage);
    
    AgentResponse finalResponse;
    finalResponse.isComplete = false;
    
    while (shouldContinue()) {
        currentIteration_++;
        
        // Build prompt
        auto prompt = conversation_.toPrompt();
        if (config_.enableToolUse && tools_ && tools_->getToolCount() > 0) {
            prompt += buildToolUsePrompt();
        }
        
        // Generate with grammar if tools enabled
        std::optional<std::string> grammar;
        if (config_.enableToolUse && tools_) {
            auto tools = tools_->getTools();
            if (!tools.empty()) {
                auto result = grammarGen_.generateToolCallGrammar(tools);
                if (result) {
                    grammar = *result;
                }
            }
        }
        
        // Run inference with retry
        auto response = retryWithFallback(prompt, 0);
        if (!response) {
            updateState(AgentState::Error);
            return handleError(response.error(), userMessage);
        }
        
        // Parse tool calls
        auto toolCalls = parseToolCalls(*response);
        
        if (!toolCalls.empty()) {
            // Execute tools
            updateState(AgentState::ToolCall);
            conversation_.addAssistantMessage(*response);
            
            auto toolResults = executeTools(toolCalls);
            if (!toolResults) {
                updateState(AgentState::Error);
                return handleError(toolResults.error(), userMessage);
            }
            
            // Add tool results to conversation
            for (const auto& call : toolCalls) {
                conversation_.addToolResult(call.id, nlohmann::json::parse(
                    std::format("{{\"result\": \"{}\"}}", toolResults->substr(0, 100))
                ));
            }
            
        } else {
            // No tool calls, this is the final response
            updateState(AgentState::Responding);
            conversation_.addAssistantMessage(*response);
            
            finalResponse.content = *response;
            finalResponse.toolCalls = toolCalls;
            finalResponse.isComplete = true;
            updateState(AgentState::Idle);
            return finalResponse;
        }
    }
    
    // Max iterations reached
    updateState(AgentState::Error);
    return handleError("Maximum iterations reached", userMessage);
}

std::expected<AgentResponse, std::string> AgentRuntime::processMessageStream(
    const std::string& userMessage,
    std::function<void(const StreamChunk&)> callback) {
    
    // For now, delegate to non-streaming version and simulate streaming
    // Full streaming implementation would require changes to LlamaWrapper
    auto result = processMessage(userMessage);
    
    if (!result) {
        return result;
    }
    
    // Simulate streaming by breaking content into chunks
    if (callback && !result->content.empty()) {
        const size_t chunkSize = 4; // Characters per chunk
        for (size_t i = 0; i < result->content.size(); i += chunkSize) {
            StreamChunk chunk;
            chunk.content = result->content.substr(i, chunkSize);
            chunk.isFinished = (i + chunkSize >= result->content.size());
            callback(chunk);
        }
    }
    
    return result;
}

void AgentRuntime::resetConversation() {
    conversation_.clear();
    resetIteration();
    if (!config_.systemPrompt.empty()) {
        conversation_.addSystemMessage(config_.systemPrompt);
    } else {
        conversation_.addSystemMessage(buildSystemPrompt());
    }
}

void AgentRuntime::registerTool(const ToolDefinition& definition, 
                               ToolFunction function) {
    if (tools_) {
        tools_->registerTool(definition, std::move(function));
    }
}

nlohmann::json AgentRuntime::getToolsDescription() const {
    if (!tools_) {
        return nlohmann::json::array();
    }
    return tools_->generateToolDescriptions();
}

bool AgentRuntime::isReady() const {
    return llama_ && llama_->isLoaded();
}

std::expected<std::string, std::string> AgentRuntime::runInference(
    const std::string& prompt,
    const std::optional<std::string>& grammar) {
    
    if (!llama_) {
        return std::unexpected("Llama wrapper not initialized");
    }
    
    return llama_->generate(prompt, grammar, config_.maxTokensPerResponse);
}

std::vector<ToolCall> AgentRuntime::parseToolCalls(const std::string& response) {
    return toolCallParser_.parse(response);
}

std::expected<std::string, std::string> AgentRuntime::executeTool(const ToolCall& toolCall) {
    if (!tools_) {
        return std::unexpected("Tool manager not initialized");
    }
    
    if (!tools_->hasTool(toolCall.functionName)) {
        return std::unexpected(std::format("Tool '{}' not found", toolCall.functionName));
    }
    
    auto result = tools_->executeTool(toolCall.functionName, toolCall.arguments);
    
    if (result.contains("error")) {
        return std::unexpected(result["error"].get<std::string>());
    }
    
    return result.dump();
}

std::expected<std::string, std::string> AgentRuntime::executeTools(
    const std::vector<ToolCall>& toolCalls) {
    
    std::string combinedResults;
    
    for (const auto& call : toolCalls) {
        auto result = executeTool(call);
        if (!result) {
            return result;
        }
        combinedResults += std::format("[{}]: {}\n", call.functionName, *result);
    }
    
    return combinedResults;
}

std::expected<AgentResponse, std::string> AgentRuntime::handleError(
    const std::string& error,
    const std::string& userMessage) {
    
    AgentResponse response;
    response.isComplete = false;
    response.error = error;
    
    // Try to recover by providing a simple response
    if (config_.retryAttempts > 0 && currentIteration_ < config_.maxIterations) {
        auto fallback = retryWithFallback(userMessage, 1);
        if (fallback) {
            response.content = *fallback;
            response.isComplete = true;
            response.error = std::nullopt;
            updateState(AgentState::Idle);
            return response;
        }
    }
    
    return std::unexpected(error);
}

std::expected<std::string, std::string> AgentRuntime::retryWithFallback(
    const std::string& prompt,
    int attempt) {
    
    auto result = runInference(prompt);
    
    if (result) {
        return result;
    }
    
    if (attempt < config_.retryAttempts) {
        // Retry without grammar constraint
        return runInference(prompt, std::nullopt);
    }
    
    return result;
}

void AgentRuntime::updateState(AgentState newState) {
    state_ = newState;
}

std::string AgentRuntime::buildSystemPrompt() const {
    std::string prompt = "You are a helpful AI assistant.";
    
    if (config_.enableToolUse && tools_ && tools_->getToolCount() > 0) {
        prompt += " You have access to tools that can help answer user questions. "
                 "When you need to use a tool, respond with a JSON object containing "
                 "'tool_calls' array with the tool invocations.";
    }
    
    return prompt;
}

std::string AgentRuntime::buildToolUsePrompt() const {
    if (!tools_ || tools_->getToolCount() == 0) {
        return "";
    }
    
    std::string prompt = "\n\nAvailable tools:\n";
    auto tools = tools_->getTools();
    
    for (const auto& tool : tools) {
        prompt += std::format("- {}: {}\n", tool.name, tool.description);
    }
    
    prompt += "\nTo use a tool, respond with JSON in this format:\n";
    prompt += R"({
  "tool_calls": [
    {
      "id": "call_1",
      "type": "function",
      "function": {
        "name": "tool_name",
        "arguments": {"param": "value"}
      }
    }
  ]
})";
    
    return prompt;
}

} // namespace llama_agent
