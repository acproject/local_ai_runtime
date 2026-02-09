#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <expected>
#include <optional>
#include <nlohmann/json.hpp>

#include "llama_agent/conversation.hpp"
#include "llama_agent/tool_manager.hpp"
#include "llama_agent/llama_wrapper.hpp"
#include "llama_agent/gbnf_generator.hpp"
#include "llama_agent/tool_call_parser.hpp"

namespace llama_agent {

enum class AgentState {
    Idle,
    Thinking,
    ToolCall,
    Responding,
    Error,
    StreamThinking   // For streaming mode
};

struct AgentConfig {
    std::string systemPrompt;
    int maxIterations = 10;
    int maxTokensPerResponse = 512;
    float temperature = 0.7f;
    bool enableToolUse = true;
    bool enableStreaming = false;
    int retryAttempts = 3;
};

struct AgentResponse {
    std::string content;
    std::vector<ToolCall> toolCalls;
    bool isComplete = true;
    std::optional<std::string> error;
    int tokensUsed = 0;
};

struct StreamChunk {
    std::string content;
    bool isToolCall = false;
    ToolCall toolCall;
    bool isFinished = false;
};

/**
 * @brief Main agent runtime
 * 
 * Orchestrates the conversation flow, manages tool calling,
 * and handles the reasoning loop with error recovery.
 */
class AgentRuntime {
public:
    AgentRuntime(std::unique_ptr<LlamaWrapper> llama,
                 std::unique_ptr<ToolManager> tools,
                 const AgentConfig& config);
    ~AgentRuntime() = default;

    // Disable copy, enable move
    AgentRuntime(const AgentRuntime&) = delete;
    AgentRuntime& operator=(const AgentRuntime&) = delete;
    AgentRuntime(AgentRuntime&&) noexcept = default;
    AgentRuntime& operator=(AgentRuntime&&) noexcept = default;

    /**
     * @brief Process a user message
     * @param userMessage User input
     * @return Agent response
     */
    std::expected<AgentResponse, std::string> processMessage(
        const std::string& userMessage);

    /**
     * @brief Process conversation with streaming support
     * @param userMessage User input
     * @param callback Called for each chunk of output
     * @return Final response
     */
    std::expected<AgentResponse, std::string> processMessageStream(
        const std::string& userMessage,
        std::function<void(const StreamChunk&)> callback);

    /**
     * @brief Reset conversation history
     */
    void resetConversation();

    /**
     * @brief Get current conversation
     * @return Reference to conversation
     */
    const Conversation& getConversation() const { return conversation_; }

    /**
     * @brief Get current state
     * @return Current agent state
     */
    AgentState getState() const { return state_; }

    /**
     * @brief Register a new tool
     */
    void registerTool(const ToolDefinition& definition, 
                     ToolFunction function);

    /**
     * @brief Get available tools as JSON for system prompt
     * @return Tools description JSON
     */
    nlohmann::json getToolsDescription() const;

    /**
     * @brief Check if agent is ready
     * @return true if model loaded and ready
     */
    bool isReady() const;

private:
    std::unique_ptr<LlamaWrapper> llama_;
    std::unique_ptr<ToolManager> tools_;
    ToolCallParser toolCallParser_;
    GrammarGenerator grammarGen_;
    Conversation conversation_;
    AgentConfig config_;
    AgentState state_ = AgentState::Idle;
    int currentIteration_ = 0;

    // Core inference methods
    std::expected<std::string, std::string> runInference(
        const std::string& prompt,
        const std::optional<std::string>& grammar = std::nullopt);
    
    // Tool handling
    std::vector<ToolCall> parseToolCalls(const std::string& response);
    std::expected<std::string, std::string> executeTool(const ToolCall& toolCall);
    std::expected<std::string, std::string> executeTools(
        const std::vector<ToolCall>& toolCalls);
    
    // Error handling and recovery
    std::expected<AgentResponse, std::string> handleError(
        const std::string& error,
        const std::string& userMessage);
    
    std::expected<std::string, std::string> retryWithFallback(
        const std::string& prompt,
        int attempt);
    
    // State management
    void updateState(AgentState newState);
    void resetIteration() { currentIteration_ = 0; }
    bool shouldContinue() const { return currentIteration_ < config_.maxIterations; }
    
    // Prompt construction
    std::string buildSystemPrompt() const;
    std::string buildToolUsePrompt() const;
};

} // namespace llama_agent
