#pragma once

#include <string>
#include <vector>
#include <optional>
#include <nlohmann/json.hpp>

namespace llama_agent {

struct Message {
    std::string role;      // "system", "user", "assistant", "tool"
    std::string content;
    std::optional<std::string> toolCallId;
    std::optional<nlohmann::json> toolCalls;
};

struct Conversation {
    std::vector<Message> messages;
    
    void addSystemMessage(const std::string& content);
    void addUserMessage(const std::string& content);
    void addAssistantMessage(const std::string& content);
    void addToolResult(const std::string& toolCallId, const nlohmann::json& result);
    void addAssistantMessageWithToolCalls(const std::string& content, 
                                          const std::vector<nlohmann::json>& toolCalls);
    
    std::string toPrompt() const;
    std::string toJson() const;
    void clear();
    void truncate(size_t maxTokens);
    size_t estimateTokenCount() const;
    size_t getMessageCount() const;
    const std::vector<Message>& getMessages() const;
    std::optional<Message> getLastMessage() const;
    
private:
    size_t estimateTokenCountFrom(size_t startIdx) const;
};

} // namespace llama_agent
