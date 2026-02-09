#include "llama_agent/conversation.hpp"
#include <format>
#include <numeric>

namespace llama_agent {

void Conversation::addSystemMessage(const std::string& content) {
    messages.push_back({"system", content, std::nullopt, std::nullopt});
}

void Conversation::addUserMessage(const std::string& content) {
    messages.push_back({"user", content, std::nullopt, std::nullopt});
}

void Conversation::addAssistantMessage(const std::string& content) {
    messages.push_back({"assistant", content, std::nullopt, std::nullopt});
}

void Conversation::addToolResult(const std::string& toolCallId, const nlohmann::json& result) {
    messages.push_back({"tool", result.dump(), toolCallId, std::nullopt});
}

void Conversation::addAssistantMessageWithToolCalls(
    const std::string& content, 
    const std::vector<nlohmann::json>& toolCalls) {
    Message msg;
    msg.role = "assistant";
    msg.content = content;
    msg.toolCalls = toolCalls;
    messages.push_back(std::move(msg));
}

std::string Conversation::toPrompt() const {
    std::string prompt;
    for (const auto& msg : messages) {
        if (msg.role == "system") {
            prompt += std::format("<|system|>\n{}\n", msg.content);
        } else if (msg.role == "user") {
            prompt += std::format("<|user|>\n{}\n", msg.content);
        } else if (msg.role == "assistant") {
            prompt += std::format("<|assistant|>\n{}\n", msg.content);
            // Include tool calls if present
            if (msg.toolCalls && !msg.toolCalls->empty()) {
                prompt += "<|tool_calls|>\n";
                for (const auto& call : *msg.toolCalls) {
                    prompt += call.dump() + "\n";
                }
            }
        } else if (msg.role == "tool") {
            prompt += std::format("<|tool|>\n{}\n", msg.content);
        }
    }
    prompt += "<|assistant|>\n";
    return prompt;
}

std::string Conversation::toJson() const {
    nlohmann::json j = nlohmann::json::array();
    for (const auto& msg : messages) {
        nlohmann::json msgJson;
        msgJson["role"] = msg.role;
        msgJson["content"] = msg.content;
        if (msg.toolCallId) {
            msgJson["tool_call_id"] = *msg.toolCallId;
        }
        if (msg.toolCalls) {
            msgJson["tool_calls"] = *msg.toolCalls;
        }
        j.push_back(msgJson);
    }
    return j.dump(2);
}

void Conversation::clear() {
    messages.clear();
}

void Conversation::truncate(size_t maxTokens) {
    // Keep system message and as many recent messages as possible
    if (messages.empty()) return;
    
    size_t currentTokens = estimateTokenCount();
    if (currentTokens <= maxTokens) return;
    
    // Always keep system message at index 0
    size_t startIdx = 1;
    while (startIdx < messages.size() && estimateTokenCountFrom(startIdx) > maxTokens) {
        ++startIdx;
    }
    
    if (startIdx > 1) {
        // Remove messages from index 1 to startIdx-1
        messages.erase(messages.begin() + 1, messages.begin() + startIdx);
    }
}

size_t Conversation::estimateTokenCount() const {
    return estimateTokenCountFrom(0);
}

size_t Conversation::estimateTokenCountFrom(size_t startIdx) const {
    size_t totalChars = 0;
    for (size_t i = startIdx; i < messages.size(); ++i) {
        totalChars += messages[i].content.size();
        // Add overhead for role markers and formatting
        totalChars += 20;
        if (messages[i].toolCalls) {
            for (const auto& call : *messages[i].toolCalls) {
                totalChars += call.dump().size();
            }
        }
    }
    // Rough estimation: ~4 chars per token on average
    return totalChars / 4;
}

size_t Conversation::getMessageCount() const {
    return messages.size();
}

const std::vector<Message>& Conversation::getMessages() const {
    return messages;
}

std::optional<Message> Conversation::getLastMessage() const {
    if (messages.empty()) {
        return std::nullopt;
    }
    return messages.back();
}

} // namespace llama_agent
