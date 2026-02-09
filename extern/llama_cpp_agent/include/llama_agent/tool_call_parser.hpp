#pragma once

#include <string>
#include <vector>
#include <optional>
#include <nlohmann/json.hpp>

namespace llama_agent {

/**
 * @brief Represents a single tool call from LLM
 */
struct ToolCall {
    std::string id;
    std::string type;           // "function"
    std::string functionName;
    nlohmann::json arguments;
    
    bool isValid() const {
        return !id.empty() && type == "function" && !functionName.empty();
    }
};

/**
 * @brief Parses tool calls from LLM response
 * 
 * Supports both JSON format and XML-like format
 */
class ToolCallParser {
public:
    ToolCallParser() = default;
    ~ToolCallParser() = default;

    /**
     * @brief Parse tool calls from LLM response text
     * @param response LLM generated text
     * @return Vector of parsed tool calls (empty if no tool calls found)
     */
    std::vector<ToolCall> parse(const std::string& response);

    /**
     * @brief Parse tool calls from JSON format
     * @param json JSON containing tool_calls array
     * @return Vector of tool calls
     */
    std::vector<ToolCall> parseFromJson(const nlohmann::json& json);

    /**
     * @brief Parse tool calls from raw JSON string
     * @param jsonStr JSON string
     * @return Vector of tool calls
     */
    std::vector<ToolCall> parseFromJsonString(const std::string& jsonStr);

    /**
     * @brief Check if response contains tool calls
     * @param response LLM response
     * @return true if tool calls detected
     */
    bool hasToolCalls(const std::string& response) const;

    /**
     * @brief Extract content without tool calls
     * @param response Original response
     * @return Clean content
     */
    std::string extractContent(const std::string& response) const;

private:
    std::optional<ToolCall> parseSingleToolCall(const nlohmann::json& callJson);
    std::string generateToolCallId();
    
    int idCounter_ = 0;
};

} // namespace llama_agent
