#include "llama_agent/tool_call_parser.hpp"
#include <format>
#include <regex>

namespace llama_agent {

std::vector<ToolCall> ToolCallParser::parse(const std::string& response) {
    // Try JSON format first
    try {
        auto json = nlohmann::json::parse(response);
        if (json.contains("tool_calls")) {
            return parseFromJson(json["tool_calls"]);
        }
    } catch (...) {
        // Not valid JSON or no tool_calls field
    }
    
    // Try to parse raw JSON string that might be embedded
    return parseFromJsonString(response);
}

std::vector<ToolCall> ToolCallParser::parseFromJson(const nlohmann::json& json) {
    std::vector<ToolCall> toolCalls;
    
    if (!json.is_array()) {
        return toolCalls;
    }
    
    for (const auto& call : json) {
        auto toolCall = parseSingleToolCall(call);
        if (toolCall && toolCall->isValid()) {
            toolCalls.push_back(*toolCall);
        }
    }
    
    return toolCalls;
}

std::vector<ToolCall> ToolCallParser::parseFromJsonString(const std::string& jsonStr) {
    std::vector<ToolCall> toolCalls;
    
    // Try to extract JSON array from the string
    // Look for patterns like [{"id": ..., "type": ..., "function": ...}]
    std::regex toolCallRegex(R"(\{[^{}]*"type"\s*:\s*"function"[^{}]*\})", std::regex::icase);
    std::sregex_iterator iter(jsonStr.begin(), jsonStr.end(), toolCallRegex);
    std::sregex_iterator end;
    
    while (iter != end) {
        std::smatch match = *iter;
        try {
            auto json = nlohmann::json::parse(match.str());
            auto toolCall = parseSingleToolCall(json);
            if (toolCall && toolCall->isValid()) {
                toolCalls.push_back(*toolCall);
            }
        } catch (...) {
            // Skip invalid JSON
        }
        ++iter;
    }
    
    return toolCalls;
}

bool ToolCallParser::hasToolCalls(const std::string& response) const {
    try {
        auto json = nlohmann::json::parse(response);
        return json.contains("tool_calls");
    } catch (...) {
        return false;
    }
}

std::string ToolCallParser::extractContent(const std::string& response) const {
    try {
        auto json = nlohmann::json::parse(response);
        if (json.contains("content")) {
            return json["content"].get<std::string>();
        }
    } catch (...) {
        // Return as-is if not JSON
    }
    
    return response;
}

std::optional<ToolCall> ToolCallParser::parseSingleToolCall(const nlohmann::json& callJson) {
    ToolCall call;
    
    // Extract ID
    if (callJson.contains("id")) {
        call.id = callJson["id"].get<std::string>();
    } else {
        call.id = generateToolCallId();
    }
    
    // Extract type
    if (callJson.contains("type")) {
        call.type = callJson["type"].get<std::string>();
    } else {
        call.type = "function"; // Default
    }
    
    // Extract function details
    if (callJson.contains("function")) {
        const auto& func = callJson["function"];
        if (func.contains("name")) {
            call.functionName = func["name"].get<std::string>();
        }
        if (func.contains("arguments")) {
            call.arguments = func["arguments"];
        }
    }
    
    // Alternative format: direct name and arguments
    if (call.functionName.empty() && callJson.contains("name")) {
        call.functionName = callJson["name"].get<std::string>();
    }
    if (call.arguments.empty() && callJson.contains("arguments")) {
        call.arguments = callJson["arguments"];
    }
    if (call.arguments.empty() && callJson.contains("parameters")) {
        call.arguments = callJson["parameters"];
    }
    
    if (call.isValid()) {
        return call;
    }
    
    return std::nullopt;
}

std::string ToolCallParser::generateToolCallId() {
    return std::format("call_{}", ++idCounter_);
}

} // namespace llama_agent
