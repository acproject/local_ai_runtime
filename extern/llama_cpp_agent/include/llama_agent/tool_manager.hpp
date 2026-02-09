#pragma once

#include <optional>
#include <string>
#include <vector>
#include <functional>
#include <unordered_map>
#include <nlohmann/json.hpp>

namespace llama_agent {

struct ToolParameter {
    std::string name;
    std::string type;
    std::string description;
    bool required = false;
    nlohmann::json schema;
};

struct ToolDefinition {
    std::string name;
    std::string description;
    std::vector<ToolParameter> parameters;
    nlohmann::json jsonSchema;
};

using ToolFunction = std::function<nlohmann::json(const nlohmann::json&)>;

/**
 * @brief Manages available tools for the agent
 */
class ToolManager {
public:
    ToolManager() = default;
    ~ToolManager() = default;

    // Disable copy, enable move
    ToolManager(const ToolManager&) = delete;
    ToolManager& operator=(const ToolManager&) = delete;
    ToolManager(ToolManager&&) noexcept = default;
    ToolManager& operator=(ToolManager&&) noexcept = default;

    /**
     * @brief Register a new tool
     * @param definition Tool metadata and schema
     * @param function The actual tool implementation
     */
    void registerTool(const ToolDefinition& definition, ToolFunction function);

    /**
     * @brief Execute a tool call
     * @param toolName Name of the tool to execute
     * @param parameters JSON parameters for the tool
     * @return Tool execution result
     */
    nlohmann::json executeTool(const std::string& toolName, 
                                const nlohmann::json& parameters);

    /**
     * @brief Get list of all registered tools
     * @return Vector of tool definitions
     */
    std::vector<ToolDefinition> getTools() const;

    /**
     * @brief Get tool definition by name
     * @param name Tool name
     * @return Tool definition or nullopt if not found
     */
    std::optional<ToolDefinition> getTool(const std::string& name) const;

    /**
     * @brief Check if tool exists
     * @param name Tool name
     * @return true if tool exists
     */
    bool hasTool(const std::string& name) const;

    /**
     * @brief Generate JSON Schema for all tools
     * @return Combined JSON schema
     */
    nlohmann::json generateToolsSchema() const;

    /**
     * @brief Generate tool descriptions for LLM prompt
     * @return Tool descriptions as JSON
     */
    nlohmann::json generateToolDescriptions() const;

    /**
     * @brief Validate parameters against tool schema
     * @param toolName Tool name
     * @param parameters Parameters to validate
     * @return true if valid, false otherwise
     */
    bool validateParameters(const std::string& toolName,
                           const nlohmann::json& parameters) const;

    /**
     * @brief Get number of registered tools
     * @return Tool count
     */
    size_t getToolCount() const;

    /**
     * @brief Unregister a tool
     * @param name Tool name
     */
    void unregisterTool(const std::string& name);

    /**
     * @brief Clear all tools
     */
    void clear();

private:
    std::unordered_map<std::string, ToolDefinition> tools_;
    std::unordered_map<std::string, ToolFunction> functions_;
};

} // namespace llama_agent
