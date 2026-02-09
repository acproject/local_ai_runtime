#include "llama_agent/tool_manager.hpp"
#include <format>
#include <ranges>

namespace llama_agent {

void ToolManager::registerTool(const ToolDefinition& definition, ToolFunction function) {
    tools_[definition.name] = definition;
    functions_[definition.name] = std::move(function);
}

void ToolManager::unregisterTool(const std::string& name) {
    tools_.erase(name);
    functions_.erase(name);
}

nlohmann::json ToolManager::executeTool(const std::string& toolName,
                                        const nlohmann::json& parameters) {
    auto it = functions_.find(toolName);
    if (it == functions_.end()) {
        return nlohmann::json{
            {"error", std::format("Tool not found: {}", toolName)},
            {"success", false}
        };
    }
    
    // Validate parameters before execution
    if (!validateParameters(toolName, parameters)) {
        return nlohmann::json{
            {"error", "Invalid parameters for tool: " + toolName},
            {"success", false}
        };
    }
    
    try {
        auto result = it->second(parameters);
        result["success"] = true;
        return result;
    } catch (const std::exception& e) {
        return nlohmann::json{
            {"error", std::format("Tool execution failed: {}", e.what())},
            {"success", false}
        };
    }
}

std::vector<ToolDefinition> ToolManager::getTools() const {
    std::vector<ToolDefinition> result;
    result.reserve(tools_.size());
    
    // Use C++23 ranges to transform map values to vector
    auto values = tools_ | std::views::values;
    result.assign(values.begin(), values.end());
    
    return result;
}

std::optional<ToolDefinition> ToolManager::getTool(const std::string& name) const {
    auto it = tools_.find(name);
    if (it != tools_.end()) {
        return it->second;
    }
    return std::nullopt;
}

bool ToolManager::hasTool(const std::string& name) const {
    return tools_.contains(name);
}

nlohmann::json ToolManager::generateToolsSchema() const {
    // OpenAI-compatible tools schema
    nlohmann::json tools = nlohmann::json::array();
    
    for (const auto& [name, definition] : tools_) {
        nlohmann::json tool;
        tool["type"] = "function";
        tool["function"] = definition.jsonSchema;
        tools.push_back(tool);
    }
    
    return tools;
}

nlohmann::json ToolManager::generateToolDescriptions() const {
    nlohmann::json descriptions = nlohmann::json::array();
    
    for (const auto& [name, definition] : tools_) {
        nlohmann::json desc;
        desc["name"] = definition.name;
        desc["description"] = definition.description;
        desc["parameters"] = nlohmann::json::object();
        
        for (const auto& param : definition.parameters) {
            nlohmann::json paramJson;
            paramJson["type"] = param.type;
            paramJson["description"] = param.description;
            paramJson["required"] = param.required;
            desc["parameters"][param.name] = paramJson;
        }
        
        descriptions.push_back(desc);
    }
    
    return descriptions;
}

bool ToolManager::validateParameters(const std::string& toolName,
                                    const nlohmann::json& parameters) const {
    auto toolOpt = getTool(toolName);
    if (!toolOpt) return false;
    
    const auto& tool = *toolOpt;
    
    // Check required parameters
    for (const auto& param : tool.parameters) {
        if (param.required && !parameters.contains(param.name)) {
            return false;
        }
    }
    
    // Check parameter types
    for (const auto& [key, value] : parameters.items()) {
        auto it = std::ranges::find_if(tool.parameters,
            [&key](const ToolParameter& p) { return p.name == key; });
        
        if (it == tool.parameters.end()) {
            return false; // Unknown parameter
        }
        
        // Basic type checking
        if (it->type == "string" && !value.is_string()) {
            return false;
        } else if (it->type == "number" && !value.is_number()) {
            return false;
        } else if (it->type == "integer" && !value.is_number_integer()) {
            return false;
        } else if (it->type == "boolean" && !value.is_boolean()) {
            return false;
        } else if (it->type == "array" && !value.is_array()) {
            return false;
        } else if (it->type == "object" && !value.is_object()) {
            return false;
        }
    }
    
    return true;
}

size_t ToolManager::getToolCount() const {
    return tools_.size();
}

void ToolManager::clear() {
    tools_.clear();
    functions_.clear();
}

} // namespace llama_agent
