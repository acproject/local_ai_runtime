#include "llama_agent/gbnf_generator.hpp"
#include "llama_agent/tool_manager.hpp"
#include <format>
#include <sstream>

namespace llama_agent {

std::expected<std::string, std::string> GrammarGenerator::generateFromSchema(
    const nlohmann::json& jsonSchema) {
    try {
        rules_.clear();
        ruleCounter_ = 0;
        
        std::stringstream grammar;
        grammar << "root ::= " << convertSchemaToRule(jsonSchema) << "\n";
        
        // Add collected rules
        for (const auto& rule : rules_) {
            grammar << rule << "\n";
        }
        
        return grammar.str();
    } catch (const std::exception& e) {
        return std::unexpected(std::format("Failed to generate grammar: {}", e.what()));
    }
}

std::expected<std::string, std::string> GrammarGenerator::generateToolCallGrammar(
    const std::vector<ToolDefinition>& tools) {
    try {
        std::string grammar;

        grammar += "tool_calls ::= \"[\" ws tool_call_list? \"]\" ws\n";
        grammar += "tool_call_list ::= tool_call (\",\" ws tool_call)*\n";
        grammar += "tool_call ::= \"{\" ws id_pair \",\" ws name_pair \",\" ws arguments_pair ws \"}\" ws\n";
        grammar += "id_pair ::= \"\\\"id\\\"\" ws \":\" ws string\n";
        grammar += "name_pair ::= \"\\\"name\\\"\" ws \":\" ws function_name\n";
        grammar += "arguments_pair ::= \"\\\"arguments\\\"\" ws \":\" ws json_value\n\n";

        grammar += "function_name ::= ";
        if (tools.empty()) {
            grammar += "string";
        } else {
            grammar += "(";
            for (size_t i = 0; i < tools.size(); ++i) {
                if (i > 0) grammar += " | ";
                grammar += std::format("\\\"{}\\\"", tools[i].name);
            }
            grammar += ") ws";
        }
        grammar += "\n\n";

        grammar += R"(
string ::= "\"" char* "\"" ws
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
number ::= ("-"? [0-9]+) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
json_object ::= "{" ws (json_pair ("," ws json_pair)*)? "}" ws
json_pair ::= string ":" ws json_value
json_array ::= "[" ws (json_value ("," ws json_value)*)? "]" ws
json_value ::= json_object | json_array | string | number | ("true" | "false" | "null") ws
ws ::= [ \t\n]*
)";

        return grammar;
    } catch (const std::exception& e) {
        return std::unexpected(std::format("Failed to generate tool call grammar: {}", e.what()));
    }
}

std::string GrammarGenerator::convertSchemaToRule(const nlohmann::json& schema) {
    if (!schema.contains("type")) {
        return "value";
    }
    
    auto type = schema["type"].get<std::string>();
    
    if (type == "object") {
        return generateObjectRule(schema);
    } else if (type == "array") {
        return generateArrayRule(schema);
    } else if (type == "string") {
        return generateStringRule(schema);
    } else if (type == "number" || type == "integer") {
        return generateNumberRule(schema);
    } else if (type == "boolean") {
        return "(\"true\" | \"false\") ws";
    } else if (type == "null") {
        return "\"null\" ws";
    }
    
    return "value";
}

std::string GrammarGenerator::generatePrimitiveRule(const nlohmann::json& schema) {
    return convertSchemaToRule(schema);
}

std::string GrammarGenerator::generateObjectRule(const nlohmann::json& schema) {
    std::string rule;
    rule += "\"{\" ws ";
    
    if (schema.contains("properties")) {
        const auto& props = schema["properties"];
        bool first = true;
        
        for (const auto& [key, value] : props.items()) {
            if (!first) {
                rule += "\",\" ws ";
            }
            first = false;
            
            rule += std::format("\\\"{}\\\" : \" ws ", key);
            rule += convertSchemaToRule(value);
        }
    }
    
    rule += " \"}\" ws";
    return rule;
}

std::string GrammarGenerator::generateArrayRule(const nlohmann::json& schema) {
    if (schema.contains("items")) {
        auto itemRule = convertSchemaToRule(schema["items"]);
        return "\"[\" ws (" + itemRule + " (\",\" ws " + itemRule + ")*)? \"]\" ws";
    }
    return "\"[\" ws (value (\",\" ws value)*)? \"]\" ws";
}

std::string GrammarGenerator::generateStringRule(const nlohmann::json& schema) {
    if (schema.contains("enum")) {
        return generateEnumRule(schema);
    }
    return "string";
}

std::string GrammarGenerator::generateNumberRule(const nlohmann::json& schema) {
    return "number";
}

std::string GrammarGenerator::generateEnumRule(const nlohmann::json& schema) {
    if (!schema.contains("enum")) {
        return "";
    }
    
    const auto& enumValues = schema["enum"];
    std::vector<std::string> options;
    
    for (const auto& value : enumValues) {
        if (value.is_string()) {
            options.push_back(std::format("\\\"{}\\\"", value.get<std::string>()));
        } else if (value.is_number()) {
            options.push_back(value.dump());
        } else if (value.is_boolean()) {
            options.push_back(value.get<bool>() ? "true" : "false");
        } else if (value.is_null()) {
            options.push_back("null");
        }
    }
    
    if (options.empty()) {
        return "";
    }
    
    std::string result = "(" + options[0];
    for (size_t i = 1; i < options.size(); ++i) {
        result += " | " + options[i];
    }
    result += ")";
    
    return result;
}

} // namespace llama_agent
