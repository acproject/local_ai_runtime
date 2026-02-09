#pragma once

#include <vector>
#include <string>
#include <expected>
#include <nlohmann/json.hpp>

namespace llama_agent {

// Forward declarations
struct ToolDefinition;
struct GBNFGrammar;

/**
 * @brief Generates GBNF grammar from JSON Schema
 * 
 * This class converts JSON Schema definitions into GBNF grammars
 * that can be used by llama.cpp for constrained generation.
 */
class GrammarGenerator {
public:
    GrammarGenerator() = default;
    ~GrammarGenerator() = default;

    // Disable copy, enable move
    GrammarGenerator(const GrammarGenerator&) = delete;
    GrammarGenerator& operator=(const GrammarGenerator&) = delete;
    GrammarGenerator(GrammarGenerator&&) noexcept = default;
    GrammarGenerator& operator=(GrammarGenerator&&) noexcept = default;

    /**
     * @brief Generate GBNF grammar from JSON Schema
     * @param jsonSchema The JSON schema definition
     * @return Generated GBNF grammar or error
     */
    std::expected<std::string, std::string> generateFromSchema(
        const nlohmann::json& jsonSchema);

    /**
     * @brief Generate GBNF grammar for tool call format
     * @param tools List of available tools
     * @return Grammar that allows calling any of the tools
     */
    std::expected<std::string, std::string> generateToolCallGrammar(
        const std::vector<ToolDefinition>& tools);

private:
    std::string convertSchemaToRule(const nlohmann::json& schema);
    std::string generatePrimitiveRule(const nlohmann::json& schema);
    std::string generateObjectRule(const nlohmann::json& schema);
    std::string generateArrayRule(const nlohmann::json& schema);
    std::string generateStringRule(const nlohmann::json& schema);
    std::string generateNumberRule(const nlohmann::json& schema);
    std::string generateEnumRule(const nlohmann::json& schema);
    
    int ruleCounter_ = 0;
    std::vector<std::string> rules_;
};

} // namespace llama_agent
