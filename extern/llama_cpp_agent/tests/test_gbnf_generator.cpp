#include <gtest/gtest.h>
#include "llama_agent/gbnf_generator.hpp"

using namespace llama_agent;

class GBNFGeneratorTest : public ::testing::Test {
protected:
    GrammarGenerator generator;
};

TEST_F(GBNFGeneratorTest, GenerateFromSchema) {
    nlohmann::json schema = {
        {"type", "object"},
        {"properties", {
            {"name", {{"type", "string"}}},
            {"age", {{"type", "integer"}}}
        }},
        {"required", nlohmann::json::array({"name", "age"})}
    };
    
    auto result = generator.generateFromSchema(schema);
    EXPECT_TRUE(result.has_value());
    EXPECT_FALSE(result->empty());
}

TEST_F(GBNFGeneratorTest, GenerateToolCallGrammar) {
    std::vector<ToolDefinition> tools;
    ToolDefinition tool1;
    tool1.name = "get_weather";
    tool1.description = "Get weather information";
    tool1.jsonSchema = {
        {"type", "function"},
        {"function", {
            {"name", "get_weather"},
            {"parameters", {
                {"type", "object"},
                {"properties", {
                    {"location", {{"type", "string"}}}
                }}
            }}
        }}
    };
    tools.push_back(tool1);
    
    auto result = generator.generateToolCallGrammar(tools);
    EXPECT_TRUE(result.has_value());
    EXPECT_FALSE(result->empty());
}

TEST_F(GBNFGeneratorTest, EmptyToolList) {
    std::vector<ToolDefinition> emptyTools;
    auto result = generator.generateToolCallGrammar(emptyTools);
    EXPECT_TRUE(result.has_value());
    EXPECT_FALSE(result->empty());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
