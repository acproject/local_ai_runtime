#include <gtest/gtest.h>
#include "llama_agent/tool_call_parser.hpp"

using namespace llama_agent;

class ToolCallParserTest : public ::testing::Test {
protected:
    ToolCallParser parser;
};

TEST_F(ToolCallParserTest, ParseEmptyResponse) {
    auto result = parser.parse("");
    EXPECT_TRUE(result.empty());
}

TEST_F(ToolCallParserTest, ParseSimpleResponse) {
    auto result = parser.parse("Hello, world!");
    EXPECT_TRUE(result.empty());
}

TEST_F(ToolCallParserTest, ParseToolCallJson) {
    std::string response = R"({
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": {"location": "Beijing"}
                }
            }
        ]
    })";
    
    auto result = parser.parse(response);
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].id, "call_1");
    EXPECT_EQ(result[0].functionName, "get_weather");
    EXPECT_EQ(result[0].arguments["location"], "Beijing");
}

TEST_F(ToolCallParserTest, ParseMultipleToolCalls) {
    std::string response = R"({
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": {"location": "Beijing"}
                }
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {
                    "name": "get_time",
                    "arguments": {}
                }
            }
        ]
    })";
    
    auto result = parser.parse(response);
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0].functionName, "get_weather");
    EXPECT_EQ(result[1].functionName, "get_time");
}

TEST_F(ToolCallParserTest, HasToolCalls) {
    std::string withTools = R"({"tool_calls": []})";
    std::string withoutTools = "Just a message";
    
    EXPECT_TRUE(parser.hasToolCalls(withTools));
    EXPECT_FALSE(parser.hasToolCalls(withoutTools));
}

TEST_F(ToolCallParserTest, ExtractContent) {
    std::string response = R"({"content": "Hello!"})";
    auto content = parser.extractContent(response);
    EXPECT_EQ(content, "Hello!");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
