#pragma once

#include <string>
#include <vector>
#include <memory>
#include <expected>
#include <optional>
#include <llama.h>

namespace llama_agent {

struct LlamaConfig {
    std::string modelPath;
    int contextSize = 4096;
    int gpuLayers = 0;
    int threads = 4;
};

/**
 * @brief Wrapper around llama.cpp C API
 * 
 * Provides a C++ interface to llama.cpp for text generation
 * with grammar-constrained output support.
 */
class LlamaWrapper {
public:
    explicit LlamaWrapper(const LlamaConfig& config);
    ~LlamaWrapper();

    // Disable copy and move
    LlamaWrapper(const LlamaWrapper&) = delete;
    LlamaWrapper& operator=(const LlamaWrapper&) = delete;
    LlamaWrapper(LlamaWrapper&&) = delete;
    LlamaWrapper& operator=(LlamaWrapper&&) = delete;

    /**
     * @brief Generate next token(s) from prompt
     * @param prompt Input prompt text
     * @param grammar Optional GBNF grammar for constrained generation
     * @param maxTokens Maximum tokens to generate
     * @return Generated text or error
     */
    std::expected<std::string, std::string> generate(
        const std::string& prompt,
        const std::optional<std::string>& grammar = std::nullopt,
        int maxTokens = 512);

    /**
     * @brief Tokenize text
     * @param text Input text
     * @return Vector of token IDs
     */
    std::vector<llama_token> tokenize(const std::string& text);

    /**
     * @brief Detokenize tokens to text
     * @param tokens Token IDs
     * @return Decoded text
     */
    std::string detokenize(const std::vector<llama_token>& tokens);

    /**
     * @brief Check if model is loaded
     * @return true if ready
     */
    bool isLoaded() const { return model_ != nullptr; }

    /**
     * @brief Get context size
     * @return Maximum context length
     */
    int getContextSize() const { return contextSize_; }

    /**
     * @brief Get vocabulary size
     * @return Number of tokens in vocabulary
     */
    int getVocabSize() const;

    /**
     * @brief Get model information
     * @return Model info string
     */
    std::string getModelInfo() const;

private:
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    llama_sampler* sampler_ = nullptr;
    int contextSize_ = 4096;
    
    void cleanup();
};

} // namespace llama_agent
