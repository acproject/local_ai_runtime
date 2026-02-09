#include "llama_agent/llama_wrapper.hpp"
#include <format>
#include <cstring>
#include <vector>

namespace llama_agent {

LlamaWrapper::LlamaWrapper(const LlamaConfig& config) 
    : contextSize_(config.contextSize) {
    
    // Initialize model parameters
    llama_model_params modelParams = llama_model_default_params();
    modelParams.n_gpu_layers = config.gpuLayers;
    
    // Load model
    model_ = llama_load_model_from_file(config.modelPath.c_str(), modelParams);
    if (!model_) {
        throw std::runtime_error(std::format("Failed to load model from: {}", config.modelPath));
    }
    
    // Initialize context parameters
    llama_context_params ctxParams = llama_context_default_params();
    ctxParams.n_ctx = config.contextSize;
    ctxParams.n_threads = config.threads;
    ctxParams.n_threads_batch = config.threads;
    
    // Create context
    ctx_ = llama_new_context_with_model(model_, ctxParams);
    if (!ctx_) {
        llama_model_free(model_);
        model_ = nullptr;
        throw std::runtime_error("Failed to create context");
    }
    
    // Initialize sampler
    sampler_ = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (!sampler_) {
        cleanup();
        throw std::runtime_error("Failed to initialize sampler");
    }
    
    // Add default sampling settings
    llama_sampler_chain_add(sampler_, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(sampler_, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(sampler_, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
}

LlamaWrapper::~LlamaWrapper() {
    cleanup();
}

void LlamaWrapper::cleanup() {
    if (sampler_) {
        llama_sampler_free(sampler_);
        sampler_ = nullptr;
    }
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
}

std::expected<std::string, std::string> LlamaWrapper::generate(
    const std::string& prompt,
    const std::optional<std::string>& grammar,
    int maxTokens) {
    
    if (!isLoaded()) {
        return std::unexpected("Model not loaded");
    }
    
    try {
        // Tokenize prompt
        auto promptTokens = tokenize(prompt);
        if (promptTokens.empty()) {
            return std::unexpected("Failed to tokenize prompt");
        }
        
        // Check context size
        if (promptTokens.size() > static_cast<size_t>(contextSize_)) {
            return std::unexpected("Prompt too long for context window");
        }
        
        // Create sampler - use grammar if provided
        llama_sampler* genSampler = sampler_;
        if (grammar && !grammar->empty()) {
            // For now, log that grammar is provided but use default sampler
            // Grammar support requires additional setup
        }
        
        // Prepare batch for prompt
        llama_batch batch = llama_batch_init(static_cast<int32_t>(promptTokens.size()), 0, 1);
        
        // Add prompt tokens to batch
        for (size_t i = 0; i < promptTokens.size(); i++) {
            batch.token[i] = promptTokens[i];
            batch.pos[i] = static_cast<int32_t>(i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 0;
        }
        batch.logits[promptTokens.size() - 1] = 1; // Get logits for last token
        batch.n_tokens = static_cast<int32_t>(promptTokens.size());
        
        // Decode prompt
        if (llama_decode(ctx_, batch) != 0) {
            llama_batch_free(batch);
            return std::unexpected("Failed to decode prompt");
        }
        
        // Generate tokens
        std::string result;
        int nCur = batch.n_tokens;
        int32_t maxCtx = llama_n_ctx(ctx_);
        
        for (int i = 0; i < maxTokens && nCur < maxCtx; i++) {
            // Sample next token
            llama_token newToken = llama_sampler_sample(genSampler, ctx_, -1);
            
            // Check for end of generation
            if (llama_vocab_is_eog(llama_model_get_vocab(model_), newToken)) {
                break;
            }
            
            // Detokenize this token
            char piece[256];
            int len = llama_token_to_piece(
                llama_model_get_vocab(model_), 
                newToken, 
                piece, 
                sizeof(piece), 
                0, 
                true
            );
            if (len > 0) {
                result.append(piece, len);
            }
            
            // Prepare next batch with single token
            batch.n_tokens = 1;
            batch.token[0] = newToken;
            batch.pos[0] = nCur;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = 1;
            nCur++;
            
            // Decode
            if (llama_decode(ctx_, batch) != 0) {
                break;
            }
        }
        
        // Cleanup
        llama_batch_free(batch);
        
        return result;
        
    } catch (const std::exception& e) {
        return std::unexpected(std::format("Generation error: {}", e.what()));
    }
}

std::vector<llama_token> LlamaWrapper::tokenize(const std::string& text) {
    if (!model_) {
        return {};
    }
    
    // Get vocab
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    
    // Calculate required tokens
    int nTokens = -llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), 
                                  nullptr, 0, true, true);
    if (nTokens <= 0) {
        return {};
    }
    
    // Allocate and tokenize
    std::vector<llama_token> tokens(nTokens);
    nTokens = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), 
                             tokens.data(), nTokens, true, true);
    
    if (nTokens < 0) {
        return {};
    }
    
    tokens.resize(nTokens);
    return tokens;
}

std::string LlamaWrapper::detokenize(const std::vector<llama_token>& tokens) {
    if (!model_ || tokens.empty()) {
        return "";
    }
    
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    std::string result;
    
    for (const auto& token : tokens) {
        char buf[256];
        int len = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
        if (len > 0) {
            result.append(buf, len);
        }
    }
    
    return result;
}

int LlamaWrapper::getVocabSize() const {
    if (!model_) return 0;
    return llama_vocab_n_tokens(llama_model_get_vocab(model_));
}

std::string LlamaWrapper::getModelInfo() const {
    if (!model_) return "No model loaded";
    
    return std::format("Model loaded, vocab size: {}, context: {}",
                      getVocabSize(), contextSize_);
}

} // namespace llama_agent