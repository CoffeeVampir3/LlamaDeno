#include "binding.h"

#include <print>
#include <source_location>

void* LoadModel(const char *modelPath, int numberGpuLayers)
{
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = numberGpuLayers;

    llama_model* model = llama_load_model_from_file(modelPath, model_params);

    return model;
}

void* InitiateCtx(void* llamaModel, const unsigned contextLength, const unsigned numBatches)
{
    auto* model = static_cast<llama_model*>(llamaModel);
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = contextLength;
    ctx_params.n_batch = numBatches;
    ctx_params.no_perf = false;
    llama_context* ctx = llama_new_context_with_model(model, ctx_params);

    if (ctx == nullptr) {
        std::print(stderr, "error: couldn't make llama ctx @ {} {}\n",
            std::source_location::current().function_name(), std::source_location::current().line());
        return nullptr;
    }

    return ctx;
}

void* GreedySampler()
{
    auto sParams = llama_sampler_chain_default_params();
    sParams.no_perf = false;
    llama_sampler* sampler = llama_sampler_chain_init(sParams);
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    return sampler;
}

std::optional<std::string> TokenToPiece(llama_model* llamaModel, unsigned id)
{
    char buf[128];
    unsigned n = llama_token_to_piece(llamaModel, id, buf, sizeof(buf), 0, true);
    if (n < 0) {
        std::print(stderr, "error: failed to convert token to piece @ {} {}\n",
            std::source_location::current().function_name(), std::source_location::current().line());
        return std::nullopt;
    }
    return std::string{buf, n};
}

std::optional<std::vector<llama_token>> TokenizePrompt(llama_model* llamaModel, const std::string_view& prompt)
{
    const int n_prompt = -llama_tokenize(llamaModel, prompt.data(), prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokenizedPrompt(n_prompt);
    if (llama_tokenize(llamaModel, prompt.data(), prompt.size(), tokenizedPrompt.data(), tokenizedPrompt.size(), true, true) < 0) {
        std::print(stderr, "error: failed to tokenize the prompt @ {} {}\n",
            std::source_location::current().function_name(), std::source_location::current().line());
        return std::nullopt;
    }
    return tokenizedPrompt;
}

void Infer(
    void* llamaModelPtr,
    void* samplerPtr,
    void* contextPtr,
    const char *prompt,
    const unsigned numberTokensToPredict)
{
    const auto llamaModel = static_cast<llama_model*>(llamaModelPtr);
    const auto sampler = static_cast<llama_sampler*>(samplerPtr);
    const auto context = static_cast<llama_context*>(contextPtr);

    auto promptTokens = TokenizePrompt(llamaModel, prompt).value();

    const int numTokensToGenerate = (promptTokens.size() - 1) + numberTokensToPredict;
    llama_batch batch = llama_batch_get_one(promptTokens.data(), promptTokens.size(), 0, 0);

    int nDecode = 0;
    llama_token newTokenId;
    //inference
    for (int tokenPosition = 0; tokenPosition + batch.n_tokens < numTokensToGenerate; ++tokenPosition ) {
        // evaluate the current batch with the transformer model
        if (llama_decode(context, batch)) {
            std::print(stderr, "error: failed to eval, return code 1 @ {} {}\n",
                std::source_location::current().function_name(), std::source_location::current().line());
            return;
        }

        tokenPosition += batch.n_tokens;

        // sample the next token
        {
            newTokenId = llama_sampler_sample(sampler, context, -1);

            // is it an end of generation?
            if (llama_token_is_eog(llamaModel, newTokenId)) {
                break;
            }

            std::print("{}", TokenToPiece(llamaModel, newTokenId).value());

            // prepare the next batch with the sampled token
            batch = llama_batch_get_one(&newTokenId, 1, tokenPosition, 0);

            nDecode += 1;
        }
    }
}

void FreeSampler(llama_sampler* sampler)
{
    llama_sampler_free(sampler);
}

void FreeCtx(llama_context* ctx)
{
    llama_free(ctx);
}

void FreeModel(llama_model* model)
{
    llama_free_model(model);
}