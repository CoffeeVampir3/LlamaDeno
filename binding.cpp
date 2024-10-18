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
    ctx_params.no_perf = true;
    llama_context* ctx = llama_new_context_with_model(model, ctx_params);

    if (ctx == nullptr) {
        std::print(stderr, "error: couldn't make llama ctx @ {} {}\n",
            std::source_location::current().function_name(), std::source_location::current().line());
        return nullptr;
    }

    return ctx;
}

void* MakeSampler(void* llamaModelPtr)
{
    const auto* llamaModel = static_cast<llama_model*>(llamaModelPtr);

    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();
    lparams.no_perf = true;
    auto sampler = llama_sampler_chain_init(lparams);
    const auto params = common_sampler_params{
        .seed = LLAMA_DEFAULT_SEED,
        .n_prev = 64,
        .n_probs = 0,
        .min_keep = 0,
        .top_k = 40,
        .top_p = 0.95f,
        .min_p = 0.05f,
        .xtc_probability = 0.00f,
        .xtc_threshold = 0.10f,
        .tfs_z = 1.00f,
        .typ_p = 1.00f,
        .temp = 15.0f,
        .dynatemp_range = 0.00f,
        .dynatemp_exponent = 1.00f,
        .penalty_last_n = 64,
        .penalty_repeat = 1.00f,
        .penalty_freq = 0.00f,
        .penalty_present = 0.00f,
        .mirostat = 0,
        .mirostat_tau = 5.00f,
        .mirostat_eta = 0.10f,
        .penalize_nl = false,
        .ignore_eos = false,
        .no_perf = false
    };
    llama_sampler_chain_add(sampler,
            llama_sampler_init_logit_bias(
                llama_n_vocab(llamaModel),
                params.logit_bias.size(),
                params.logit_bias.data()));

    llama_sampler_chain_add(sampler,
            llama_sampler_init_penalties(
                llama_n_vocab  (llamaModel),
                llama_token_eos(llamaModel),
                llama_token_nl (llamaModel),
                params.penalty_last_n,
                params.penalty_repeat,
                params.penalty_freq,
                params.penalty_present,
                params.penalize_nl,
                params.ignore_eos));

    llama_sampler_chain_add(sampler, llama_sampler_init_temp (params.temp));
    return sampler;
}

void* GreedySampler(void* sampler)
{
    llama_sampler_chain_add(static_cast<llama_sampler*>(sampler), llama_sampler_init_greedy());
    return sampler;
}

void* TopK(void* sampler, const int num)
{
    llama_sampler_chain_add(static_cast<llama_sampler*>(sampler), llama_sampler_init_top_k(num));
    llama_sampler_chain_add(static_cast<llama_sampler*>(sampler), llama_sampler_init_softmax());
    llama_sampler_chain_add(static_cast<llama_sampler*>(sampler), llama_sampler_init_dist(1337));
    return sampler;
}

void* TopP(void* sampler, const float p, const float minKeep)
{
    llama_sampler_chain_add(static_cast<llama_sampler*>(sampler), llama_sampler_init_top_p(p, minKeep));
    return sampler;
}

std::optional<std::string> TokenToPiece(llama_model* llamaModel, unsigned id)
{
    char buf[128];
    int n = llama_token_to_piece(llamaModel, id, buf, sizeof(buf), 0, true);
    if (n < 0) {
        std::print(stderr, "error: failed to convert token to piece @ {} {}\n",
            std::source_location::current().function_name(), std::source_location::current().line());
        return std::nullopt;
    }
    return std::string{buf, static_cast<size_t>(n)};
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