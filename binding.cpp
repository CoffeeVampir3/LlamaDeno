#include "binding.h"

#include <source_location>

void testThing(const char *modelPath, const char* prompt, int numberTokensToPredict, int numberGpuLayers)
{
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = numberGpuLayers;

    llama_model* model = llama_load_model_from_file(modelPath, model_params);

    const int n_prompt = -llama_tokenize(model, prompt, strlen(prompt), NULL, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(model, prompt, strlen(prompt), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        std::print(stderr, "error: failed to tokenize the prompt @ {} {}\n",
            std::source_location::current().function_name(), std::source_location::current().line());
        return;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_prompt + numberTokensToPredict - 1;
    ctx_params.n_batch = n_prompt;
    ctx_params.no_perf = false;
    llama_context * ctx = llama_new_context_with_model(model, ctx_params);

    if (ctx == nullptr) {
        std::print(stderr, "error: couldn't make llama ctx @ {} {}\n",
            std::source_location::current().function_name(), std::source_location::current().line());
        return;
    }

    auto sParams = llama_sampler_chain_default_params();
    sParams.no_perf = false;
    llama_sampler* smpl = llama_sampler_chain_init(sParams);

    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    for (auto id : prompt_tokens) {
        char buf[128];
        int n = llama_token_to_piece(model, id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            std::print(stderr, "error: failed to convert token to piece @ {} {}\n",
                std::source_location::current().function_name(), std::source_location::current().line());
            return;
        }
        std::string s(buf, n);
        std::print("{}", s);
    }

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size(), 0, 0);

    const auto t_main_start = ggml_time_us();
    int nDecode = 0;
    llama_token new_token_id;

    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + numberTokensToPredict; ++n_pos ) {
        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            std::print(stderr, "error: failed to eval, return code 1 @ {} {}\n",
                std::source_location::current().function_name(), std::source_location::current().line());
            return;
        }

        n_pos += batch.n_tokens;

        // sample the next token
        {
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // is it an end of generation?
            if (llama_token_is_eog(model, new_token_id)) {
                break;
            }

            char buf[128];
            int n = llama_token_to_piece(model, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                std::print(stderr, "error: failed to convert token to piece @ {} {}\n",
                    std::source_location::current().function_name(), std::source_location::current().line());
                return;
            }
            std::string s(buf, n);
            std::print("{}", s);

            // prepare the next batch with the sampled token
            batch = llama_batch_get_one(&new_token_id, 1, n_pos, 0);

            nDecode += 1;
        }
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_free_model(model);
}
