#include "binding.h"

void testThing() {
  std::print("Hello world.");
}

static std::string llama_token_to_str(const struct llama_context * ctx, llama_token token) {
  std::vector<char> result(8, 0);
  const int n_tokens = llama_token_to_piece(llama_get_model(ctx), token, result.data(), result.size(), 0, false);
  if (n_tokens < 0) {
    result.resize(-n_tokens);
    int check = llama_token_to_piece(llama_get_model(ctx), token, result.data(), result.size(), 0, false);
    GGML_ASSERT(check == -n_tokens);
  } else {
    result.resize(n_tokens);
  }

  return std::string(result.data(), result.size());
}