#include "llama.h"
import std;

#ifndef BINDING_H
#define BINDING_H

#ifdef __cplusplus
extern "C" {
#endif

    void testThing();

    static std::string llama_token_to_str(const struct llama_context * ctx, llama_token token);

#ifdef __cplusplus
}
#endif

#endif // BINDING_H