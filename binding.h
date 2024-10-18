#include "llama.h"
#include "common/common.h"
import std;

#ifndef BINDING_H
#define BINDING_H

#ifdef __cplusplus
extern "C" {
#endif

    void testThing(const char* modelPath, const char* prompt, int numberTokensToPredict, int numberGpuLayers);

#ifdef __cplusplus
}
#endif

#endif // BINDING_H