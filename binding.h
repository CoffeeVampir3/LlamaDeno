#include "llama.h"
#include "common/common.h"

#ifndef BINDING_H
#define BINDING_H

#ifdef __cplusplus
extern "C" {
#endif

    void* LoadModel(const char *modelPath, int numberGpuLayers);
    void* InitiateCtx(void* llamaModel, unsigned contextLength, unsigned numBatches);
    void* MakeSampler(void* llamaModelPtr);
    void* GreedySampler(void* sampler);
    void* TopK(void* sampler, int num);
    void* TopP(void* sampler, float p, float minKeep);
    void Infer(
        void* llamaModelPtr,
        void* samplerPtr,
        void* contextPtr,
        const char *prompt,
        unsigned numberTokensToPredict);

#ifdef __cplusplus
}
#endif

#endif // BINDING_H