// main.ts

// Define the library interface
const libInterface = {
    LoadModel: {
        parameters: [
            "pointer",  // const char *modelPath
            "i32",      // int numberGpuLayers
        ],
        result: "pointer" as const  // void*
    },
    InitiateCtx: {
        parameters: [
            "pointer",  // void* llamaModel
            "u32",      // unsigned contextLength
            "u32",      // unsigned numBatches
        ],
        result: "pointer" as const  // void*
    },
    MakeSampler: {
        parameters: [
            "pointer",  // void* llamaModel
        ],
        result: "pointer" as const  // void*
    },
    GreedySampler: {
        parameters: ["pointer"],  // void* sampler
        result: "pointer" as const  // void*
    },
    TopK: {
        parameters: ["pointer", "i32"],  // void* sampler, int num
        result: "pointer" as const  // void*
    },
    TopP: {
        parameters: ["pointer", "f32", "f32"],  // void* sampler, float p, float minKeep
        result: "pointer" as const  // void*
    },
    Infer: {
        parameters: [
            "pointer",  // void* llamaModelPtr
            "pointer",  // void* samplerPtr
            "pointer",  // void* contextPtr
            "pointer",  // const char *prompt
            "u32",      // unsigned numberTokensToPredict
        ],
        result: "void" as const
    },
} as const;

class SamplerBuilder {
    private sampler: Deno.PointerValue;

    constructor(private lib: Deno.DynamicLibrary<typeof libInterface>, llamaModel: Deno.PointerValue) {
        this.sampler = this.lib.symbols.MakeSampler(llamaModel);
    }

    greedy(): this {
        this.sampler = this.lib.symbols.GreedySampler(this.sampler);
        return this;
    }

    topK(num: number): this {
        this.sampler = this.lib.symbols.TopK(this.sampler, num);
        return this;
    }

    topP(p: number, minKeep: number): this {
        this.sampler = this.lib.symbols.TopP(this.sampler, p, minKeep);
        return this;
    }

    build(): Deno.PointerValue {
        return this.sampler;
    }
}

// Define the library name and path
const libName = "deno_cpp_binding";
const libSuffix = {
    "windows": "dll",
    "darwin": "dylib",
    "linux": "so"
}[Deno.build.os];

if (!libSuffix) {
    throw new Error(`Unsupported operating system: ${Deno.build.os}`);
}

const scriptDir = new URL(".", import.meta.url).pathname;
const libPath = `${scriptDir}lib/${libName}.${libSuffix}`;
try {
    console.log(`Attempting to load library from: ${libPath}`);

    // Load the library
    const lib: Deno.DynamicLibrary<typeof libInterface> = Deno.dlopen(libPath, libInterface);

    console.log("Library loaded successfully.");

    // Example usage of the functions
    const modelPath = "/home/blackroot/Desktop/LlamaDeno/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf";
    const numberGpuLayers = 40;
    const contextLength = 2048;
    const numBatches = 1;
    const numberTokensToPredict = 20;

    const modelPathPtr = new TextEncoder().encode(modelPath + "\0");

    // LoadModel
    const llamaModel = lib.symbols.LoadModel(
        Deno.UnsafePointer.of(modelPathPtr),
        numberGpuLayers
    );

    // InitiateCtx
    const context = lib.symbols.InitiateCtx(
        llamaModel,
        contextLength,
        numBatches
    );

    // GreedySampler
    const samplerBuilder = new SamplerBuilder(lib, llamaModel);
    const sampler = samplerBuilder
        .topK(40)
        .build();

    // For Infer, we need to create a vector of tokens
    // This is a simplified example and may need adjustment based on your actual implementation
    const prompt = "Hello, how are you?";
    const promptPtr = new TextEncoder().encode(prompt + "\0");

    // Infer
    lib.symbols.Infer(
        llamaModel,
        sampler,
        context,
        Deno.UnsafePointer.of(promptPtr),
        numberTokensToPredict
    );

    // Close the library when done
    lib.close();
    console.log("Library closed.");
} catch (error) {
    console.error("Error:", error.message);
}
