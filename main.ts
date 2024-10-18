// main.ts

// Define the library interface
const libInterface = {
    testThing: {
        parameters: [
            "pointer",  // const char *modelPath
            "pointer",  // const char *prompt
            "i32",      // int numberTokensToPredict
            "i32"       // int numberGpuLayers
        ],
        result: "void" as const
    },
} as const;

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

    // Call the function
    console.log("Calling C++ function from Deno:");
    const modelPath = "/home/blackroot/Desktop/LlamaDeno/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf";
    const prompt = "hi";
    const numberTokensToPredict = 100;
    const numberGpuLayers = 1;

    const modelPathPtr = new TextEncoder().encode(modelPath + "\0");
    const promptPtr = new TextEncoder().encode(prompt + "\0");

    lib.symbols.testThing(
        Deno.UnsafePointer.of(modelPathPtr),
        Deno.UnsafePointer.of(promptPtr),
        numberTokensToPredict,
        numberGpuLayers
    );

    // Close the library when done
    lib.close();
    console.log("Library closed.");
} catch (error) {
    console.error("Error:", error.message);
}