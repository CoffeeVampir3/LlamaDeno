cmake_minimum_required(VERSION 3.29.0 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "0e5b6991-d74f-4b3d-a41c-cf096e0b2508")
set(CMAKE_CXX_MODULE_STD ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")

project(DenoCppInterop)

#add_compile_definitions(GGML_USE_CUDA)
#add_compile_definitions(GGML_USE_CUBLAS)

include(FetchContent)
FetchContent_Declare(
        llama
        GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
        GIT_TAG master
)

# Disable unused components
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "llama.cpp: build examples" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "llama.cpp: build tests" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "llama.cpp: build server" FORCE)

FetchContent_MakeAvailable(llama)

add_library(deno_cpp_binding SHARED
        binding.cpp
)

file(GLOB_RECURSE MODULE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/Modules/*.ixx"
)

target_include_directories(deno_cpp_binding
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${llama_SOURCE_DIR}
)

target_link_libraries(deno_cpp_binding PRIVATE llama)

set_target_properties(deno_cpp_binding PROPERTIES
        OUTPUT_NAME "deno_cpp_binding"
        PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib"
)

if(WIN32)
    set_target_properties(deno_cpp_binding PROPERTIES
            WINDOWS_EXPORT_ALL_SYMBOLS TRUE
    )
endif()

target_sources(deno_cpp_binding
        PUBLIC
        FILE_SET CXX_MODULES TYPE CXX_MODULES
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
        FILES ${MODULE_FILES}
)