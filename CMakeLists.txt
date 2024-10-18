cmake_minimum_required(VERSION 3.29.0 FATAL_ERROR)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "0e5b6991-d74f-4b3d-a41c-cf096e0b2508")
set(CMAKE_CXX_MODULE_STD ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")

# Generate position-independent code (-fPIC)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(DenoCppInterop)

# Add your source files
add_library(deno_cpp_binding SHARED
        binding.cpp
)

file(GLOB_RECURSE MODULE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/Modules/*.ixx"
)

# Include directories
target_include_directories(deno_cpp_binding PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Set output name
set_target_properties(deno_cpp_binding PROPERTIES
        OUTPUT_NAME "deno_cpp_binding"
        PREFIX ""
)

# Set output directory
set_target_properties(deno_cpp_binding PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib"
)

# On Windows, we need to export symbols
if(WIN32)
    set_target_properties(deno_cpp_binding PROPERTIES
            WINDOWS_EXPORT_ALL_SYMBOLS TRUE
    )
endif()

# Add C++ module sources
target_sources(deno_cpp_binding
        PUBLIC
        FILE_SET CXX_MODULES TYPE CXX_MODULES
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
        FILES ${MODULE_FILES}
)