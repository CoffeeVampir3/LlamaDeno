cmake_minimum_required(VERSION 3.29)
project(DenoCppInterop)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Generate position-independent code (-fPIC)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Add your source files
add_library(deno_cpp_binding SHARED
        binding.cpp
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