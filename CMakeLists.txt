cmake_minimum_required(VERSION 3.24)
project(mumaxplus LANGUAGES CUDA CXX)

set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (WIN32)
    set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS TRUE)
endif()

set(CMAKE_CUDA_ARCHITECTURES native)

add_definitions(-DFP_PRECISION=SINGLE) # FP_PRECISION should be SINGLE or DOUBLE
add_definitions(-D_USE_MATH_DEFINES) # Needed for cmath constants on Windows

add_subdirectory(src/bindings)
add_subdirectory(src/cmd)
add_subdirectory(src/core)
add_subdirectory(src/cudautil)
add_subdirectory(src/linsolver)
add_subdirectory(src/physics)
