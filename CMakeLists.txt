cmake_minimum_required(VERSION 3.18)
project(mumax5 LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

if (WIN32)
    set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS TRUE)
endif()

# if environment variable CUDA_CC is set, then compile for this architecture,
# otherwise compile for 3.0 (lowest supported architecture for cuda10.0).
# CUDA_CC can be a space-delimited list of architectures
if ($ENV{CUDA_CC})
    set(CMAKE_CUDA_ARCHITECTURES $ENV{CUDA_CC})
else()
    set(CMAKE_CUDA_ARCHITECTURES 30)
endif()

add_definitions(-DFP_PRECISION=SINGLE) # FP_PRECISION should be SINGLE or DOUBLE
add_definitions(-D_USE_MATH_DEFINES) # Needed for cmath constants on Windows

add_subdirectory(src/core)
add_subdirectory(src/bindings)
add_subdirectory(src/physics)
