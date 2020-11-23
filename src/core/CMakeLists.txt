cmake_minimum_required(VERSION 3.18) 
project(mumax5core)

add_library(core STATIC 
    butchertableau.cpp
    butchertableau.hpp
    cudastream.cu
    cudastream.hpp
    cudalaunch.hpp
    cudaerror.hpp 
    datatypes.hpp 
    dynamicequation.hpp
    dynamicequation.cpp
    field.cu 
    field.hpp 
    fieldops.cu 
    fieldops.hpp 
    fieldquantity.cpp 
    fieldquantity.hpp 
    grid.cu 
    grid.hpp 
    gpubuffer.cu
    gpubuffer.hpp
    parameter.cu
    parameter.hpp
    rungekutta.cpp
    rungekutta.hpp
    reduce.cu
    reduce.hpp
    scalarquantity.cpp 
    scalarquantity.hpp 
    stepper.cpp
    stepper.hpp
    system.cpp
    system.hpp
    timesolver.cpp
    timesolver.hpp
    timer.cpp
    timer.hpp
    variable.hpp
    variable.cpp
    world.cpp
    world.hpp
)

target_include_directories(core PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(core PUBLIC ${PROJECT_SOURCE_DIR})

set_target_properties(core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(core PROPERTIES POSITION_INDEPENDENT_CODE ON)