cmake_minimum_required(VERSION 3.18) 
project(mumax5physics)

add_library(physics STATIC 
    anisotropy.cu 
    anisotropy.hpp 
    constants.hpp
    demag.cpp
    demag.hpp
    effectivefield.hpp 
    effectivefield.cpp
    energy.hpp 
    energy.cu
    exchange.hpp 
    exchange.cu 
    ferromagnet.cpp 
    ferromagnet.hpp 
    ferromagnetquantity.cpp 
    ferromagnetquantity.hpp 
    interfacialdmi.cu
    interfacialdmi.hpp
    minimizer.cu
    minimizer.hpp
    magnetfield.cpp
    magnetfield.hpp
    magnetfieldbrute.cu
    magnetfieldbrute.hpp
    magnetfieldfft.cu
    magnetfieldfft.hpp
    magnetfieldkernel.cu
    magnetfieldkernel.hpp
    mumaxworld.cpp
    mumaxworld.hpp
    newell.cu
    newell.hpp
    stt.cu
    stt.hpp
    thermalnoise.cu
    thermalnoise.hpp
    torque.cu 
    torque.hpp 
    zeeman.cu
    zeeman.hpp
)

target_include_directories(physics PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(physics PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(physics PUBLIC core)

link_directories(${CUDA_LIBRARY_DIR})
target_link_libraries(physics PRIVATE cufft curand)

set_target_properties(physics PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(physics PROPERTIES POSITION_INDEPENDENT_CODE ON)