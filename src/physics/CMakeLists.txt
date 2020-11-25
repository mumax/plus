#project(mumax5)

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
    strayfield.cpp
    strayfield.hpp
    strayfieldbrute.cu
    strayfieldbrute.hpp
    strayfieldfft.cu
    strayfieldfft.hpp
    strayfieldkernel.cu
    strayfieldkernel.hpp
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
target_include_directories(physics PUBLIC .)

target_link_libraries(physics PUBLIC core timesolver)

link_directories(${CUDA_LIBRARY_DIR})
target_link_libraries(physics PRIVATE cufft curand)