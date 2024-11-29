#project(mumaxplus)

add_library(physics STATIC 
    afmexchange.cu
    afmexchange.hpp
    anisotropy.cu 
    anisotropy.hpp 
    antiferromagnet.cpp
    antiferromagnet.hpp
    conductivitytensor.cu
    conductivitytensor.hpp
    constants.hpp
    demag.cpp
    demag.hpp
    dmi.cu
    dmi.hpp
    dmitensor.cpp
    dmitensor.hpp
    effectivefield.hpp 
    effectivefield.cpp
    elasticdamping.cu
    elasticdamping.hpp
    elasticenergies.cu
    elasticenergies.hpp
    elasticforce.cu
    elasticforce.hpp
    elastodynamics.cu
    elastodynamics.hpp
    electricalpotential.hpp
    electricalpotential.cu
    energy.hpp 
    energy.cu
    exchange.hpp 
    exchange.cu 
    ferromagnet.cpp 
    ferromagnet.hpp 
    fullmag.cu
    fullmag.hpp 
    magnet.cpp
    magnet.hpp 
    magnetoelasticfield.cu
    magnetoelasticfield.hpp
    magnetoelasticforce.cu
    magnetoelasticforce.hpp
    minimizer.cu
    minimizer.hpp
    mumaxworld.cpp
    mumaxworld.hpp
    neel.cu
    neel.hpp
    newell.cu
    newell.hpp
    poissonsystem.hpp
    poissonsystem.cu
    poyntingvector.cu
    poyntingvector.hpp
    relaxer.hpp
    relaxer.cpp
    straintensor.cu
    straintensor.hpp
    stresstensor.cu
    stresstensor.hpp
    strayfield.cpp
    strayfield.hpp
    strayfieldbrute.cu
    strayfieldbrute.hpp
    strayfieldfft.cu
    strayfieldfft.hpp
    strayfieldkernel.cu
    strayfieldkernel.hpp
    stt.cu
    stt.hpp
    thermalnoise.cu
    thermalnoise.hpp
    torque.cu 
    torque.hpp
    quantityevaluator.hpp
    zeeman.cu
    zeeman.hpp
)

target_include_directories(physics PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(physics PUBLIC .)

target_link_libraries(physics PUBLIC core linsolver)

link_directories(${CUDA_LIBRARY_DIR})
target_link_libraries(physics PRIVATE cufft curand)
