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
    newellasymptotic.cu
    newellasymptotic.hpp
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
    hostmagnet.cpp
    hostmagnet.hpp
    internalbodyforce.cu
    internalbodyforce.hpp
    local_dmi.cu
    local_dmi.hpp
    magnet.cpp
    magnet.hpp 
    magnetoelasticfield.cu
    magnetoelasticfield.hpp
    magnetoelasticforce.cu
    magnetoelasticforce.hpp
    mfm.cu
    mfm.hpp
    minimizer.cu
    minimizer.hpp
    mumaxworld.cpp
    mumaxworld.hpp
    ncafm.cpp
    ncafm.hpp
    ncafmangle.cu
    ncafmangle.hpp
    neel.cu
    neel.hpp
    newell.cu
    newell.hpp
    octupole.cu
    octupole.hpp
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
    traction.cpp
    traction.hpp
    quantityevaluator.hpp
    zeeman.cu
    zeeman.hpp
)

target_include_directories(physics PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(physics PUBLIC .)

target_link_libraries(physics PUBLIC core linsolver)

link_directories(${CUDA_LIBRARY_DIR})
target_link_libraries(physics PRIVATE cufft curand)


# On Windows, the cufft and curand DLLs (also required by mumax³) are not found by default.
# Therefore, we look for them in the installed NVIDIA CUDA Toolkit and copy them next to the .pyd.
if (WIN32)
    find_package(CUDAToolkit REQUIRED)

    file(GLOB CUDA_RUNTIME_DLLS
        "${CUDAToolkit_BIN_DIR}/cufft64_*.dll"
        "${CUDAToolkit_BIN_DIR}/curand64_*.dll"
    )

    add_custom_command(TARGET physics POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CUDA_RUNTIME_DLLS}
            $<TARGET_FILE_DIR:_mumaxpluscpp>
    )
endif()
