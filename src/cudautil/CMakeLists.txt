add_library(cudautil STATIC 
    cudastream.cu
    cudastream.hpp
    cudalaunch.hpp
    cudaerror.hpp 
    gpubuffer.hpp
    gpumemorypool.cu
    gpumemorypool.hpp
)

target_include_directories(cudautil PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(cudautil PUBLIC .)