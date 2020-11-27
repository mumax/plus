add_library(core STATIC 
    datatypes.hpp 
    field.cu 
    field.hpp 
    fieldops.cu 
    fieldops.hpp 
    fieldquantity.cpp 
    fieldquantity.hpp 
    grid.cu 
    grid.hpp 
    parameter.cu
    parameter.hpp
    reduce.cu
    reduce.hpp
    scalarquantity.cpp 
    scalarquantity.hpp 
    system.cpp
    system.hpp
    variable.hpp
    variable.cpp
    world.cpp
    world.hpp
)

target_include_directories(core PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(core PUBLIC .)

target_link_libraries(core PUBLIC cudautil)