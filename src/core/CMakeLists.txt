add_library(core STATIC 
    butchertableau.cpp
    butchertableau.hpp
    datatypes.hpp 
    dynamicequation.hpp
    dynamicequation.cpp
    dynamic_parameter.hpp
    dynamic_parameter.cpp
    field.cu 
    field.hpp 
    fieldops.cu 
    fieldops.hpp 
    fieldquantity.cpp 
    fieldquantity.hpp 
    grid.cu 
    grid.hpp
    inter_parameter.cu
    inter_parameter.hpp
    parameter.cu
    parameter.hpp
    reduce.cu
    reduce.hpp
    rungekutta.cpp
    rungekutta.hpp
    scalarquantity.cpp 
    scalarquantity.hpp
    shift.cu
    shift.hpp
    stepper.cpp
    stepper.hpp
    system.cpp
    system.hpp
    timesolver.cpp
    timesolver.hpp
    variable.hpp
    variable.cpp
    voronoi.cpp
    voronoi.hpp
    window.cpp
    window.hpp
    world.cpp
    world.hpp
)

target_include_directories(core PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(core PUBLIC .)

target_link_libraries(core PUBLIC cudautil)