project(mumax5)

add_subdirectory(pybind11)

pybind11_add_module(_mumax5cpp
    cast.hpp
    main.cpp
    wrappers.hpp
    wrap_ferromagnet.cpp
    wrap_field.cpp
    wrap_fieldquantity.cpp
    wrap_grid.cpp
    wrap_parameter.cpp
    wrap_scalarquantity.cpp
    wrap_strayfield.cpp
    wrap_system.cpp
    wrap_timesolver.cpp
    wrap_variable.cpp
    wrap_world.cpp
)

target_include_directories(_mumax5cpp PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(_mumax5cpp PRIVATE core physics)