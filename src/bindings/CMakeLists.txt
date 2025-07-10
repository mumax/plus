project(mumaxplus)

# disable pybind11::(thin_)lto, as this does not work with CUDA
# https://github.com/pybind/pybind11/issues/4825
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

add_subdirectory(pybind11)

pybind11_add_module(_mumaxpluscpp
    cast.hpp
    main.cpp
    wrappers.hpp
    wrap_antiferromagnet.cpp
    wrap_dmitensor.cpp
    wrap_ferromagnet.cpp
    wrap_field.cpp
    wrap_fieldquantity.cpp
    wrap_grid.cpp
    wrap_magnet.cpp
    wrap_mfm.cpp
    wrap_ncafm.cpp
    wrap_parameter.cpp
    wrap_scalarquantity.cpp
    wrap_strayfield.cpp
    wrap_system.cpp
    wrap_timesolver.cpp
    wrap_variable.cpp
    wrap_world.cpp
    wrap_poissonsystem.cpp
    wrap_linsolver.cpp
    wrap_voronoi.cpp
    wrap_traction.cpp
)

target_include_directories(_mumaxpluscpp PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(_mumaxpluscpp PRIVATE core physics)