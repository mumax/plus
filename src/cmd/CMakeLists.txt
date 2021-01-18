add_executable(cmd
    main.cpp
    examples.hpp
    skyrmion_excitation.cpp
    spinwave_dispersion.cpp
    standard_problem4.cpp
)

target_include_directories(cmd PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_link_libraries(cmd PUBLIC physics)
target_link_libraries(cmd PUBLIC core)
