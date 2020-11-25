add_library(timesolver STATIC 
    butchertableau.cpp
    butchertableau.hpp
    dynamicequation.hpp
    dynamicequation.cpp
    rungekutta.cpp
    rungekutta.hpp
    stepper.cpp
    stepper.hpp
    timesolver.cpp
    timesolver.hpp
)

target_include_directories(timesolver PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(timesolver PUBLIC .)

target_link_libraries(timesolver PUBLIC core)