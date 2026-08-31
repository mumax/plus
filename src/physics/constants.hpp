#pragma once

#include "datatypes.hpp"

#if FP_PRECISION == SINGLE

// Gyromagnetic ratio in rad/Ts
#define GAMMALL_DEFAULT = 1.7595E11F
// Vacuum permeability in H/m
#define MU0 1.256637062E-6F
// Boltzmann constant in J/K
#define KB 1.38064852E-23F
// Electron charge in C
#define QE 1.60217646E-19F
// Bohr magneton in J/T
#define MUB 9.274009152E-24F
// Reduced Planck constant in Js
#define HBAR 1.054571817E-34F
#elif FP_PRECISION == DOUBLE
// Gyromagnetic ratio in rad/Ts
#define GAMMALL_DEFAULT = 1.7595E11
// Vacuum permeability in H/m
#define MU0 1.25663706212E-6
// Boltzmann constant in J/K
#define KB 1.38064852E-23
// Electron charge in C
#define QE 1.60217646E-19
// Bohr magneton in J/T
#define MUB 9.2740091523E-24
// Reduced Planck constant in Js
#define HBAR 1.054571817E-34
#else
#error FP_PRECISION should be SINGLE or DOUBLE
#endif
