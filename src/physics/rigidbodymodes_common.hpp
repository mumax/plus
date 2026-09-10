#pragma once
#include "datatypes.hpp"   // for real3, real

struct ComResult { real3 com; real weightSum; };
struct Mat3x3 { double m[3][3]; };

// f64 cell position — for accumulation kernels where cancellation matters.
__device__ inline double3 cellPositionDeviceD(const CuSystem& system, int idx) {
  int3 coord = system.grid.index2coord(idx);
  return double3{double(coord.x) * double(system.cellsize.x),
                 double(coord.y) * double(system.cellsize.y),
                 double(coord.z) * double(system.cellsize.z)};
}

// f32 cell position
__device__ inline real3 cellPositionDevice(const CuSystem& system, int idx) {
  int3 coord = system.grid.index2coord(idx);
  return real3{real(coord.x) * system.cellsize.x,
               real(coord.y) * system.cellsize.y,
               real(coord.z) * system.cellsize.z};
}



__host__ __device__ inline real3 toReal3(double3 v) {
  return real3{real(v.x), real(v.y), real(v.z)};
}
__host__ __device__ inline double3 toDouble3(real3 v) {
  return double3{double(v.x), double(v.y), double(v.z)};
}


