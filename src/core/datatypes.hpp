#pragma once

#include <cuComplex.h>
#include <cuda_runtime_api.h>

#include <iostream>

#define SINGLE 1
#define DOUBLE 2

#ifndef FP_PRECISION
#define FP_PRECISION SINGLE
#endif

#if FP_PRECISION == SINGLE
using real = float;
using real3 = float3;
using complex = cuComplex;
#elif FP_PRECISION == DOUBLE
using real = double;
using real3 = double3;
using complex = cuDoubleComplex;
#else
#error FP_PRECISION should be SINGLE or DOUBLE
#endif

#define __CUDAOP__ inline __device__ __host__

__CUDAOP__ void operator+=(int3& a, const int3& b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}

__CUDAOP__ void operator-=(int3& a, const int3& b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
}

__CUDAOP__ int3 operator+(const int3& a, const int3& b) {
  return int3{a.x + b.x, a.y + b.y, a.z + b.z};
}

__CUDAOP__ int3 operator-(const int3& a) {
  return int3{-a.x, -a.y, -a.z};
}

__CUDAOP__ int3 operator-(const int3& a, const int3& b) {
  return int3{a.x - b.x, a.y - b.y, a.z - b.z};
}

__CUDAOP__ int3 operator*(const int3& a, const int3& b) {
  return int3{a.x * b.x, a.y * b.y, a.z * b.z};
}

__CUDAOP__ int3 operator/(const int3& a, const int3& b) {
  return int3{a.x / b.x, a.y / b.y, a.z / b.z};
}

__CUDAOP__ bool operator==(const int3& a, const int3& b) {
  return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);
}

__CUDAOP__ bool operator!=(const int3& a, const int3& b) {
  return (a.x != b.x) || (a.y != b.y) || (a.z != b.z);
}

inline __host__ std::ostream& operator<<(std::ostream& os, const int3 a) {
  os << "(" << a.x << "," << a.y << "," << a.z << ")";
  return os;
}

__CUDAOP__ void operator+=(real3& a, const real3& b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}

__CUDAOP__ void operator+=(real3& a, const real& b) {
  a.x += b;
  a.y += b;
  a.z += b;
}

__CUDAOP__ void operator-=(real3& a, const real3& b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
}

__CUDAOP__ void operator-=(real3& a, const real& b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
}

__CUDAOP__ void operator*=(real3& a, const real& b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
}

__CUDAOP__ void operator/=(real3& a, const real& b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
}

__CUDAOP__ real3 operator+(const real3& a, const real3& b) {
  return real3{a.x + b.x, a.y + b.y, a.z + b.z};
}

__CUDAOP__ real3 operator+(const real& a, const real3& b) {
  return real3{a + b.x, a + b.y, a + b.z};
}

__CUDAOP__ real3 operator+(const real3& a, const real& b) {
  return real3{a.x + b, a.y + b, a.z + b};
}

__CUDAOP__ real3 operator-(const real3& a) {
  return real3{-a.x, -a.y, -a.z};
}

__CUDAOP__ real3 operator-(const real3& a, const real3& b) {
  return real3{a.x - b.x, a.y - b.y, a.z - b.z};
}

__CUDAOP__ real3 operator-(const real3& a, const real& b) {
  return real3{a.x - b, a.y - b, a.z - b};
}

__CUDAOP__ real3 operator-(const real& a, const real3& b) {
  return real3{a - b.x, a - b.y, a - b.z};
}

__CUDAOP__ real3 operator*(const real& a, const real3& b) {
  return real3{a * b.x, a * b.y, a * b.z};
}

__CUDAOP__ real3 operator*(const real3& a, const real& b) {
  return real3{a.x * b, a.y * b, a.z * b};
}

__CUDAOP__ real3 operator*(const real3& a, const real3& b) {
  return real3{a.x * b.x, a.y * b.y, a.z * b.z};
}

__CUDAOP__ real3 operator/(const real3& a, const real& b) {
  return real3{a.x / b, a.y / b, a.z / b};
}

__CUDAOP__ real3 operator/(const real3& a, const real3& b) {
  return real3{a.x / b.x, a.y / b.y, a.z / b.z};
}

__CUDAOP__ real dot(const real3& a, const real3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__CUDAOP__ real3 cross(const real3& a, const real3& b) {
  return real3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
               a.x * b.y - a.y * b.x};
}

__CUDAOP__ real norm(const real3& a) {
  return sqrt(dot(a, a));
}

// Returns the normalized vector.
// If the norm is zero, it returns a zero vector
__CUDAOP__ real3 normalized(const real3& a) {
  real scalingsFactor = (norm(a) == real(0)) ? 0. : 1. / norm(a);
  return scalingsFactor * a;
}

__CUDAOP__ bool operator==(const real3& a, const real3& b) {
  return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);
}

__CUDAOP__ bool operator!=(const real3& a, const real3& b) {
  return (a.x != b.x) || (a.y != b.y) || (a.z != b.z);
}

inline __host__ std::ostream& operator<<(std::ostream& os, const real3 a) {
  os << "(" << a.x << "," << a.y << "," << a.z << ")";
  return os;
}
