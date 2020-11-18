#include "newell.hpp"

/** Indefinite integral needed to compute the demagkernel component Nxx.
 *  This function implements Eq. 27 in the paper of Newell.
 *  @see https://doi.org/10.1029/93JB00694
 */
__host__ __device__ static inline double Nxx_indefinite(int3 idx,
                                                        real3 cellsize) {
  double x = idx.x * cellsize.x;
  double y = idx.y * cellsize.y;
  double z = idx.z * cellsize.z;
  double R = sqrt(x * x + y * y + z * z);

  double result = 0;

  if (idx.x != 0 || idx.z != 0)  // avoid DBZ: this term->0 if x->0 and z->0
    result += (y / 2) * (z * z - x * x) * asinh(y / sqrt(x * x + z * z));

  if (idx.x != 0 || idx.y != 0)  // avoid DBZ: this term->0 if x->0 and y->0
    result += (z / 2) * (y * y - x * x) * asinh(z / sqrt(x * x + y * y));

  if (idx.x != 0)  // avoid DBZ: this term->0 if x->0
    result -= x * y * z * atan((y * z) / (x * R));

  result += (1. / 6.) * (2 * x * x - y * y - z * z) * R;

  return result;
}

/** Indefinite integral needed to compute the demagkernel component Nxy.
 *  This function implements Eq. 32 in the paper of Newell.
 *  @see https://doi.org/10.1029/93JB00694
 */
__host__ __device__ static inline double Nxy_indefinite(int3 idx,
                                                        real3 cellsize) {
  // Nxy=0 if x=0 and y=0, return early and avoid DBZ
  if (idx.y == 0 || idx.x == 0)
    return 0.0;

  double x = idx.x * cellsize.x;
  double y = idx.y * cellsize.y;
  double z = idx.z * cellsize.z;
  double R = sqrt(x * x + y * y + z * z);

  double result = 0;

  result += (x * y * z) * asinh(z / sqrt(x * x + y * y));
  result += (y / 6) * (3 * z * z - y * y) * asinh(x / sqrt(y * y + z * z));
  result += (x / 6) * (3 * z * z - x * x) * asinh(y / sqrt(x * x + z * z));
  result -= (z * y * y / 2) * atan((x * z) / (y * R));
  result -= (z * x * x / 2) * atan((y * z) / (x * R));
  result -= x * y * R / 3;

  if (idx.z != 0)  // avoid DBZ: this term->0 if z->0
    result -= (z * z * z / 6) * atan((x * y) / (z * R));

  return result;
}

/** Returns the weight of an indefinite integral term.
 *  The kernel components Nxx, Nxy, ... are a weighted sum of the indefinite
 *  integrals. This helper function returns the correct weight.
 */
__host__ __device__ static inline int newellWeight(int3 dr) {
  switch (dr.x * dr.x + dr.y * dr.y + dr.z * dr.z) {
    case 0:  // center
      return 8;
    case 1:  // side face
      return -4;
    case 2:  // edge
      return 2;
    case 3:  // corner
      return -1;
    default:
      return 0;
  }
}

__host__ __device__ real calcNewellNxx(int3 idx, real3 cellsize) {
  // TODO: the computation of the kernel can maybe be further optimized
  //       by caching (or pre-computation of) the Nxx_indefinite results
  double result = 0;

  for (int dx = -1; dx <= 1; dx++) {
    for (int dy = -1; dy <= 1; dy++) {
      for (int dz = -1; dz <= 1; dz++) {
        int3 dr{dx, dy, dz};
        result += newellWeight(dr) * Nxx_indefinite(idx + dr, cellsize);
      }
    }
  }

  result /= 4 * M_PI * cellsize.x * cellsize.y * cellsize.z;

  return result;
}

__host__ __device__ real calcNewellNxy(int3 idx, real3 cellsize) {
  if (idx.x == 0 || idx.y == 0)
    return 0;

  double result = 0;

  for (int dx = -1; dx <= 1; dx++) {
    for (int dy = -1; dy <= 1; dy++) {
      for (int dz = -1; dz <= 1; dz++) {
        int3 dr{dx, dy, dz};
        result += newellWeight(dr) * Nxy_indefinite(idx + dr, cellsize);
      }
    }
  }

  result /= 4 * M_PI * cellsize.x * cellsize.y * cellsize.z;
  return result;
}

// reuse Nxx and Nxy by permutating the arguments to implement the other kernel
// components
real calcNewellNyy(int3 idx, real3 cs) {
  return calcNewellNxx({idx.y, idx.z, idx.x}, {cs.y, cs.z, cs.x});
}
real calcNewellNzz(int3 idx, real3 cs) {
  return calcNewellNxx({idx.z, idx.x, idx.y}, {cs.z, cs.x, cs.y});
}
real calcNewellNxz(int3 idx, real3 cs) {
  return calcNewellNxy({idx.x, idx.z, idx.y}, {cs.x, cs.z, cs.y});
}
real calcNewellNyx(int3 idx, real3 cs) {
  return calcNewellNxy({idx.y, idx.x, idx.z}, {cs.y, cs.x, cs.z});
}
real calcNewellNyz(int3 idx, real3 cs) {
  return calcNewellNxy({idx.y, idx.z, idx.x}, {cs.y, cs.z, cs.x});
}
real calcNewellNzx(int3 idx, real3 cs) {
  return calcNewellNxy({idx.z, idx.x, idx.y}, {cs.z, cs.x, cs.y});
}
real calcNewellNzy(int3 idx, real3 cs) {
  return calcNewellNxy({idx.z, idx.y, idx.x}, {cs.z, cs.y, cs.x});
}
