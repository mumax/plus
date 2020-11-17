#include "newell.hpp"


// Eq. 27 in paper of Newell (doi.org/10.1029/93JB00694)
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

// Eq. 32 in paper of Newell
__host__ __device__ static inline double Nxy_indefinite(int3 idx,
                                                        real3 cellsize) {
  if (idx.y == 0 ||
      idx.x == 0)  // Nxy=0 if x=0 and y=0, return early and avoid DBZ
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

// Eq. 19 in paper of Newell
__host__ __device__ real calcNewellNxx(int3 idx, real3 cellsize) {
  double result = 0;

  for (int dx = -1; dx <= 1; dx++) {
    for (int dy = -1; dy <= 1; dy++) {
      for (int dz = -1; dz <= 1; dz++) {
        // TODO: outer loop can be optimized for dz = 0 because
        //       newell_xx yiels the same for z+dz and z-dz
        // printf("We are in the loop!\n");
        // weight factor:
        //    8 for the center
        //   -4 for side faces
        //    2 for edges
        //   -1 for corners
        printf("Power %d, %E\n", dx * dx + dy * dy + dz * dz, dx * dx + dy * dy + dz * dz);
        int weight = 8 / pow(-2, dx * dx + dy * dy + dz * dz);
        // printf("Pow works just fine!");

        // TODO: the computation of the kernel can maybe be further optimized
        //       by caching (or pre-computation of) the Nxx_indefinite results
        result += weight * Nxx_indefinite(idx + int3{dx, dy, dz}, cellsize);
        printf("We calculated our result.");
      }
    }
  }

  result /= 4 * M_PI * cellsize.x * cellsize.y * cellsize.z;

  return result;
}

// Eq. 28 in paper of Newell
__host__ __device__ real calcNewellNxy(int3 idx, real3 cellsize) {
  if (idx.x == 0 || idx.y == 0)
    return 0;

  double result = 0;

  for (int dx = -1; dx <= 1; dx++) {
    for (int dy = -1; dy <= 1; dy++) {
      for (int dz = -1; dz <= 1; dz++) {
        // weight factor:
        //    8 for the center
        //   -4 for side faces
        //    2 for edges
        //   -1 for corners
        int weight = 8 / pow(-2, dx * dx + dy * dy + dz * dz);

        result += weight * Nxy_indefinite(idx + int3{dx, dy, dz}, cellsize);
      }
    }
  }

  result /= 4 * M_PI * cellsize.x * cellsize.y * cellsize.z;
  return result;
}

// reuse Nxx and Nxy by permutating the arguments
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
