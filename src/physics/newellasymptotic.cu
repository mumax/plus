#include "newellasymptotic.hpp"

#include "gpubuffer.hpp"
#include <cstdio>


// Function to calculate factorials
__host__ __device__ int fac(int val) {
  static const int factorials[] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600};
  return factorials[val];
}

/** Calculate the asymptotic solution of the demagkernel component Nxx. 
    This method is based on the method used in OOMMF.
    https://math.nist.gov/~MDonahue/talks/mmm2020-talk.pdf
 */
__host__ __device__ real calcAsymptoticNxx(int3 idx, real3 cellsize, int* expansionNxxptr, size_t sizeNxx) {

  double hx = cellsize.x;
  double hy = cellsize.y;
  double hz = cellsize.z;
  double x = idx.x * hx;
  double y = idx.y * hy;
  double z = idx.z * hz;
  double R = sqrt(x*x + y*y + z*z);
  
  double result = 0;
  
  int N, a, b, c, P, d, e, f;
  double A, weight, hi, xyz, Rpow;

  for (int i = 0; i < sizeNxx; i+=8) {
    N = expansionNxxptr[i];
    a = expansionNxxptr[i+1];
    b = expansionNxxptr[i+2];
    c = expansionNxxptr[i+3];
    P = expansionNxxptr[i+4];
    d = expansionNxxptr[i+5];
    e = expansionNxxptr[i+6];
    f = expansionNxxptr[i+7];

    Rpow = pow(R,P);

    A = fac(d+2) * fac(e+2) * fac(f+2);
    weight = N/(A*Rpow);

    hi = pow(hx,d)*pow(hy,e)*pow(hz,f);
    xyz = pow(x,a)*pow(y,b)*pow(z,c);

    result += weight * hi * xyz;
  }

  // multiply with prefactor
  result *= 8 * cellsize.x * cellsize.y * cellsize.z;
  result /= -4*M_PI;

  return result;
}

// reuse Nxx and Nxy by permutating the arguments to implement the other kernel
// components
real calcAsymptoticNyy(int3 idx, real3 cs, int* expansionNxxptr, size_t sizeNxx) {
  return calcAsymptoticNxx({idx.y, idx.z, idx.x}, {cs.y, cs.z, cs.x}, expansionNxxptr, sizeNxx);
}
real calcAsymptoticNzz(int3 idx, real3 cs, int* expansionNxxptr, size_t sizeNxx) {
  return calcAsymptoticNxx({idx.z, idx.x, idx.y}, {cs.z, cs.x, cs.y}, expansionNxxptr, sizeNxx);
}
real calcAsymptoticNxy(int3 idx, real3 cs, int* expansionNxyptr, size_t sizeNxy) {
  return calcAsymptoticNxx({idx.x, idx.y, idx.z}, {cs.x, cs.y, cs.z}, expansionNxyptr, sizeNxy);
}
real calcAsymptoticNxz(int3 idx, real3 cs, int* expansionNxyptr, size_t sizeNxy) {
  return calcAsymptoticNxx({idx.x, idx.z, idx.y}, {cs.x, cs.z, cs.y}, expansionNxyptr, sizeNxy);
}
real calcAsymptoticNyx(int3 idx, real3 cs, int* expansionNxyptr, size_t sizeNxy) {
  return calcAsymptoticNxx({idx.y, idx.x, idx.z}, {cs.y, cs.x, cs.z}, expansionNxyptr, sizeNxy);
}
real calcAsymptoticNyz(int3 idx, real3 cs, int* expansionNxyptr, size_t sizeNxy) {
  return calcAsymptoticNxx({idx.y, idx.z, idx.x}, {cs.y, cs.z, cs.x}, expansionNxyptr, sizeNxy);
}
real calcAsymptoticNzx(int3 idx, real3 cs, int* expansionNxyptr, size_t sizeNxy) {
  return calcAsymptoticNxx({idx.z, idx.x, idx.y}, {cs.z, cs.x, cs.y}, expansionNxyptr, sizeNxy);
}
real calcAsymptoticNzy(int3 idx, real3 cs, int* expansionNxyptr, size_t sizeNxy) {
  return calcAsymptoticNxx({idx.z, idx.y, idx.x}, {cs.z, cs.y, cs.x}, expansionNxyptr, sizeNxy);
}
