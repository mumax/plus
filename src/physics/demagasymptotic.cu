#include "demagasymptotic.hpp"

#include "gpubuffer.hpp"
#include <cstdio>


// Function to calculate factorials
__host__ __device__ int fac(int val) {
  int result = 1;
  for (int i = 1; i <= val; i++) {
    result *= i;
  }
  return result;
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
  
  for (int i = 0; i < sizeNxx; i+=8) {
    int N = expansionNxxptr[i];
    int a = expansionNxxptr[i+1];
    int b = expansionNxxptr[i+2];
    int c = expansionNxxptr[i+3];
    int P = expansionNxxptr[i+4];
    int d = expansionNxxptr[i+5];
    int e = expansionNxxptr[i+6];
    int f = expansionNxxptr[i+7];

    double A = fac(d+2) * fac(e+2) * fac(f+2);
    double weight = N/A;

    double hi = pow(hx,d)*pow(hy,e)*pow(hz,f);
    double xyz = pow(x,a)*pow(y,b)*pow(z,c);

    double Rpow = pow(R,P);

    result += weight * hi * xyz / Rpow;
  }

  // multiply with prefactor
  result *= 8 * cellsize.x * cellsize.y * cellsize.z;
  result /= -4*M_PI;

  return result;
}

/** Calculate the asymptotic solution of the demagkernel component Nxy.
    This method is based on the method used in OOMMF.
    https://math.nist.gov/~MDonahue/talks/mmm2020-talk.pdf
 */
__host__ __device__ real calcAsymptoticNxy(int3 idx, real3 cellsize, int* expansionNxyptr, size_t sizeNxy) {

  double hx = cellsize.x;
  double hy = cellsize.y;
  double hz = cellsize.z;
  double x = idx.x * hx;
  double y = idx.y * hy;
  double z = idx.z * hz;
  float R = sqrtf(x*x + y*y + z*z);
  
  double result = 0;

  for (int i = 0; i < sizeNxy; i+=8) {
    int N = expansionNxyptr[i];
    int a = expansionNxyptr[i+1];
    int b = expansionNxyptr[i+2];
    int c = expansionNxyptr[i+3];
    int P = expansionNxyptr[i+4];
    int d = expansionNxyptr[i+5];
    int e = expansionNxyptr[i+6];
    int f = expansionNxyptr[i+7];

    double A = fac(d+2) * fac(e+2) * fac(f+2);
    double weight = N/A;

    double hi = pow(hx,d)*pow(hy,e)*pow(hz,f);
    double xyz = pow(x,a)*pow(y,b)*pow(z,c);

    double Rpow = pow(R,P);

    result += weight * hi * xyz / Rpow;
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
real calcAsymptoticNxz(int3 idx, real3 cs, int* expansionNxyptr, size_t sizeNxy) {
  return calcAsymptoticNxy({idx.x, idx.z, idx.y}, {cs.x, cs.z, cs.y}, expansionNxyptr, sizeNxy);
}
real calcAsymptoticNyx(int3 idx, real3 cs, int* expansionNxyptr, size_t sizeNxy) {
  return calcAsymptoticNxy({idx.y, idx.x, idx.z}, {cs.y, cs.x, cs.z}, expansionNxyptr, sizeNxy);
}
real calcAsymptoticNyz(int3 idx, real3 cs, int* expansionNxyptr, size_t sizeNxy) {
  return calcAsymptoticNxy({idx.y, idx.z, idx.x}, {cs.y, cs.z, cs.x}, expansionNxyptr, sizeNxy);
}
real calcAsymptoticNzx(int3 idx, real3 cs, int* expansionNxyptr, size_t sizeNxy) {
  return calcAsymptoticNxy({idx.z, idx.x, idx.y}, {cs.z, cs.x, cs.y}, expansionNxyptr, sizeNxy);
}
real calcAsymptoticNzy(int3 idx, real3 cs, int* expansionNxyptr, size_t sizeNxy) {
  return calcAsymptoticNxy({idx.z, idx.y, idx.x}, {cs.z, cs.y, cs.x}, expansionNxyptr, sizeNxy);
}
