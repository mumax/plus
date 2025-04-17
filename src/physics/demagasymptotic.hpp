#pragma once

#include "datatypes.hpp"
#include "gpubuffer.hpp"

/** 
 *  The derivatives dx, dy and dz of a function of the form
 *  f(x,y,z) = N * hx^f * hy^g * hz^i * x^a * y^b * z^c / R^P 
 *  where N is a constant, R = sqrt(x² + y² + z²), hx, hy and hz are cell sizes
 *  and x, y and z are coordinates.
 *  These functions can be rewritten as vectors (N,a,b,c,P,f,g,i).
 *  dx, dy and dz accept a vector containing vectors of this form return the
 *  derivativesand as a vector containing vectors of that same form

 *  dx: (N,a,b,c,P,f,g,i) --> (-N(P-a),a+1,b,c,P+2,f+1,g,i) +
                              (N*a,a-1,b+2,c,P+2,f+1,g,i) + 
                              (N*a,a-1,b,c+2,P+2,f+1,g,i)
    
 *   dy: (N,a,b,c,P,f,g,i) --> (-N(P-b),a,b+1,c,P+2,f,g+1,i) +
                               (N*b,a+2,b-1,c,P+2,f,g+1,i) + 
                               (N*b,a,b-1,c+2,P+2,f,g+1,i)
    
 *   dz: (N,a,b,c,P,f,g,i) --> (-N(P-c),a,b,c+1,P+2,f,g,i+1) +
                               (N*c,a+2,b,c-1,P+2,f,g,i+1) + 
                               (N*c,a,b+2,c-1,P+2,f,g,i+1)
 */
std::vector<std::vector<int>> dx(std::vector<std::vector<int>>);
std::vector<std::vector<int>> dy(std::vector<std::vector<int>>);
std::vector<std::vector<int>> dz(std::vector<std::vector<int>>);

/** 
 *  This function cleans up a vector containing vectors of the form
 *  (N,a,b,c,P,f,g,i) by comparing a,b,c,P,f,g and i of one vector with another
 *  vector. If those are all equal, the N values are added together and one of
 *  them is removed.
 */
std::vector<std::vector<int>> cleanup(std::vector<std::vector<int>>);

// Determines the derivatives combinations recursively up to a specified order.
void combinationsRecursive(
    const std::vector<int>&,
    int,
    int,
    std::vector<int>&,
    std::vector<std::vector<int>>&
);
// Determine the amount of derivatives to x, y and z up to a given order.
std::vector<std::vector<int>> derivativeCombinations(int);

/** Determine all terms in the asymptotic expansion. Inputs are an order and a
 *  vector containing vectors of the form (N,a,b,c,P,f,g,i).
 */
std::vector<int> upToOrder(int, std::vector<std::vector<int>>);


/** Computes demagkernel component Nxx using asymptotic method.
 *  @param idx       distance (in cells) between the source and destiny cell
 *  @param cellsize  cellsize of the grid
 *  @see https://math.nist.gov/~MDonahue/talks/mmm2020-talk.pdf
 */
__host__ __device__ real calcAsymptoticNxx(int3 idx, real3 cellsize, int* expansionNxx, size_t sizeNxx);

/** Computes demagkernel component Nyy using asymptotic method.
 *  @param idx       distance (in cells) between the source and destiny cell
 *  @param cellsize  cellsize of the grid
 *  @see https://math.nist.gov/~MDonahue/talks/mmm2020-talk.pdf
 */
__host__ __device__ real calcAsymptoticNyy(int3 idx, real3 cellsize, int* expansionNxx, size_t sizeNxx);

/** Computes demagkernel component Nzz using asymptotic method.
 *  @param idx       distance (in cells) between the source and destiny cell
 *  @param cellsize  cellsize of the grid
 *  @see https://math.nist.gov/~MDonahue/talks/mmm2020-talk.pdf
 */
__host__ __device__ real calcAsymptoticNzz(int3 idx, real3 cellsize, int* expansionNxx, size_t sizeNxx);

/** Computes demagkernel component Nxy using asymptotic method.
 *  @param idx       distance (in cells) between the source and destiny cell
 *  @param cellsize  cellsize of the grid
 *  @see https://math.nist.gov/~MDonahue/talks/mmm2020-talk.pdf
 */
__host__ __device__ real calcAsymptoticNxy(int3 idx, real3 cellsize, int* expansionNxy, size_t sizeNxy);

/** Computes demagkernel component Nxz using asymptotic method.
 *  @param idx       distance (in cells) between the source and destiny cell
 *  @param cellsize  cellsize of the grid
 *  @see https://math.nist.gov/~MDonahue/talks/mmm2020-talk.pdf
 */
__host__ __device__ real calcAsymptoticNxz(int3 idx, real3 cellsize, int* expansionNxy, size_t sizeNxy);

/** Computes demagkernel component Nyx using asymptotic method.
 *  @param idx       distance (in cells) between the source and destiny cell
 *  @param cellsize  cellsize of the grid
 *  @see https://math.nist.gov/~MDonahue/talks/mmm2020-talk.pdf
 */
__host__ __device__ real calcAsymptoticNyx(int3 idx, real3 cellsize, int* expansionNxy, size_t sizeNxy);

/** Computes demagkernel component Nyz using asymptotic method.
 *  @param idx       distance (in cells) between the source and destiny cell
 *  @param cellsize  cellsize of the grid
 *  @see https://math.nist.gov/~MDonahue/talks/mmm2020-talk.pdf
 */
__host__ __device__ real calcAsymptoticNyz(int3 idx, real3 cellsize, int* expansionNxy, size_t sizeNxy);

/** Computes demagkernel component Nzx using asymptotic method.
 *  @param idx       distance (in cells) between the source and destiny cell
 *  @param cellsize  cellsize of the grid
 *  @see https://math.nist.gov/~MDonahue/talks/mmm2020-talk.pdf
 */
__host__ __device__ real calcAsymptoticNzx(int3 idx, real3 cellsize, int* expansionNxy, size_t sizeNxy);

/** Computes demagkernel component Nzy using asymptotic method.
 *  @param idx       distance (in cells) between the source and destiny cell
 *  @param cellsize  cellsize of the grid
 *  @see https://math.nist.gov/~MDonahue/talks/mmm2020-talk.pdf
 */
__host__ __device__ real calcAsymptoticNzy(int3 idx, real3 cellsize,int* expansionNxy, size_t sizeNxy);
