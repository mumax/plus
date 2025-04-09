#pragma once

#include "datatypes.hpp"
#include "gpubuffer.hpp"

std::vector<std::vector<int>> dx(std::vector<std::vector<int>>);
std::vector<std::vector<int>> dy(std::vector<std::vector<int>>);
std::vector<std::vector<int>> dz(std::vector<std::vector<int>>);

std::vector<std::vector<int>> cleanup(std::vector<std::vector<int>>);

void combinationsRecursive(
    const std::vector<int>&,
    int,
    int,
    std::vector<int>&,
    std::vector<std::vector<int>>&
);

std::vector<std::vector<int>> derivativeCombinations(int, int);
std::vector<int> uptoOrder(int, std::vector<std::vector<int>>);


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
