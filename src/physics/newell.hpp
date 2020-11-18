#pragma once

#include "datatypes.hpp"

/** Computes demagkernel component Nxx using Newell's method.
 *  @param idx       distance (in cells) between the source and destiny cell
 *  @param cellsize  cellsize of the grid
 *  @see https://doi.org/10.1029/93JB00694
 */
__host__ __device__ real calcNewellNxx(int3 idx, real3 cellsize);

/** Computes demagkernel component Nyy using Newell's method.
 *  @param idx       distance (in cells) between the source and destiny cell
 *  @param cellsize  cellsize of the grid
 *  @see https://doi.org/10.1029/93JB00694
 */
__host__ __device__ real calcNewellNyy(int3 idx, real3 cellsize);

/** Computes demagkernel component Nzz using Newell's method.
 *  @param idx       distance (in cells) between the source and destiny cell
 *  @param cellsize  cellsize of the grid
 *  @see https://doi.org/10.1029/93JB00694
 */
__host__ __device__ real calcNewellNzz(int3 idx, real3 cellsize);

/** Computes demagkernel component Nxy using Newell's method.
 *  @param idx       distance (in cells) between the source and destiny cell
 *  @param cellsize  cellsize of the grid
 *  @see https://doi.org/10.1029/93JB00694
 */
__host__ __device__ real calcNewellNxy(int3 idx, real3 cellsize);

/** Computes demagkernel component Nxz using Newell's method.
 *  @param idx       distance (in cells) between the source and destiny cell
 *  @param cellsize  cellsize of the grid
 *  @see https://doi.org/10.1029/93JB00694
 */
__host__ __device__ real calcNewellNxz(int3 idx, real3 cellsize);

/** Computes demagkernel component Nyx using Newell's method.
 *  @param idx       distance (in cells) between the source and destiny cell
 *  @param cellsize  cellsize of the grid
 *  @see https://doi.org/10.1029/93JB00694
 */
__host__ __device__ real calcNewellNyx(int3 idx, real3 cellsize);

/** Computes demagkernel component Nyz using Newell's method.
 *  @param idx       distance (in cells) between the source and destiny cell
 *  @param cellsize  cellsize of the grid
 *  @see https://doi.org/10.1029/93JB00694
 */
__host__ __device__ real calcNewellNyz(int3 idx, real3 cellsize);

/** Computes demagkernel component Nzx using Newell's method.
 *  @param idx       distance (in cells) between the source and destiny cell
 *  @param cellsize  cellsize of the grid
 *  @see https://doi.org/10.1029/93JB00694
 */
__host__ __device__ real calcNewellNzx(int3 idx, real3 cellsize);

/** Computes demagkernel component Nzy using Newell's method.
 *  @param idx       distance (in cells) between the source and destiny cell
 *  @param cellsize  cellsize of the grid
 *  @see https://doi.org/10.1029/93JB00694
 */
__host__ __device__ real calcNewellNzy(int3 idx, real3 cellsize);