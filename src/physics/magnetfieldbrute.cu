#include "cudalaunch.hpp"
#include "field.hpp"
#include "magnetfieldbrute.hpp"
#include "parameter.hpp"

// TODO: figure out why including constants.hpp leads to errors
__device__ static const real PI = 3.14159265358979323846;
__device__ static const real MU0 = 4 * PI * 1e-7;

__global__ void k_demagfield(CuField hField,
                             const CuField mField,
                             const CuField kernel,
                             const CuParameter msat) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!hField.cellInGrid(idx))
    return;

  real3 h{0, 0, 0};

  int3 dstcoo = hField.grid.index2coord(idx);

  for (int i = 0; i < mField.grid.ncells(); i++) {
    int3 srccoo = mField.grid.index2coord(i);
    int3 r = dstcoo - srccoo;
    real nxx = kernel.valueAt(r, 0);
    real nyy = kernel.valueAt(r, 1);
    real nzz = kernel.valueAt(r, 2);
    real nxy = kernel.valueAt(r, 3);
    real nxz = kernel.valueAt(r, 4);
    real nyz = kernel.valueAt(r, 5);

    real3 M = msat.valueAt(i) * mField.vectorAt(i);

    h.x -= nxx * M.x + nxy * M.y + nxz * M.z;
    h.y -= nxy * M.x + nyy * M.y + nyz * M.z;
    h.z -= nxz * M.x + nyz * M.y + nzz * M.z;
  }

  hField.setVectorInCell(idx, MU0 * h);
}

MagnetFieldBruteExecutor::MagnetFieldBruteExecutor(Grid gridOut,
                                                   Grid gridIn,
                                                   real3 cellsize)
    : kernel_(gridOut, gridIn, cellsize) {}

void MagnetFieldBruteExecutor::exec(Field* h,
                                    const Field* m,
                                    const Parameter* msat) const {
  // TODO: check dimensions of fields
  int ncells = h->grid().ncells();
  cudaLaunch(ncells, k_demagfield, h->cu(), m->cu(), kernel_.field().cu(), msat->cu());
}