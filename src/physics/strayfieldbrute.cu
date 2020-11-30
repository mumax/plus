#include <memory>

#include "constants.hpp"
#include "cudalaunch.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "parameter.hpp"
#include "strayfieldbrute.hpp"
#include "system.hpp"

__global__ void k_demagfield(CuField hField,
                             const CuField mField,
                             const CuField kernel,
                             const CuParameter msat) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!hField.cellInGrid(idx))
    return;

  real3 h{0, 0, 0};

  int3 dstcoo = hField.system.grid.index2coord(idx);

  for (int i = 0; i < mField.system.grid.ncells(); i++) {
    int3 srccoo = mField.system.grid.index2coord(i);
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

StrayFieldBruteExecutor::StrayFieldBruteExecutor(
    const Ferromagnet* magnet,
    std::shared_ptr<const System> system)
    : StrayFieldExecutor(magnet, system),
      kernel_(system->grid(), magnet_->grid(), magnet_->world()) {}

Field StrayFieldBruteExecutor::exec() const {
  Field h(system_, 3);
  int ncells = h.grid().ncells();
  cudaLaunch(ncells, k_demagfield, h.cu(),
             magnet_->magnetization()->field().cu(), kernel_.field().cu(),
             magnet_->msat.cu());
  return h;
}
