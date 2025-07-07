#include <memory>

#include "constants.hpp"
#include "cudalaunch.hpp"
#include "fullmag.hpp"
#include "magnet.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "grid.hpp"
#include "parameter.hpp"
#include "strayfieldbrute.hpp"
#include "system.hpp"

__global__ void k_demagfield(CuField hField,
                             const CuField mField,
                             const CuField kernel,
                             CuParameter msat) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry of destiny field, set to zero and return
  // early
  if (!hField.cellInGeometry(idx)) {
    if (hField.cellInGrid(idx))
      hField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  int3 dstcoo = hField.system.grid.index2coord(idx);
  real3 h{0, 0, 0};

  for (int i = 0; i < mField.system.grid.ncells(); i++) {
    if (!mField.cellInGeometry(i))
      continue;

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
    const Magnet* magnet,
    std::shared_ptr<const System> system, int order, double eps, double switchingradius)
    : StrayFieldExecutor(magnet, system),
      kernel_(system->grid(), magnet_->grid(), magnet_->world(), order, eps, switchingradius) {}

Field StrayFieldBruteExecutor::exec() const {
  
  Field h(system_, 3);
  int ncells = h.grid().ncells();

  if(const Ferromagnet* mag = magnet_->asFM()) {
    auto m = mag->magnetization()->field().cu();
    auto msat = mag->msat.cu();
    cudaLaunch(ncells, k_demagfield, h.cu(), m, kernel_.field().cu(), msat);
  }
  else {
    auto hostmag = evalHMFullMag(magnet_->asHost());
    auto msat = Parameter(magnet_->system(), 1.0);
    cudaLaunch(ncells, k_demagfield, h.cu(), hostmag.cu(), kernel_.field().cu(), msat.cu());
  }
  return h;
}
