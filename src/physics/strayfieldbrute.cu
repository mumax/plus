#include <memory>

#include "antiferromagnet.hpp"
#include "constants.hpp"
#include "cudalaunch.hpp"
#include "ferromagnet.hpp"
#include "magnet.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "grid.hpp"
#include "parameter.hpp"
#include "strayfieldbrute.hpp"
#include "system.hpp"

__global__ void k_demagfield(CuField hField,
                             const CuField mField1,
                             const CuField mField2,
                             const CuField kernel,
                             const CuParameter msat1,
                             const CuParameter msat2,
                             real fac) {
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

  for (int i = 0; i < mField1.system.grid.ncells(); i++) {
    if (!mField1.cellInGeometry(i))
      continue;

    int3 srccoo = mField1.system.grid.index2coord(i);
    int3 r = dstcoo - srccoo;
    real nxx = kernel.valueAt(r, 0);
    real nyy = kernel.valueAt(r, 1);
    real nzz = kernel.valueAt(r, 2);
    real nxy = kernel.valueAt(r, 3);
    real nxz = kernel.valueAt(r, 4);
    real nyz = kernel.valueAt(r, 5);
    
    real3 M = (msat1.valueAt(i) * mField1.vectorAt(i) +
               msat2.valueAt(i) * mField2.vectorAt(i)) / fac;

    h.x -= nxx * M.x + nxy * M.y + nxz * M.z;
    h.y -= nxy * M.x + nyy * M.y + nyz * M.z;
    h.z -= nxz * M.x + nyz * M.y + nzz * M.z;
  }
  
  hField.setVectorInCell(idx, MU0 * h);
}

StrayFieldBruteExecutor::StrayFieldBruteExecutor(
    const Magnet* magnet,
    std::shared_ptr<const System> system, int order, double switchingradius)
    : StrayFieldExecutor(magnet, system),
      kernel_(system->grid(), magnet_->grid(), magnet_->world(), order, switchingradius) {}

Field StrayFieldBruteExecutor::exec() const {
  
  Field h(system_, 3);
  int ncells = h.grid().ncells();
  real fac;

  if(const Ferromagnet* mag = dynamic_cast<const Ferromagnet*>(magnet_)) {
    auto m = mag->magnetization()->field().cu();
    auto msat = mag->msat.cu();
    fac = 2.0;
    cudaLaunch(ncells, k_demagfield, h.cu(), m, m, kernel_.field().cu(),
              msat, msat, fac);
  }
  else if (const Antiferromagnet* mag = dynamic_cast<const Antiferromagnet*>(magnet_)) {
    auto m1 = mag->sub1()->magnetization()->field().cu();
    auto m2 = mag->sub2()->magnetization()->field().cu();
    auto ms1 = mag->sub1()->msat.cu();
    auto ms2 = mag->sub2()->msat.cu();
    fac = 1.0;
    cudaLaunch(ncells, k_demagfield, h.cu(), m1, m2, kernel_.field().cu(),
              ms1, ms2, fac);
  }
  return h;
}
