#include "antiferromagnet.hpp"
#include "cudalaunch.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "neel.hpp"

__global__ void k_neelvector(CuField neel,
                             const CuField mag1,
                             const CuField mag2,
                             const CuParameter msat1,
                             const CuParameter msat2) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!neel.cellInGeometry(idx) || (!msat1.valueAt(idx) && !msat2.valueAt(idx))) {
    if (neel.cellInGrid(idx)) 
        neel.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }
    real3 m1 = mag1.vectorAt(idx);
    real3 m2 = mag2.vectorAt(idx);
    real ms1 = msat1.valueAt(idx);
    real ms2 = msat2.valueAt(idx);

    neel.setVectorInCell(idx, (ms1 * m1 - ms2 * m2) / (ms1 + ms2));
}

Field evalNeelvector(const Antiferromagnet* magnet) {
  // Calculate a weighted Neel vector (dimensionless) to account for ferrimagnets
  Field neel(magnet->system(), 3);

  if (magnet->sub1()->msat.assuredZero() && magnet->sub2()->msat.assuredZero()) {
    neel.makeZero();
    return neel;
  }
  cudaLaunch(neel.grid().ncells(), k_neelvector, neel.cu(),
             magnet->sub1()->magnetization()->field().cu(),
             magnet->sub2()->magnetization()->field().cu(),
             magnet->sub1()->msat.cu(), magnet->sub2()->msat.cu());
  return neel;
}

AFM_FieldQuantity neelVectorQuantity(const Antiferromagnet* magnet) {
    return AFM_FieldQuantity(magnet, evalNeelvector, 3, "neel_vector", "");
}
