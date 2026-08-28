#include "altermagnet.hpp"
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
  if (!neel.cellInGeometry(idx)) {
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

Field evalNeelvector(const HostMagnet* magnet) {
  // Calculate a weighted Neel vector (dimensionless) to account for ferrimagnets
  if (magnet->sublattices().size() != 2)
    throw std::runtime_error("Cannot compute the Néel vector if the magnet has no two sublattices");

  Field neel(magnet->system(), 3);

  if (magnet->sublattices()[0]->msat.assuredZero() && magnet->sublattices()[1]->msat.assuredZero()) {
    neel.makeZero();
    return neel;
  }
  auto sub1 = magnet->sublattices()[0];
  auto sub2 = magnet->sublattices()[1];
  cudaLaunch(neel.grid().ncells(), k_neelvector, neel.cu(),
             sub1->magnetization()->field().cu(),
             sub2->magnetization()->field().cu(),
             sub1->msat.cu(), sub2->msat.cu());
  return neel;
}

AFM_FieldQuantity neelVectorQuantity(const Antiferromagnet* magnet) {
    return AFM_FieldQuantity(magnet, evalNeelvector, 3, "neel_vector", "");
}

ATM_FieldQuantity neelVectorQuantity(const Altermagnet* magnet) {
    return ATM_FieldQuantity(magnet, evalNeelvector, 3, "neel_vector", "");
}
