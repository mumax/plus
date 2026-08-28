#include "antiferromagnet.hpp"
#include "cudalaunch.hpp"
#include "datatypes.hpp"
#include "energy.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "local_dmi.hpp"
#include "ncafm.hpp"
#include "parameter.hpp"

bool homoDmiAssuredZero(const Ferromagnet* magnet) {
  // Function returns true if magnet is no sublattice
  bool dmiVectorIsZero = true;
  if (magnet->hostMagnet())
    dmiVectorIsZero = magnet->hostMagnet()->dmiVector.assuredZero();
  return magnet->msat.assuredZero() || dmiVectorIsZero;
}

__global__ void k_homoDmiFieldAFM(CuField hField,
                                  const CuField m2Field,
                                  const CuVectorParameter dmiVector,
                                  const CuParameter msat,
                                  const real symmetry_factor) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto system = hField.system;
 
  if (!system.grid.cellInGrid(idx))
    return;

  // When outside the geometry or msat=0, set to zero and return early
  if (!system.inGeometry(idx) || (msat.valueAt(idx) == 0)) {
    hField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  real3 m2 = m2Field.vectorAt(idx);
  real3 D = dmiVector.vectorAt(idx);
  real3 h0 = hField.vectorAt(idx);

  hField.setVectorInCell(idx, h0 + symmetry_factor * cross(D, m2) / msat.valueAt(idx));
}

__global__ void k_homoDmiFieldNcAfm(CuField hField,
                                  const CuField m2Field,
                                  const CuField m3Field,
                                  const CuVectorParameter dmiVector,
                                  const CuParameter msat,
                                  const real symmetry_factor) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto system = hField.system;

  if (!system.grid.cellInGrid(idx))
    return;

  // When outside the geometry or msat=0, set to zero and return early
  if (!system.inGeometry(idx) || (msat.valueAt(idx) == 0)) {
    hField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  real3 m2 = m2Field.vectorAt(idx);
  real3 m3 = m3Field.vectorAt(idx);
  real3 D = dmiVector.vectorAt(idx);
  hField.setVectorInCell(idx, symmetry_factor * cross(D, m2 - m3) / msat.valueAt(idx));
}

Field evalHomoDmiField(const Ferromagnet* magnet) {
  Field hField(magnet->system(), 3, real3{0, 0, 0});
  if (homoDmiAssuredZero(magnet)) {
    hField.makeZero();
    return hField;
  }

  int ncells = hField.grid().ncells();
  auto msat = magnet->msat.cu();
  auto host = magnet->hostMagnet();
  auto D = host->dmiVector.cu();
  auto subs = host->getOtherSublattices(magnet);

  int i = host->getSublatticeIndex(magnet);
  for (auto sub : subs) {
    auto m2 = sub->magnetization()->field().cu();
    real symmetry_factor = (i % 2 == 0) ? 1 : -1;
    cudaLaunch(ncells, k_homoDmiFieldAFM, hField.cu(), m2, D, msat, symmetry_factor);
    i += 1;
  }
  return hField;
}

Field evalHomoDmiEnergyDensity(const Ferromagnet* magnet) {
  if (homoDmiAssuredZero(magnet))
    return Field(magnet->system(), 1, 0.0);

  return evalEnergyDensity(magnet, evalHomoDmiField(magnet), 0.5);
}

real evalHomoDmiEnergy(const Ferromagnet* magnet) {
  if (homoDmiAssuredZero(magnet))
    return 0;

  real edens = homoDmiEnergyDensityQuantity(magnet).average()[0];

  int ncells = magnet->grid().ncells();
  real cellVolume = magnet->world()->cellVolume();
  return ncells * edens * cellVolume;
}

FM_FieldQuantity homoDmiFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalHomoDmiField, 3, "homogeneous_dmi_field", "T");
}

FM_FieldQuantity homoDmiEnergyDensityQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalHomoDmiEnergyDensity, 1,
                          "homogeneous_dmi_emergy_density", "J/m3");
}

FM_ScalarQuantity homoDmiEnergyQuantity(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, evalHomoDmiEnergy, "homogeneous_dmi_energy", "J");
}
