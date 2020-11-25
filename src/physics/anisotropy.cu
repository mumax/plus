#include "anisotropy.hpp"
#include "cudalaunch.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "world.hpp"

bool anisotropyAssuredZero(const Ferromagnet* magnet) {
  return (magnet->ku1.assuredZero() && magnet->ku2.assuredZero()) ||
         magnet->anisU.assuredZero() || magnet->msat.assuredZero();
}

__global__ void k_anisotropyField(CuField hField,
                                  const CuField mField,
                                  const CuVectorParameter anisU,
                                  const CuParameter Ku1,
                                  const CuParameter Ku2,
                                  const CuParameter msat) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (!hField.cellInGrid(idx))
    return;

  if (msat.valueAt(idx) == 0) {
    hField.setVectorInCell(idx, {0, 0, 0});
    return;
  }

  real3 u = normalized(anisU.vectorAt(idx));
  real3 m = mField.vectorAt(idx);
  real k1 = Ku1.valueAt(idx);
  real k2 = Ku2.valueAt(idx);
  real Ms = msat.valueAt(idx);

  real mu = dot(m, u);

  real3 h = (2 * k1 * mu + 4 * k2 * mu * mu * mu) * u / Ms;

  hField.setVectorInCell(idx, h);
}

Field evalAnisotropyField(const Ferromagnet* magnet) {
  Field result(magnet, 3);
  if (anisotropyAssuredZero(magnet)) {
    result.makeZero();
    return result;
  }
  CuField h = result.cu();
  const CuField m = magnet->magnetization()->field().cu();
  auto anisU = magnet->anisU.cu();
  auto ku1 = magnet->ku1.cu();
  auto ku2 = magnet->ku2.cu();
  auto msat = magnet->msat.cu();
  int ncells = magnet->grid().ncells();
  cudaLaunch(ncells, k_anisotropyField, h, m, anisU, ku1, ku2, msat);
  return result;
}

__global__ void k_anisotropyEnergyDensity(CuField edens,
                                          const CuField mField,
                                          const CuVectorParameter anisU,
                                          const CuParameter Ku1,
                                          const CuParameter Ku2,
                                          const CuParameter msat) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (!edens.cellInGrid(idx))
    return;

  if (msat.valueAt(idx) == 0.0) {
    edens.setValueInCell(idx, 0, 0.0);
    return;
  }

  real3 u = normalized(anisU.vectorAt(idx));
  real3 m = mField.vectorAt(idx);
  real k1 = Ku1.valueAt(idx);
  real k2 = Ku2.valueAt(idx);

  real mu = dot(m, u);

  real e = 0.0;
  e -= k1 * mu * mu;
  e -= k2 * mu * mu * mu * mu;
  edens.setValueInCell(idx, 0, e);
}

Field evalAnisotropyEnergyDensity(const Ferromagnet* magnet) {
  Field edens(magnet, 1);
  if (anisotropyAssuredZero(magnet)) {
    edens.makeZero();
    return edens;
  }

  CuField e = edens.cu();
  const CuField m = magnet->magnetization()->field().cu();
  auto anisU = magnet->anisU.cu();
  auto ku1 = magnet->ku1.cu();
  auto ku2 = magnet->ku1.cu();
  auto msat = magnet->msat.cu();
  int ncells = magnet->grid().ncells();
  cudaLaunch(ncells, k_anisotropyEnergyDensity, e, m, anisU, ku1, ku2, msat);
  return edens;
}

real evalAnisotropyEnergy(const Ferromagnet* magnet) {
  if (anisotropyAssuredZero(magnet))
    return 0;

  real edens = anisotropyEnergyDensityQuantity(magnet).average()[0];
  int ncells = magnet->grid().ncells();
  real cellVolume = magnet->world()->cellVolume();
  return ncells * edens * cellVolume;
}

FM_FieldQuantity anisotropyFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalAnisotropyField, 3, "anisotropy_field",
                          "T");
}

FM_FieldQuantity anisotropyEnergyDensityQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalAnisotropyEnergyDensity, 1,
                          "anisotropy_energy_density", "J/m3");
}

FM_ScalarQuantity anisotropyEnergyQuantity(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, evalAnisotropyEnergy, "anisotropy_energy",
                           "J");
}
