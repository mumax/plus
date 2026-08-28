#include "anisotropy.hpp"
#include "cudalaunch.hpp"
#include "energy.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "world.hpp"
#include <stdio.h>

bool unianisotropyAssuredZero(const Ferromagnet* magnet) {
  return (magnet->ku1.assuredZero() && magnet->ku2.assuredZero())
       || magnet->anisU.assuredZero() || magnet->msat.assuredZero();
}

bool cubicanisotropyAssuredZero(const Ferromagnet* magnet) {
  return (magnet->kc1.assuredZero() && magnet->kc2.assuredZero() && magnet->kc3.assuredZero())
       || (magnet->anisC1.assuredZero() && magnet->anisC2.assuredZero())
       || magnet->msat.assuredZero();
}

bool anisotropyAssuredZero(const Ferromagnet* magnet) {
  return (unianisotropyAssuredZero(magnet)
          && cubicanisotropyAssuredZero(magnet));
}

__global__ void k_unianisotropyField(CuField hField,
                                  const CuField mField,
                                  const CuVectorParameter anisU,
                                  const CuParameter FM_Ku1,
                                  const CuParameter FM_Ku2,
                                  const CuParameter msat) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!hField.cellInGeometry(idx)) {
    if (hField.cellInGrid(idx))
      hField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  if (msat.valueAt(idx) == 0.) {
      hField.setVectorInCell(idx, real3{0, 0, 0});
      return;
    }

  real3 u = normalized(anisU.vectorAt(idx));
  real3 m = mField.vectorAt(idx);
  real k1 = FM_Ku1.valueAt(idx);
  real k2 = FM_Ku2.valueAt(idx);
  real Ms = msat.valueAt(idx);

  real mu = dot(m, u);

  real3 h = (2 * k1 * mu + 4 * k2 * mu * mu * mu) * u / Ms;
  hField.setVectorInCell(idx, h);
}

__global__ void k_cubicanisotropyField(CuField hField,
                                  const CuField mField,
                                  const CuVectorParameter anisC1,
                                  const CuVectorParameter anisC2,
                                  const CuParameter Kc1,
                                  const CuParameter Kc2,
                                  const CuParameter Kc3,
                                  const CuParameter msat) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!hField.cellInGeometry(idx)) {
    if (hField.cellInGrid(idx))
      hField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  if (msat.valueAt(idx) == 0.) {
      hField.setVectorInCell(idx, real3{0, 0, 0});
      return;
    }

  real3 c1 = normalized(anisC1.vectorAt(idx));
  real3 c2 = normalized(anisC2.vectorAt(idx));
  real3 c3 = cross(c1, c2);
  
  real3 m = mField.vectorAt(idx);

  real kc1 = Kc1.valueAt(idx);
  real kc2 = Kc2.valueAt(idx);
  real kc3 = Kc3.valueAt(idx);

  real Ms = msat.valueAt(idx);

  real c1m = dot(c1, m);
  real c2m = dot(c2, m);
  real c3m = dot(c3, m);

  real3 h = -2 * kc1 * ( (c2m * c2m + c3m * c3m) * (c1m * c1)
            + (c1m * c1m + c3m * c3m) * (c2m * c2)
            + (c1m * c1m + c2m * c2m) * (c3m * c3)) / Ms
            - 2 * kc2 * ((c2m * c2m * c3m * c3m) * (c1m * c1)
            + (c1m * c1m * c3m * c3m) * (c2m * c2)
            + (c1m * c1m * c2m * c2m) * (c3m * c3)) / Ms
            - 4 * kc3 * ((c2m * c2m * c2m * c2m + c3m * c3m * c3m * c3m) * (c1m * c1m * c1m * c1)
            + (c1m * c1m * c1m * c1m + c3m * c3m * c3m * c3m) * (c2m * c2m * c2m * c2)
            + (c1m * c1m * c1m * c1m + c2m * c2m * c2m * c2m) * (c3m * c3m * c3m * c3)) / Ms;

  hField.setVectorInCell(idx, h);
}


Field evalAnisotropyField(const Ferromagnet* magnet) {

  Field result(magnet->system(), 3);
  
  if (anisotropyAssuredZero(magnet)) {
    result.makeZero();
    return result;
  }

  CuField h = result.cu();
  const CuField m = magnet->magnetization()->field().cu();
  auto msat = magnet->msat.cu();
  int ncells = magnet->grid().ncells();

  if (!unianisotropyAssuredZero(magnet)) {
    auto anisU = magnet->anisU.cu();
    auto ku1 = magnet->ku1.cu();
    auto ku2 = magnet->ku2.cu();
    cudaLaunch(ncells, k_unianisotropyField, h, m,
               anisU, ku1, ku2, msat);
  }
  else if(!cubicanisotropyAssuredZero(magnet)) {
    auto anisC1 = magnet->anisC1.cu();
    auto anisC2 = magnet->anisC2.cu();
    auto kc1 = magnet->kc1.cu();
    auto kc2 = magnet->kc2.cu();
    auto kc3 = magnet->kc3.cu();
    cudaLaunch(ncells, k_cubicanisotropyField, h, m,
               anisC1, anisC2, kc1, kc2, kc3, msat);
  }
  return result;
}

__global__ void k_unianisotropyEnergyDensity(CuField edens,
                                          const CuField mField,
                                          const CuVectorParameter anisU,
                                          const CuParameter Ku1,
                                          const CuParameter Ku2,
                                          const CuParameter msat) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!edens.cellInGeometry(idx)) {
    if (edens.cellInGrid(idx)) {
      edens.setValueInCell(idx, 0, 0.0);
    }
    return;
  }

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

__global__ void k_cubanisotropyEnergyDensity(CuField edens,
                                          const CuField mField,
                                          const CuVectorParameter anisC1,
                                          const CuVectorParameter anisC2,
                                          const CuParameter Kc1,
                                          const CuParameter Kc2,
                                          const CuParameter Kc3,
                                          const CuParameter msat) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!edens.cellInGeometry(idx)) {
    if (edens.cellInGrid(idx)) {
      edens.setValueInCell(idx, 0, 0.0);
    }
    return;
  }

  if (msat.valueAt(idx) == 0.0) {
    edens.setValueInCell(idx, 0, 0.0);
    return;
  }

  real3 c1 = normalized(anisC1.vectorAt(idx));
  real3 c2 = normalized(anisC2.vectorAt(idx));
  real3 c3 = cross(c1, c2);

  real kc1 = Kc1.valueAt(idx);
  real kc2 = Kc2.valueAt(idx);
  real kc3 = Kc3.valueAt(idx);
  real3 m = mField.vectorAt(idx);

  real c1m = dot(c1, m);
  real c2m = dot(c2, m);
  real c3m = dot(c3, m);
  
  real e = 0.0;
  e += kc1 * (c1m * c1m * c2m * c2m
            + c1m * c1m * c3m * c3m
            + c2m * c2m * c3m * c3m);
  e += kc2 * c1m * c1m * c2m * c2m * c3m * c3m;
  e += kc3 * (c1m * c1m * c1m * c1m * c2m * c2m * c2m * c2m
            + c1m * c1m * c1m * c1m * c3m * c3m * c3m * c3m
            + c2m * c2m * c2m * c2m * c3m * c3m * c3m * c3m);
  edens.setValueInCell(idx, 0, e); 
}

Field evalAnisotropyEnergyDensity(const Ferromagnet* magnet) {

  Field edens(magnet->system(), 1);

  if (anisotropyAssuredZero(magnet)) {
    edens.makeZero();
    return edens;
  }

  CuField e = edens.cu();

  const CuField m = magnet->magnetization()->field().cu();
  auto msat = magnet->msat.cu();
  int ncells = magnet->grid().ncells();

  if(!unianisotropyAssuredZero(magnet)) {
    auto anisU = magnet->anisU.cu();
    auto ku1 = magnet->ku1.cu();
    auto ku2 = magnet->ku2.cu();
    cudaLaunch(ncells, k_unianisotropyEnergyDensity, e, m,
               anisU, ku1, ku2, msat);
  }
  else if(!cubicanisotropyAssuredZero(magnet)) {
    auto anisC1 = magnet->anisC1.cu();
    auto anisC2 = magnet-> anisC2.cu();
    auto kc1 = magnet->kc1.cu();
    auto kc2 = magnet->kc2.cu();
    auto kc3 = magnet->kc3.cu();
    cudaLaunch(ncells, k_cubanisotropyEnergyDensity, e, m,
               anisC1, anisC2, kc1, kc2, kc3, msat);
  }
  return edens;
}

real evalAnisotropyEnergy(const Ferromagnet* magnet) {
  if (unianisotropyAssuredZero(magnet) && cubicanisotropyAssuredZero(magnet))
    return 0;

  real edens = anisotropyEnergyDensityQuantity(magnet).average()[0];
  return energyFromEnergyDensity(magnet, edens);
}

FM_FieldQuantity anisotropyFieldQuantity(const Ferromagnet* magnet) {

  return FM_FieldQuantity(magnet, evalAnisotropyField, 3, "anisotropy_field", "T");  
}

FM_FieldQuantity anisotropyEnergyDensityQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalAnisotropyEnergyDensity, 1,
                            "anisotropy_energy_density", "J/m3");
}

FM_ScalarQuantity anisotropyEnergyQuantity(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, evalAnisotropyEnergy, "anisotropy_energy", "J");
}
