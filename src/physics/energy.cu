#include "afmexchange.hpp"
#include "anisotropy.hpp"
#include "antiferromagnet.hpp"
#include "cudalaunch.hpp"
#include "demag.hpp"
#include "dmi.hpp"
#include "elasticenergies.hpp"
#include "elastodynamics.hpp"
#include "energy.hpp"
#include "exchange.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "magnetoelasticfield.hpp"
#include "world.hpp"
#include "zeeman.hpp"

__global__ void k_energyDensity(CuField edens,
                                const CuField mag,
                                const CuField hfield,
                                const CuParameter msat,
                                const real prefactor) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!edens.cellInGeometry(idx)) {
    if (edens.cellInGrid(idx))
      edens.setValueInCell(idx, 0, 0.0);
    return;
  }
  real Ms = msat.valueAt(idx);
  real3 h = hfield.vectorAt(idx);
  real3 m = mag.vectorAt(idx);
  edens.setValueInCell(idx, 0, -prefactor * Ms * dot(m, h));
}

Field evalEnergyDensity(const Ferromagnet* magnet,
                        const Field& h,
                        real prefactor) {
  Field edens(magnet->system(), 1);
  if (magnet->msat.assuredZero()) {
    edens.makeZero();
    return edens;
  }

  cudaLaunch(edens.grid().ncells(), k_energyDensity, edens.cu(),
             magnet->magnetization()->field().cu(), h.cu(),
             magnet->msat.cu(), prefactor);
  return edens;
}

Field evalTotalEnergyDensity(const Ferromagnet* magnet) {
  Field edens(magnet->system(), 1, 0.0);
  if (!exchangeAssuredZero(magnet)) {edens += evalExchangeEnergyDensity(magnet);}
  if (!anisotropyAssuredZero(magnet)) {edens += evalAnisotropyEnergyDensity(magnet);}
  if (!externalFieldAssuredZero(magnet)) {edens += evalZeemanEnergyDensity(magnet);}
  if (!dmiAssuredZero(magnet)) {edens += evalDmiEnergyDensity(magnet);}
  if (!demagFieldAssuredZero(magnet)) {edens += evalDemagEnergyDensity(magnet);}
  if (!homoAfmExchangeAssuredZero(magnet)) {edens += evalHomoAfmExchangeEnergyDensity(magnet);}
  if (!inHomoAfmExchangeAssuredZero(magnet)) {edens += evalInHomoAfmExchangeEnergyDensity(magnet);}
  
  // magnetoelastics; works if host or if sublattice
  if (!magnetoelasticAssuredZero(magnet)) {edens += evalMagnetoelasticEnergyDensity(magnet);}
  // elastics; only works if independent host
  if (!kineticEnergyAssuredZero(magnet)) {edens += evalKineticEnergyDensity(magnet);}
  if (!elasticityAssuredZero(magnet)) {edens += evalElasticEnergyDensity(magnet);}
  return edens;
}

Field evalTotalEnergyDensity(const Antiferromagnet* magnet) {
  Field edens = evalTotalEnergyDensity(magnet->sub1()) +
                evalTotalEnergyDensity(magnet->sub2());
  if (!kineticEnergyAssuredZero(magnet)) {edens += evalKineticEnergyDensity(magnet);}
  if (!elasticityAssuredZero(magnet)) {edens += evalElasticEnergyDensity(magnet);}
  return edens;
}

real evalTotalEnergy(const Magnet* magnet) {
  int ncells = magnet->grid().ncells();
  real cellVolume = magnet->world()->cellVolume();
  real edensAverage;
  
  if (const Ferromagnet* mag = magnet->asFM())
    edensAverage = totalEnergyDensityQuantity(mag).average()[0];
  else if (const Antiferromagnet* mag = magnet->asAFM())
    edensAverage = totalEnergyDensityQuantity(mag).average()[0];
  else
    throw std::invalid_argument("Cannot calculate energy of instance which"
                                "is no Ferromagnet or Antiferromagnet.");
                 
  return ncells * edensAverage * cellVolume;
}

FM_FieldQuantity totalEnergyDensityQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet,
    static_cast<Field(*)(const Ferromagnet*)>(evalTotalEnergyDensity),
    1, "total_energy_density", "J/m3");
}

FM_ScalarQuantity totalEnergyQuantity(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, evalTotalEnergy, "total_energy", "J");
}

AFM_FieldQuantity totalEnergyDensityQuantity(const Antiferromagnet* magnet) {
  return AFM_FieldQuantity(magnet,
    static_cast<Field(*)(const Antiferromagnet*)>(evalTotalEnergyDensity),
    1, "total_energy_density", "J/m3");
}

AFM_ScalarQuantity totalEnergyQuantity(const Antiferromagnet* magnet) {
  return AFM_ScalarQuantity(magnet, evalTotalEnergy, "total_energy", "J");
}
