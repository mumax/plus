#include "anisotropy.hpp"
#include "cudalaunch.hpp"
#include "demag.hpp"
#include "dmi.hpp"
#include "energy.hpp"
#include "exchange.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
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
  // there will probably be exchange and need to construct a Field anyway
  Field edens = evalExchangeEnergyDensity(magnet);
  if (!anisotropyAssuredZero(magnet)) {edens += evalAnisotropyEnergyDensity(magnet);}
  if (!externalFieldAssuredZero(magnet)) {edens += evalZeemanEnergyDensity(magnet);}
  if (!dmiAssuredZero(magnet)) {edens += evalDmiEnergyDensity(magnet);}
  if (!demagFieldAssuredZero(magnet)) {edens += evalDemagEnergyDensity(magnet);}
  return edens;
}

real evalTotalEnergy(const Ferromagnet* magnet) {
  int ncells = magnet->grid().ncells();
  real edensAverage = totalEnergyDensityQuantity(magnet).average()[0];
  real cellVolume = magnet->world()->cellVolume();
  return ncells * edensAverage * cellVolume;
}

FM_FieldQuantity totalEnergyDensityQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalTotalEnergyDensity, 1, "total_energy_density", "J/m3");
}

FM_ScalarQuantity totalEnergyQuantity(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, evalTotalEnergy, "total_energy", "J");
}