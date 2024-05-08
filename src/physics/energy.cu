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
                                const CuParameter msat2,
                                const real prefactor,
                                const int comp) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!edens.cellInGeometry(idx)) {
    if (edens.cellInGrid(idx)) {
      if (comp == 3)
        edens.setValueInCell(idx, 0, 0.0);
      else if (comp == 6)
        edens.setValueInCell(idx, real2{0.0, 0.0});
    }
    return;
  }

  if (comp == 3) {
    real Ms = msat.valueAt(idx);
    real3 h = hfield.FM_vectorAt(idx);
    real3 m = mag.FM_vectorAt(idx);
    edens.setValueInCell(idx, 0, -prefactor * Ms * dot(m, h));
  }
  else if (comp == 6) {
    real Ms = msat.valueAt(idx);
    real Ms2 = msat2.valueAt(idx);
    real6 h = hfield.AFM_vectorAt(idx);
    real6 m = mag.AFM_vectorAt(idx);
    edens.setValueInCell(idx, -prefactor * real2{Ms, Ms2} * dot(m, h));
  }
}

Field evalEnergyDensity(const Ferromagnet* magnet,
                        const Field& h,
                        real prefactor) {
  int comp = magnet->magnetization()->ncomp();
  Field edens(magnet->system(), comp / 3);
  if (magnet->msat.assuredZero() && magnet->msat2.assuredZero()) {
    edens.makeZero();
    return edens;
  }

  cudaLaunch(edens.grid().ncells(), k_energyDensity, edens.cu(),
             magnet->magnetization()->field().cu(), h.cu(), magnet->msat.cu(),
             magnet->msat2.cu(), prefactor, comp);
  return edens;
}

Field evalTotalEnergyDensity(const Ferromagnet* magnet) {
  Field edens = evalAnisotropyEnergyDensity(magnet);
  edens += evalExchangeEnergyDensity(magnet);
  edens += evalZeemanEnergyDensity(magnet);
  edens += evalDmiEnergyDensity(magnet);
  if (magnet->magnetization()->ncomp() == 3)
    edens += evalDemagEnergyDensity(magnet); //ignore for now in case of AFM
  return edens;
}

real evalTotalSublatticeEnergy(const Ferromagnet* magnet, const bool sub2) {
  int ncells = magnet->grid().ncells();
  real edensAverage;
  if (!sub2) 
    edensAverage = totalEnergyDensityQuantity(magnet).average()[0];
  else if (sub2)
    edensAverage = totalEnergyDensityQuantity(magnet).average()[1];
  
  real cellVolume = magnet->world()->cellVolume();
  return ncells * edensAverage * cellVolume;
}

real evalTotalEnergy(const Ferromagnet* magnet, const bool sub2) {
  int ncells = magnet->grid().ncells();
  std::vector<real> edensAverage = totalEnergyDensityQuantity(magnet).average();
  real cellVolume = magnet->world()->cellVolume();
  if (magnet->magnetization()->ncomp() == 3)
    return ncells * cellVolume * edensAverage[0];
  else 
    return ncells * cellVolume * (edensAverage[0] + edensAverage[1]);
}

FM_FieldQuantity totalEnergyDensityQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalTotalEnergyDensity, magnet->magnetization()->ncomp() / 3,
                          "total_energy_density", "J/m3");
}

FM_ScalarQuantity totalSublatticeEnergyQuantity(const Ferromagnet* magnet, const bool sub2) {
  std::string name = (sub2) ? "total_energy2" : "total_energy";
  return FM_ScalarQuantity(magnet, evalTotalSublatticeEnergy, sub2, name, "J");
}

FM_ScalarQuantity totalEnergyQuantity(const Ferromagnet* magnet, const bool sub2) {
  return FM_ScalarQuantity(magnet, evalTotalEnergy, sub2, "total_energy", "J");
}