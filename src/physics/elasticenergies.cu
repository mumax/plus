#include "cudalaunch.hpp"
#include "elasticenergies.hpp"
#include "elastodynamics.hpp"
#include "energy.hpp"
#include "magnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "straintensor.hpp"
#include "stresstensor.hpp"


// ========== Kinetic Energy ==========

bool kineticEnergyAssuredZero(const Magnet* magnet) {
  return ((!magnet->enableElastodynamics()) || magnet->rho.assuredZero());
}

__global__ void k_kineticEnergyDensity(CuField kinField,
                                const CuField velocity,
                                const CuParameter rho) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = kinField.system;

  // When outside the geometry, set to zero and return early
  if (!system.inGeometry(idx)) {
    if (system.grid.cellInGrid(idx)) {
      kinField.setValueInCell(idx, 0, 0);
    }
    return;
  }

  real3 v = velocity.vectorAt(idx);
  kinField.setValueInCell(idx, 0, 0.5 * dot(v, v) * rho.valueAt(idx));
}

Field evalKineticEnergyDensity(const Magnet* magnet) {
  Field kinField(magnet->system(), 1);
  if (kineticEnergyAssuredZero(magnet)) {
    kinField.makeZero();
    return kinField;
  }

  int ncells = kinField.grid().ncells();
  CuField velocity = magnet->elasticVelocity()->field().cu();
  CuParameter rho = magnet->rho.cu();
  cudaLaunch(ncells, k_kineticEnergyDensity, kinField.cu(), velocity, rho);
  return kinField;
}

real evalKineticEnergy(const Magnet* magnet) {
  if (kineticEnergyAssuredZero(magnet))
    return 0.0;

  real edens = kineticEnergyDensityQuantity(magnet).average()[0];
  return energyFromEnergyDensity(magnet, edens);
}

M_FieldQuantity kineticEnergyDensityQuantity(const Magnet* magnet) {
  return M_FieldQuantity(magnet, evalKineticEnergyDensity, 1, "kinetic_energy_density", "J/m3");
}

M_ScalarQuantity kineticEnergyQuantity(const Magnet* magnet) {
  return M_ScalarQuantity(magnet, evalKineticEnergy, "kinetic_energy", "J");
}

// ========== Elastic Energy ==========

__global__ void k_elasticEnergyDensity(CuField elField,
                                  const CuField stress,
                                  const CuField strain) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = elField.system;

  // When outside the geometry, set to zero and return early
  if (!system.inGeometry(idx)) {
    if (system.grid.cellInGrid(idx)) {
      elField.setValueInCell(idx, 0, 0);
    }
    return;
  }

  real value = 0;
  for (int i = 0; i < 3; i++){
    value += 0.5 * stress.valueAt(idx, i) * strain.valueAt(idx, i);
    value += stress.valueAt(idx, i + 3) * strain.valueAt(idx, i + 3);
  }

  elField.setValueInCell(idx, 0, value);
}

Field evalElasticEnergyDensity(const Magnet* magnet) {
  Field elField(magnet->system(), 1);
  if (elasticityAssuredZero(magnet)) {
    elField.makeZero();
    return elField;
  }

  int ncells = elField.grid().ncells();
  Field stress = evalStressTensor(magnet);
  Field strain = evalStrainTensor(magnet);
  cudaLaunch(ncells, k_elasticEnergyDensity, elField.cu(), stress.cu(), strain.cu());
  return elField;
}

real evalElasticEnergy(const Magnet* magnet) {
  if (elasticityAssuredZero(magnet))
    return 0.0;

  real edens = elasticEnergyDensityQuantity(magnet).average()[0];
  return energyFromEnergyDensity(magnet, edens);
}

M_FieldQuantity elasticEnergyDensityQuantity(const Magnet* magnet) {
  return M_FieldQuantity(magnet, evalElasticEnergyDensity, 1, "elastic_energy_density", "J/m3");
}

M_ScalarQuantity elasticEnergyQuantity(const Magnet* magnet) {
  return M_ScalarQuantity(magnet, evalElasticEnergy, "elastic_energy", "J");
}