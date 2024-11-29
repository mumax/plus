#include "cudalaunch.hpp"
#include "elastodynamics.hpp"
#include "poyntingvector.hpp"
#include "stresstensor.hpp"


__global__ void k_poyntingVector(CuField poyntingField,
                                 const CuField stress,
                                 const CuField velocity) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = poyntingField.system;

  // When outside the geometry, set to zero and return early
  if (!system.inGeometry(idx)) {
    if (system.grid.cellInGrid(idx)) {
      poyntingField.setVectorInCell(idx, real3{0, 0, 0});
    }
    return;
  }

  real value;
  int stressComp;
  for (int i = 0; i < 3; i++) {
    value = 0;
    for (int j = 0; j < 3; j++) {
      stressComp = (i == j) ? i : i+j+2;
      value += - stress.valueAt(idx, stressComp) * velocity.valueAt(idx, j);
    }
    poyntingField.setValueInCell(idx, i, value);
  }
}

Field evalPoyntingVector(const Magnet* magnet) {
  Field poyntingField(magnet->system(), 3);
  if (elasticityAssuredZero(magnet)) {
    poyntingField.makeZero();
    return poyntingField;
  }

  int ncells = poyntingField.grid().ncells();
  Field stress = evalStressTensor(magnet);
  CuField velocity = magnet->elasticVelocity()->field().cu();

  cudaLaunch(ncells, k_poyntingVector, poyntingField.cu(), stress.cu(), velocity);
  return poyntingField;
}

M_FieldQuantity poyntingVectorQuantity(const Magnet* magnet) {
  return M_FieldQuantity(magnet, evalPoyntingVector, 3, "poynting_vector", "W/m2");
}