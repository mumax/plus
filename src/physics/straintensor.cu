#include "cudalaunch.hpp"
#include "magnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "straintensor.hpp"


bool strainTensorAssuredZero(const Magnet* magnet) {
  return !magnet->enableElastodynamics();
}

// Return 0 if denom is 0, else return quotient
__device__ inline real safeDiv(real num, real denom) {
  if (denom == 0.) return 0.;
  return num / denom;
}

__global__ void k_strainTensor(CuField strain,
                               const CuField stress,  // stress tensor
                               const CuParameter C11,
                               const CuParameter C12,
                               const CuParameter C44) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = strain.system;
  const Grid grid = system.grid;

  // When outside the geometry, set to zero and return early
  if (!system.inGeometry(idx)) {
    if (grid.cellInGrid(idx)) {
      for (int i = 0; i < strain.ncomp; i++)
        strain.setValueInCell(idx, i, 0);
    }
    return;
  }

  // norm strain
  real c11 = C11.valueAt(idx);
  real c12 = C12.valueAt(idx);
  // values from inverted 3x3 normal stiffness matrix
  real denom = c11*(c11 + c12) - 2.*c12*c12;
  real A = safeDiv(c11 + c12, denom);
  real B = safeDiv(-c12, denom);
#pragma unroll
  for (int i=0; i<3; i++){
    strain.setValueInCell(idx, i,
      A * stress.valueAt(idx, i) +
      B * (stress.valueAt(idx, (i+1)%3) + stress.valueAt(idx, (i+2)%3))
    );
  }

  // shear strain
  // stress times inverted shear stiffness (matrix)
  // factor 1/2 because we use real strain, not engineering strain
  real ci = safeDiv(0.5, C44.valueAt(idx));
#pragma unroll
  for (int i=3; i<6; i++) {
    strain.setValueInCell(idx, i, ci * stress.valueAt(idx, i));
  }
}


Field evalStrainTensor(const Magnet* magnet) {
  Field strain(magnet->system(), 6, 0.0);

  if (strainTensorAssuredZero(magnet)) return strain;

  int ncells = strain.grid().ncells();
  CuField stress = magnet->elasticStressTensor()->field().cu();
  CuParameter C11 = magnet->C11.cu();
  CuParameter C12 = magnet->C12.cu();
  CuParameter C44 = magnet->C44.cu();

  cudaLaunch(ncells, k_strainTensor, strain.cu(), stress, C11, C12, C44);
  return strain;
}


M_FieldQuantity strainTensorQuantity(const Magnet* magnet) {
  return M_FieldQuantity(magnet, evalStrainTensor, 6, "strain_tensor", "");
}
