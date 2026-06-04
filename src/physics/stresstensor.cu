#include "cudalaunch.hpp"
#include "elastodynamics.hpp"
#include "magnet.hpp"
#include "field.hpp"
#include "straintensor.hpp"
#include "stresstensor.hpp"


// --------------------------------------------------
// Elastic Stress Tensor

__global__ void k_elasticStress(CuField stressTensor,
                                const CuField strain,
                                const CuParameter C11,
                                const CuParameter C12,
                                const CuParameter C44) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = stressTensor.system;

  // When outside the geometry, set to zero and return early
  if (!system.inGeometry(idx)) {
    if (system.grid.cellInGrid(idx)) {
      for (int i = 0; i < stressTensor.ncomp; i++)
        stressTensor.setValueInCell(idx, i, 0);
    }
    return;
  }

  int ip1, ip2;
  for (int i=0; i<3; i++) {
    ip1 = i+1; ip2 = i+2;
    if (ip1 >= 3) ip1 -= 3;
    if (ip2 >= 3) ip2 -= 3;

    stressTensor.setValueInCell(idx, i,
                               C11.valueAt(idx) * strain.valueAt(idx, i) +
                               C12.valueAt(idx) * strain.valueAt(idx, ip1) +
                               C12.valueAt(idx) * strain.valueAt(idx, ip2));
    
    // factor two because we use real strain, not engineering strain
    stressTensor.setValueInCell(idx, i+3, 2 * C44.valueAt(idx) * strain.valueAt(idx, i+3));
  }
}

Field evalElasticStress(const Magnet* magnet) {
  Field stressTensor(magnet->system(), 6);
  if (elasticityAssuredZero(magnet)) {
    stressTensor.makeZero();
    return stressTensor;
  }

  int ncells = stressTensor.grid().ncells();
  Field strain = evalStrainTensor(magnet);
  CuParameter C11 = magnet->C11.cu();
  CuParameter C12 = magnet->C12.cu();
  CuParameter C44 = magnet->C44.cu();

  cudaLaunch(ncells, k_elasticStress, stressTensor.cu(), strain.cu(), C11, C12, C44);
  return stressTensor;
}

M_FieldQuantity elasticStressQuantity(const Magnet* magnet) {
  return M_FieldQuantity(magnet, evalElasticStress, 6, "elastic_stress", "N/m2");
}

// --------------------------------------------------
// Viscous Stress Tensor

bool viscousDampingAssuredZero(const Magnet* magnet) {
  return (viscousDampingTensorAssuredZero(magnet) &&
          viscousDampingRayleighAssuredZero(magnet));
}

bool viscousDampingTensorAssuredZero(const Magnet* magnet) {
  return ((!magnet->enableElastodynamics()) ||
          (magnet->eta11.assuredZero() && magnet->eta12.assuredZero() &&
           magnet->eta44.assuredZero()));
}

bool viscousDampingRayleighAssuredZero(const Magnet* magnet) {
  return ((!magnet->enableElastodynamics()) ||
          magnet->stiffnessDamping.assuredZero() ||
          (magnet->C11.assuredZero() && magnet->C12.assuredZero() &&
           magnet->C44.assuredZero()));
}

// Multiplication field operation, but with CuParameter `a` instead.
__global__ void k_multiplyParameterField(CuField y,
                                         const CuParameter a,
                                         const CuField x) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!y.cellInGeometry(idx))
    return;
  for (int c = 0; c < y.ncomp; c++) {
    y.setValueInCell(idx, c, a.valueAt(idx) * x.valueAt(idx, c));
  }
}

Field evalViscousStress(const Magnet* magnet) {
  Field stressTensor(magnet->system(), 6);
  if (viscousDampingAssuredZero(magnet)) {
    stressTensor.makeZero();
    return stressTensor;
  }

  int ncells = stressTensor.grid().ncells();
  Field strainRate = evalStrainRate(magnet);

  // use viscosity tensor if set
  if (!viscousDampingTensorAssuredZero(magnet)) {
    CuParameter eta11 = magnet->eta11.cu();
    CuParameter eta12 = magnet->eta12.cu();
    CuParameter eta44 = magnet->eta44.cu();

    // same mathematics
    cudaLaunch(ncells, k_elasticStress, stressTensor.cu(), strainRate.cu(), eta11, eta12, eta44);

  } else {  // or use stiffness tensor multiplied by rayleigh damping stiffness coefficient
    CuParameter C11 = magnet->C11.cu();
    CuParameter C12 = magnet->C12.cu();
    CuParameter C44 = magnet->C44.cu();
    CuField stressCu = stressTensor.cu();

    // same mathematics
    cudaLaunch(ncells, k_elasticStress, stressCu, strainRate.cu(), C11, C12, C44);

    // use linearity, multiply at the end
    CuParameter coeff = magnet->stiffnessDamping.cu();
    cudaLaunch(ncells, k_multiplyParameterField, stressCu, coeff, stressCu);
  }

  return stressTensor;
}

M_FieldQuantity viscousStressQuantity(const Magnet* magnet) {
  return M_FieldQuantity(magnet, evalViscousStress, 6, "viscous_stress", "N/m2");
}

// --------------------------------------------------
// Total Stress Tensor

bool stressTensorAssuredZero(const Magnet* magnet) {
  return elasticityAssuredZero(magnet) && viscousDampingAssuredZero(magnet);
}

Field evalStressTensor(const Magnet* magnet) {
  Field stressTensor = evalElasticStress(magnet);  // elastic stress or safely 0
  if (!viscousDampingAssuredZero(magnet)) stressTensor += evalViscousStress(magnet);

  return stressTensor;
}

M_FieldQuantity stressTensorQuantity(const Magnet* magnet) {
  return M_FieldQuantity(magnet, evalStressTensor, 6, "stress_tensor", "N/m2");
}
