#include "conductivitytensor.hpp"
#include "cudalaunch.hpp"
#include "quantityevaluator.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"

bool conductivityTensorAssuredZero(const Ferromagnet* magnet) {
  return magnet->conductivity.assuredZero();
}

__global__ static void k_conductTensor(CuField conductivity,
                                       const CuParameter conductivity0,
                                       const CuParameter amrRatio,
                                       const CuField mField,
                                       const CuParameter msat) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (!mField.cellInGeometry(idx) || !msat.valueAt(idx))
    return;

  const real amr = amrRatio.valueAt(idx);
  const real c0 = conductivity0.valueAt(idx);
  const real fac = 6 * amr / (6 + amr);

  const real3 m = mField.vectorAt(idx);
  conductivity.setValueInCell(idx, 0, c0 * (1 - fac * (m.x * m.x - 1. / 3.)));
  conductivity.setValueInCell(idx, 1, c0 * (1 - fac * (m.y * m.y - 1. / 3.)));
  conductivity.setValueInCell(idx, 2, c0 * (1 - fac * (m.z * m.z - 1. / 3.)));
  conductivity.setValueInCell(idx, 3, c0 * fac * m.x * m.y);
  conductivity.setValueInCell(idx, 4, c0 * fac * m.x * m.z);
  conductivity.setValueInCell(idx, 5, c0 * fac * m.y * m.z);
}

Field evalConductivityTensor(const Ferromagnet* magnet) {
  Field conductivity(magnet->system(), 6);
  int ncells = magnet->grid().ncells();
  auto conduct0 = magnet->conductivity.cu();
  auto amr = magnet->amrRatio.cu();
  auto mField = magnet->magnetization()->field().cu();
  auto msat = magnet->msat.cu();
  cudaLaunch(ncells, k_conductTensor, conductivity.cu(), conduct0, amr, mField, msat);
  return conductivity;
}

FM_FieldQuantity conductivityTensorQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalConductivityTensor, 6,
                          "conductivity_tensor", "S/m");
}
