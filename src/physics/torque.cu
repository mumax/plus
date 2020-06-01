#include <memory>

#include "constants.hpp"
#include "cudalaunch.hpp"
#include "effectivefield.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "torque.hpp"

__global__ void k_torque(CuField torque,
                         CuField mField,
                         CuField hField,
                         CuParameter alpha) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!torque.cellInGrid(idx))
    return;
  real3 m = mField.vectorAt(idx);
  real3 h = hField.vectorAt(idx);
  real a = alpha.valueAt(idx);
  real3 mxh = cross(m, h);
  real3 mxmxh = cross(m, mxh);
  real3 t = -GAMMALL / (1 + a * a) * (mxh + a * mxmxh);
  torque.setVectorInCell(idx, t);
}

Field evalTorque(const Ferromagnet* magnet) {
  Field torque(magnet->grid(),3);
  Field h = evalEffectiveField(magnet);
  const Field& m = magnet->magnetization()->field();
  const Parameter& alpha = magnet->alpha;
  int ncells = torque.grid().ncells();
  cudaLaunch(ncells, k_torque, torque.cu(), m.cu(), h.cu(), alpha.cu());
  return torque;
}

__global__ void k_dampingtorque(CuField torque,
                                CuField mField,
                                CuField hField) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!torque.cellInGrid(idx))
    return;
  real3 m = mField.vectorAt(idx);
  real3 h = hField.vectorAt(idx);
  real3 t = -GAMMALL * cross(m, cross(m, h));
  torque.setVectorInCell(idx, t);
}

Field evalRelaxTorque(const Ferromagnet* magnet){
  Field torque(magnet->grid(),3);
  Field h = evalEffectiveField(magnet);
  const Field& m = magnet->magnetization()->field();
  int ncells = torque.grid().ncells();
  cudaLaunch(ncells, k_dampingtorque, torque.cu(), m.cu(), h.cu());
  return torque;
}

FM_FieldQuantity torqueQuantity(const Ferromagnet * magnet) {
  return FM_FieldQuantity(magnet, evalTorque, 3, "torque", "T");
}

FM_FieldQuantity relaxTorqueQuantity(const Ferromagnet * magnet) {
  return FM_FieldQuantity(magnet, evalRelaxTorque, 3, "damping_torque", "T");
}