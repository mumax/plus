#include <memory>

#include "antiferromagnet.hpp"
#include "constants.hpp"
#include "cudalaunch.hpp"
#include "effectivefield.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "stt.hpp"
#include "timesolver.hpp"
#include "torque.hpp"

Field evalTorque(const Ferromagnet* magnet) {
  Field torque = evalLlgTorque(magnet);
  if (!spinTransferTorqueAssuredZero(magnet))
    torque += evalSpinTransferTorque(magnet);
  return torque;
}

__global__ void k_llgtorque(CuField torque,
                            const CuField mField,
                            const CuField hField,
                            const CuParameter alpha) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!torque.cellInGeometry(idx)) {
    if (torque.cellInGrid(idx))
      torque.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  real3 m = mField.vectorAt(idx);
  real3 h = hField.vectorAt(idx);
  real a = alpha.valueAt(idx);
  real3 mxh = cross(m, h);
  real3 mxmxh = cross(m, mxh);
  real3 t = -GAMMALL / (1 + a * a) * (mxh + a * mxmxh);
  torque.setVectorInCell(idx, t);

}

Field evalLlgTorque(const Ferromagnet* magnet) {
  const Field& m = magnet->magnetization()->field();
  Field torque(magnet->system(), 3);
  Field h = evalEffectiveField(magnet);
  const Parameter& alpha = magnet->alpha;
  int ncells = torque.grid().ncells();
  cudaLaunch(ncells, k_llgtorque, torque.cu(), m.cu(), h.cu(), alpha.cu());
  return torque;
}

__global__ void k_dampingtorque(CuField torque,
                                const CuField mField,
                                const CuField hField) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  // When outside the geometry, set to zero and return early
  if (!torque.cellInGeometry(idx)) {
    if (torque.cellInGrid(idx))
      torque.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  real3 m = mField.vectorAt(idx);
  real3 h = hField.vectorAt(idx);
  real3 t = -GAMMALL * cross(m, cross(m, h));
  torque.setVectorInCell(idx, t);
}

Field evalRelaxTorque(const Ferromagnet* magnet) {
  const Field& m = magnet->magnetization()->field();
  Field torque(magnet->system(), 3);
  Field h = evalEffectiveField(magnet);
  int ncells = torque.grid().ncells();
  cudaLaunch(ncells, k_dampingtorque, torque.cu(), m.cu(), h.cu());
  return torque;
}

FM_FieldQuantity torqueQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalTorque, 3, "torque", "rad/s");
}

FM_FieldQuantity llgTorqueQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalLlgTorque, 3, "llg_torque", "rad/s");
}

FM_FieldQuantity relaxTorqueQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalRelaxTorque, 3, "damping_torque", "rad/s");
}