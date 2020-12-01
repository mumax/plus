#include "constants.hpp"
#include "cudalaunch.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "stt.hpp"
#include "world.hpp"

bool spinTransferTorqueAssuredZero(const Ferromagnet* magnet) {
  return magnet->msat.assuredZero() || magnet->jcur.assuredZero() ||
         magnet->pol.assuredZero();
}

__global__ void k_spinTransferTorque(CuField torque,
                                     const CuField mField,
                                     const CuParameter msatParam,
                                     const CuParameter polParam,
                                     const CuParameter xiParam,
                                     const CuParameter alphaParam,
                                     const CuVectorParameter jcurParam) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  const Grid grid = torque.system.grid;
  const real3 cellsize = torque.system.cellsize;

  // When outside the geometry, set to zero and return early
  if (!torque.cellInGeometry(idx)) {
    if (torque.cellInGrid(idx))
      torque.setVectorInCell(idx, {0, 0, 0});
    return;
  }

  const real3 m = mField.vectorAt(idx);
  const real3 j = jcurParam.vectorAt(idx);
  const real msat = msatParam.valueAt(idx);
  const real pol = polParam.valueAt(idx);
  const real xi = xiParam.valueAt(idx);
  const real alpha = alphaParam.valueAt(idx);

  if (msat == 0 || pol == 0 || j == real3{0, 0, 0}) {
    torque.setVectorInCell(idx, {0, 0, 0});
    return;
  }

  const int3 coo = grid.index2coord(idx);

  const real3 u = MUB / (QE * msat * (1 + xi * xi)) * j;

  real3 hspin{0, 0, 0};

  // x derivative
  for (int sign : {-1, 1}) {  // left and right neighbor
    const int3 coo_ = coo + int3{sign, 0, 0};
    if (grid.cellInGrid(coo_) && msatParam.valueAt(coo_) != 0) {
      const real3 m_ = mField.vectorAt(coo_);
      hspin += sign * u.x * m_ / (2 * cellsize.x);  // central finite difference
    }
  }
  // y derivative
  for (int sign : {-1, 1}) {
    const int3 coo_ = coo + int3{0, sign, 0};
    if (grid.cellInGrid(coo_) && msatParam.valueAt(coo_) != 0) {
      const real3 m_ = mField.vectorAt(coo_);
      hspin += sign * u.y * m_ / (2 * cellsize.y);
    }
  }
  // z derivative
  for (int sign : {-1, 1}) {
    const int3 coo_ = coo + int3{0, 0, sign};
    if (grid.cellInGrid(coo_) && msatParam.valueAt(coo_) != 0) {
      const real3 m_ = mField.vectorAt(coo_);
      hspin += sign * u.z * m_ / (2 * cellsize.z);  // central finite difference
    }
  }

  const real3 mxh = cross(m, hspin);
  const real3 mxmxh = cross(m, mxh);
  const real3 t = (-1 / (1 + alpha * alpha)) *
                  ((1 + xi * alpha) * mxmxh + (xi - alpha) * mxh);

  torque.setVectorInCell(idx, t);
}

Field evalSpinTransferTorque(const Ferromagnet* magnet) {
  Field torque(magnet->system(), 3);
  if (spinTransferTorqueAssuredZero(magnet)) {
    torque.makeZero();
    return torque;
  }
  int ncells = magnet->grid().ncells();
  auto m = magnet->magnetization()->field().cu();
  auto msat = magnet->msat.cu();
  auto pol = magnet->pol.cu();
  auto xi = magnet->xi.cu();
  auto alpha = magnet->alpha.cu();
  auto jcur = magnet->jcur.cu();
  auto cellsize = magnet->world()->cellsize();
  cudaLaunch(ncells, k_spinTransferTorque, torque.cu(), m, msat, pol, xi, alpha,
             jcur);
  return torque;
}

FM_FieldQuantity spinTransferTorqueQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalSpinTransferTorque, 3,
                          "spintransfer_torque", "1/s");
}
