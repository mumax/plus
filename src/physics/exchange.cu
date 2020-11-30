#include "cudalaunch.hpp"
#include "energy.hpp"
#include "exchange.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "reduce.hpp"
#include "world.hpp"

bool exchangeAssuredZero(const Ferromagnet* magnet) {
  return magnet->aex.assuredZero() || magnet->msat.assuredZero();
}

__device__ static inline real harmonicMean(real a, real b) {
  if (a + b == 0.0)
    return 0.0;
  if (a == b)
    return a;
  return 2 * a * b / (a + b);
}

__global__ void k_exchangeField(CuField hField,
                                const CuField mField,
                                const CuParameter aex,
                                const CuParameter msat,
                                const real3 w,  // w = 1/cellsize^2
                                Grid mastergrid) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  const Grid grid = mField.system.grid;

  if (!grid.cellInGrid(idx))
    return;

  if (msat.valueAt(idx) == 0) {
    hField.setVectorInCell(idx, {0, 0, 0});
    return;
  }

  const int3 coo = grid.index2coord(idx);
  const real3 m = mField.vectorAt(idx);
  const real a = aex.valueAt(idx);

  // accumulate exchange field in h for cell at idx, divide by msat at the end
  real3 h{0, 0, 0};

  // X direction
#pragma unroll
  for (int sgn : {-1, 1}) {
    const int3 coo_ = mastergrid.wrap(coo + int3{sgn, 0, 0});
    if (!grid.cellInGrid(coo_))
      continue;
    const int idx_ = grid.coord2index(coo_);
    if (msat.valueAt(idx_) != 0) {
      const real3 m_ = mField.vectorAt(idx_);
      const real a_ = aex.valueAt(idx_);
      h += 2 * w.x * harmonicMean(a, a_) * (m_ - m);
    }
  }

  // Y direction
#pragma unroll
  for (int sgn : {-1, 1}) {
    const int3 coo_ = mastergrid.wrap(coo + int3{0, sgn, 0});
    if (!grid.cellInGrid(coo_))
      continue;
    const int idx_ = grid.coord2index(coo_);
    if (msat.valueAt(idx_) != 0) {
      const real3 m_ = mField.vectorAt(idx_);
      const real a_ = aex.valueAt(idx_);
      h += 2 * w.y * harmonicMean(a, a_) * (m_ - m);
    }
  }

  // Z direction
  if (grid.size().z > 1) {
#pragma unroll
    for (int sgn : {-1, 1}) {
      const int3 coo_ = mastergrid.wrap(coo + int3{0, 0, sgn});
      if (!grid.cellInGrid(coo_))
        continue;
      const int idx_ = grid.coord2index(coo_);
      if (msat.valueAt(idx_) != 0) {
        const real3 m_ = mField.vectorAt(idx_);
        const real a_ = aex.valueAt(idx_);
        h += 2 * w.z * harmonicMean(a, a_) * (m_ - m);
      }
    }
  }

  //  int3 neighborRelativeCoordinat__restrict__ es[6] = {int3{-1, 0, 0},
  //  int3{0, -1, 0},
  //                                         int3{0, 0, -1}, int3{1, 0, 0},
  //                                         int3{0, 1, 0},  int3{0, 0, 1}};
  //
  //  #pragma unroll
  //  for (int3 relcoo : neighborRelativeCoordinates) {
  //    const int3 coo_ = coo + relcoo;
  //    const int idx_ = hField.grid.coord2index(coo_);
  //
  //    if (hField.cellInGrid(coo_) && msat.valueAt(idx_) != 0) {
  //      real dr =
  //          cellsize.x * relcoo.x + cellsize.y * relcoo.y + cellsize.z *
  //          relcoo.z;
  //      real3 m_ = mField.vectorAt(idx_);
  //      real a_ = aex.valueAt(idx_);
  //
  //      h += 2 * harmonicMean(a, a_) * (m_ - m) / (dr * dr);
  //    }
  //  }

  h /= msat.valueAt(idx);
  hField.setVectorInCell(idx, h);
}

Field evalExchangeField(const Ferromagnet* magnet) {
  Field hField(magnet->system(), 3);
  if (exchangeAssuredZero(magnet)) {
    hField.makeZero();
    return hField;
  }

  real3 c = magnet->cellsize();
  real3 w = {1 / (c.x * c.x), 1 / (c.y * c.y), 1 / (c.z * c.z)};
  cudaLaunch(hField.grid().ncells(), k_exchangeField, hField.cu(),
             magnet->magnetization()->field().cu(), magnet->aex.cu(),
             magnet->msat.cu(), w, magnet->world()->mastergrid());
  return hField;
}

Field evalExchangeEnergyDensity(const Ferromagnet* magnet) {
  if (exchangeAssuredZero(magnet))
    return Field(magnet->system(), 1, 0.0);
  return evalEnergyDensity(magnet, evalExchangeField(magnet), 0.5);
}

real evalExchangeEnergy(const Ferromagnet* magnet) {
  if (exchangeAssuredZero(magnet))
    return 0;
  real edens = exchangeEnergyDensityQuantity(magnet).average()[0];
  int ncells = magnet->grid().ncells();
  real cellVolume = magnet->world()->cellVolume();
  return ncells * edens * cellVolume;
}

FM_FieldQuantity exchangeFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalExchangeField, 3, "exchange_field", "T");
}

FM_FieldQuantity exchangeEnergyDensityQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalExchangeEnergyDensity, 1,
                          "exchange_energy_density", "J/m3");
}

FM_ScalarQuantity exchangeEnergyQuantity(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, evalExchangeEnergy, "exchange_energy", "J");
}

__global__ void k_maxangle(CuField maxAngleField,
                           const CuField mField,
                           const CuParameter aex,
                           const CuParameter msat) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  const Grid grid = maxAngleField.system.grid;

  if (!grid.cellInGrid(idx))
    return;

  if (msat.valueAt(idx) == 0) {
    maxAngleField.setValueInCell(idx, 0, 0);
    return;
  }

  const int3 coo = grid.index2coord(idx);
  const real3 m = mField.vectorAt(idx);
  const real a = aex.valueAt(idx);

  real maxAngle{0};  // maximum angle in this cell

  int3 neighborRelativeCoordinates[6] = {int3{-1, 0, 0}, int3{0, -1, 0},
                                         int3{0, 0, -1}, int3{1, 0, 0},
                                         int3{0, 1, 0},  int3{0, 0, 1}};

#pragma unroll
  for (int3 relcoo : neighborRelativeCoordinates) {
    const int3 coo_ = coo + relcoo;
    const int idx_ = grid.coord2index(coo_);
    if (grid.cellInGrid(coo_) && msat.valueAt(idx_) != 0) {
      real3 m_ = mField.vectorAt(idx_);
      real a_ = aex.valueAt(idx_);
      real angle = m == m_ ? 0 : acos(dot(m, m_));
      if (harmonicMean(a, a_) != 0 && angle > maxAngle) {
        maxAngle = angle;
      }
    }
  }

  maxAngleField.setValueInCell(idx, 0, maxAngle);
}

real evalMaxAngle(const Ferromagnet* magnet) {
  Field maxAngleField(magnet->system(), 1);
  cudaLaunch(maxAngleField.grid().ncells(), k_maxangle, maxAngleField.cu(),
             magnet->magnetization()->field().cu(), magnet->aex.cu(),
             magnet->msat.cu());
  return maxAbsValue(maxAngleField);
}

FM_ScalarQuantity maxAngle(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, evalMaxAngle, "max_angle", "");
}
