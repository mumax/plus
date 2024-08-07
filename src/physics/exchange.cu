#include "antiferromagnet.hpp"
#include "cudalaunch.hpp"
#include "dmi.hpp" // used for Neumann BC
#include "dmitensor.hpp"
#include "energy.hpp"
#include "exchange.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "reduce.hpp"
#include "world.hpp"

bool exchangeAssuredZero(const Ferromagnet* magnet) {
  return (magnet->aex.assuredZero() || magnet->msat.assuredZero());
}

__global__ void k_exchangeField(CuField hField,
                                const CuField mField,
                                const CuParameter aex,
                                const CuParameter msat,
                                const real3 w,  // w = 1/cellsize^2
                                Grid mastergrid,
                                bool openBC,
                                const CuDmiTensor dmiTensor) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto system = hField.system;

  // When outside the geometry, set to zero and return early
  if (!hField.cellInGeometry(idx)) {
    if (hField.cellInGrid(idx)) {
      hField.setVectorInCell(idx, real3{0, 0, 0});
    }
    return;
  }

  const Grid grid = mField.system.grid;

  if (!grid.cellInGrid(idx))
    return;

  if (msat.valueAt(idx) == 0) {
    hField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  const int3 coo = grid.index2coord(idx);
  const real3 m = mField.vectorAt(idx);
  const real a = aex.valueAt(idx);
  
  // accumulate exchange field in h for cell at idx, divide by msat at the end
  real3 h{0, 0, 0};

  // FM exchange in NN cells
  // X direction
#pragma unroll
  for (int sgn : {-1, 1}) {
    const int3 coo_ = mastergrid.wrap(coo + int3{sgn, 0, 0});
    if (!hField.cellInGeometry(coo_) && openBC)
      continue;
    
    const int idx_ = grid.coord2index(coo_);
    if (msat.valueAt(idx_) != 0 || !openBC) {
      real3 m_;
      real a_;

      if (hField.cellInGeometry(coo_)) {
        m_ = mField.vectorAt(idx_);
        a_ = aex.valueAt(idx_);
      }
      else { // Neumann BC
        real3 Gamma = getGamma(dmiTensor, idx, int3{sgn * sgn, 0, 0}, m);
        m_ = m + (Gamma / (2*a)) * sgn * system.cellsize.x;
        a_ = a;
      }
      h += 2 * harmonicMean(a, a_) * w.x * (m_ - m);      
    }
  }

  // Y direction
#pragma unroll
  for (int sgn : {-1, 1}) {
    const int3 coo_ = mastergrid.wrap(coo + int3{0, sgn, 0});
    if (!hField.cellInGeometry(coo_) && openBC)
      continue;

    const int idx_ = grid.coord2index(coo_);

    if (msat.valueAt(idx_) != 0 || !openBC) {
      real3 m_;
      real a_;

      if (hField.cellInGeometry(coo_)) {
        m_ = mField.vectorAt(idx_);
        a_ = aex.valueAt(idx_);
      }
      else { // Neumann BC
        real3 Gamma = getGamma(dmiTensor, idx, int3{0, sgn * sgn, 0}, m);
        m_ = m + (Gamma / (2*a)) * sgn * system.cellsize.y;
        a_ = a;
      }
      h += 2 * harmonicMean(a, a_) * w.y * (m_ - m);
    }
  }

  // Z direction
  if (grid.size().z > 1) {
#pragma unroll
    for (int sgn : {-1, 1}) {
      const int3 coo_ = mastergrid.wrap(coo + int3{0, 0, sgn});
      if (!hField.cellInGeometry(coo_) && openBC)
        continue;

      const int idx_ = grid.coord2index(coo_);
      
      if (msat.valueAt(idx_) != 0 || !openBC) {      
        real3 m_;
        real a_;

        if (hField.cellInGeometry(coo_)) {
          m_ = mField.vectorAt(idx_);
          a_ = aex.valueAt(idx_);
        }
        else { // Neumann BC
        real3 Gamma = getGamma(dmiTensor, idx, int3{0, 0, sgn * sgn}, m);
        m_ = m + (Gamma / (2*a)) * sgn * system.cellsize.z;
        a_ = a;
        }
        h += 2 * harmonicMean(a, a_) * w.z * (m_ - m);        
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

  hField.setVectorInCell(idx, h / msat.valueAt(idx));
}

__global__ void k_exchangeField(CuField hField,
                                const CuField m1Field,
                                const CuField m2Field,
                                const CuParameter aex,
                                const CuParameter afmex_nn,
                                const CuParameter msat,
                                const real3 w,  // w = 1/cellsize^2
                                Grid mastergrid,
                                const CuDmiTensor dmiTensor) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto system = hField.system;

  // When outside the geometry, set to zero and return early
  if (!hField.cellInGeometry(idx)) {
    if (hField.cellInGrid(idx)) {
      hField.setVectorInCell(idx, real3{0, 0, 0});
    }
    return;
  }

  const Grid grid = m1Field.system.grid;

  if (!grid.cellInGrid(idx))
    return;

  if (msat.valueAt(idx) == 0) {
    hField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  const int3 coo = grid.index2coord(idx);
  const real3 m = m1Field.vectorAt(idx);
  const real a = aex.valueAt(idx);
  const real an = afmex_nn.valueAt(idx);
  
  // accumulate exchange field in h for cell at idx, divide by msat at the end
  real3 h{0, 0, 0};

  // FM exchange in NN cells
  // X direction
#pragma unroll
  for (int sgn : {-1, 1}) {
    const int3 coo_ = mastergrid.wrap(coo + int3{sgn, 0, 0});   
    const int idx_ = grid.coord2index(coo_);

    if (msat.valueAt(idx_) != 0) {
      real3 m_;
      real a_;

      if (hField.cellInGeometry(coo_)) {
        m_ = m1Field.vectorAt(idx_);
        a_ = aex.valueAt(idx_);
      }
      else { // Neumann BC
        real3 Gamma1 = getGamma(dmiTensor, idx, int3{sgn * sgn, 0, 0}, m);
        real3 Gamma2 = getGamma(dmiTensor, idx, int3{sgn * sgn, 0, 0}, m2Field.vectorAt(idx));
        real a2 = 2 * a;
        real an_a2 = an / a2;
        m_ = m + sgn * system.cellsize.x / (a2 * (1 - an_a2*an_a2)) * (Gamma1 - an_a2 * Gamma2);
        a_ = a;
      }
      h += 2 * harmonicMean(a, a_) * w.x * (m_ - m);      
    }
  }

  // Y direction
#pragma unroll
  for (int sgn : {-1, 1}) {
    const int3 coo_ = mastergrid.wrap(coo + int3{0, sgn, 0});
    const int idx_ = grid.coord2index(coo_);

    if (msat.valueAt(idx_) != 0) {
      real3 m_;
      real a_;

      if (hField.cellInGeometry(coo_)) {
        m_ = m1Field.vectorAt(idx_);
        a_ = aex.valueAt(idx_);
      }
      else { // Neumann BC
        real3 Gamma1 = getGamma(dmiTensor, idx, int3{0, sgn * sgn, 0}, m);
        real3 Gamma2 = getGamma(dmiTensor, idx, int3{0, sgn * sgn, 0}, m2Field.vectorAt(idx));
        real a2 = 2 * a;
        real an_a2 = an / a2;
        m_ = m + sgn * system.cellsize.y / (a2 * (1 - an_a2*an_a2)) * (Gamma1 - an_a2 * Gamma2);
        a_ = a;
      }
      h += 2 * harmonicMean(a, a_) * w.y * (m_ - m);
    }
  }

  // Z direction
  if (grid.size().z > 1) {
#pragma unroll
    for (int sgn : {-1, 1}) {
      const int3 coo_ = mastergrid.wrap(coo + int3{0, 0, sgn});
      const int idx_ = grid.coord2index(coo_);
      
      if (msat.valueAt(idx_) != 0) {      
        real3 m_;
        real a_;

        if (hField.cellInGeometry(coo_)) {
          m_ = m1Field.vectorAt(idx_);
          a_ = aex.valueAt(idx_);
        }
        else { // Neumann BC
        real3 Gamma1 = getGamma(dmiTensor, idx, int3{0, 0, sgn * sgn}, m);
        real3 Gamma2 = getGamma(dmiTensor, idx, int3{0, 0, sgn * sgn}, m2Field.vectorAt(idx));
        real a2 = 2 * a;
        real an_a2 = an / a2;
        m_ = m + sgn * system.cellsize.z / (a2 * (1 - an_a2*an_a2)) * (Gamma1 - an_a2 * Gamma2);
        a_ = a;
        }
        h += 2 * harmonicMean(a, a_) * w.z * (m_ - m);        
      }
    }
  }
  hField.setVectorInCell(idx, h / msat.valueAt(idx));
}

Field evalExchangeField(const Ferromagnet* magnet) {

  Field hField(magnet->system(), 3);
  
  if (exchangeAssuredZero(magnet)) {
    hField.makeZero();
    return hField;
  }

  real3 c = magnet->cellsize();
  real3 w = {1 / (c.x * c.x), 1 / (c.y * c.y), 1 / (c.z * c.z)};
  
  int ncells = hField.grid().ncells();
  auto mag = magnet->magnetization()->field().cu();
  auto msat = magnet->msat.cu();
  auto aex = magnet->aex.cu();
  auto grid = magnet->world()->mastergrid();
  auto dmiTensor = magnet->dmiTensor.cu();

  if (!magnet->isSublattice() || magnet->enableOpenBC)
    cudaLaunch(ncells, k_exchangeField, hField.cu(), mag,
              aex, msat, w, grid, magnet->enableOpenBC, dmiTensor);
  else {
    // In case `magnet` is a sublattice, it's sister sublattice affects
    // the Neumann BC. There are no open boundaries when in this scope.
    auto mag2 = magnet->hostMagnet()->getOtherSublattice(magnet)->magnetization()->field().cu();
    auto afmex_nn = magnet->hostMagnet()->afmex_nn.cu();
    cudaLaunch(ncells, k_exchangeField, hField.cu(), mag,
              mag2, aex, afmex_nn, msat, w, grid, dmiTensor);
  }
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

  // When outside the geometry, set to zero and return early
  if (!maxAngleField.cellInGeometry(idx)) {
    if (maxAngleField.cellInGrid(idx)) 
      maxAngleField.setValueInCell(idx, 0, 0);
    return;
  }

  const Grid grid = maxAngleField.system.grid;

  if (msat.valueAt(idx) == 0) {
    maxAngleField.setValueInCell(idx, 0, 0);
    return;
  }

  const int3 coo = grid.index2coord(idx);
  const real a = aex.valueAt(idx);

  real maxAngle{0};  // maximum angle in this cell

  int3 neighborRelativeCoordinates[6] = {int3{-1, 0, 0}, int3{0, -1, 0},
                                         int3{0, 0, -1}, int3{1, 0, 0},
                                         int3{0, 1, 0},  int3{0, 0, 1}};

#pragma unroll
  for (int3 relcoo : neighborRelativeCoordinates) {
    const int3 coo_ = coo + relcoo;
    const int idx_ = grid.coord2index(coo_);
    if (mField.cellInGeometry(coo_) && msat.valueAt(idx_) != 0) {
      real a_ = aex.valueAt(idx_);
      real3 m = mField.vectorAt(idx);
      real3 m_ = mField.vectorAt(idx_);
      real angle = m == m_ ? 0 : acos(dot(m, m_));
      if (harmonicMean(a, a_) != 0 && angle > maxAngle)
        maxAngle = angle;
    }
  }
  maxAngleField.setValueInCell(idx, 0, maxAngle);
}

real evalMaxAngle(const Ferromagnet* magnet) {
  Field maxAngleField(magnet->system(), 1);
  cudaLaunch(maxAngleField.grid().ncells(), k_maxangle, maxAngleField.cu(),
             magnet->magnetization()->field().cu(), magnet->aex.cu(), magnet->msat.cu());
  return maxAbsValue(maxAngleField);
}

FM_ScalarQuantity maxAngle(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, evalMaxAngle, "max_angle", "");
}
