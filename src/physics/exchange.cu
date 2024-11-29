#include "antiferromagnet.hpp"
#include "cudalaunch.hpp"
#include "dmi.hpp" // used for Neumann BC
#include "dmitensor.hpp"
#include "energy.hpp"
#include "exchange.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "inter_parameter.hpp"
#include "parameter.hpp"
#include "reduce.hpp"
#include "world.hpp"

bool exchangeAssuredZero(const Ferromagnet* magnet) {
  return ((magnet->aex.assuredZero() && magnet->interExch.assuredZero())
        || magnet->scaleExch.assuredZero()
        || magnet->msat.assuredZero());
}

__global__ void k_exchangeField(CuField hField,
                                const CuField mField,
                                const CuParameter aex,
                                const CuParameter msat,
                                const real3 w,  // w = 1/cellsize^2
                                const Grid mastergrid,
                                bool openBC,
                                const CuDmiTensor dmiTensor,
                                const CuInterParameter interEx,
                                const CuInterParameter scaleEx) {
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
#pragma unroll
  for (int3 rel_coo : {int3{-1, 0, 0}, int3{1, 0, 0}, int3{0, -1, 0},
                            int3{0, 1, 0}, int3{0, 0, -1}, int3{0, 0, 1}}) {
    const int3 coo_ = mastergrid.wrap(coo + rel_coo);
    if(!hField.cellInGeometry(coo_) && openBC)
      continue;

    const int idx_ = grid.coord2index(coo_);

    if(msat.valueAt(idx_) != 0 || !openBC) {
      real3 m_;
      real a_;
      int3 normal = rel_coo * rel_coo;

      real inter = 0;
      real scale = 1;
      real Aex;

      if(hField.cellInGeometry(coo_)) {
        m_ = mField.vectorAt(idx_);
        a_ = aex.valueAt(idx_);

        uint ridx = system.getRegionIdx(idx);
        uint ridx_ = system.getRegionIdx(idx_);

        if (ridx != ridx_) {
          scale = scaleEx.valueBetween(ridx, ridx_);
          inter = interEx.valueBetween(ridx, ridx_);
        }
      }
      else { // Neumann BC
        if (a == 0)
          continue;

        real3 Gamma = getGamma(dmiTensor, idx, normal, m);
        real delta = dot(rel_coo, system.cellsize);
        m_ = m + (Gamma / (2*a)) * delta;
        a_ = a;
      }

      Aex = (inter != 0) ? inter : harmonicMean(a, a_);
      Aex *= scale;
      h += 2 * Aex * dot(normal, w) * (m_ - m);
    }
  }
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
                                const CuDmiTensor dmiTensor,
                                const CuInterParameter interEx,
                                const CuInterParameter scaleEx) {
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
#pragma unroll
  for (int3 rel_coo : {int3{-1, 0, 0}, int3{1, 0, 0}, int3{0, -1, 0},
                            int3{0, 1, 0}, int3{0, 0, -1}, int3{0, 0, 1}}) {
    const int3 coo_ = mastergrid.wrap(coo + rel_coo);
    const int idx_ = grid.coord2index(coo_);

    real3 m_;
    real a_;
    int3 normal = rel_coo * rel_coo;

    real inter = 0;
    real scale = 1;
    real Aex;

    if(hField.cellInGeometry(coo_)) {
      if (msat.valueAt(idx_) == 0)
        continue;

      m_ = m1Field.vectorAt(idx_);
      a_ = aex.valueAt(idx_);

      uint ridx = system.getRegionIdx(idx);
      uint ridx_ = system.getRegionIdx(idx_);

      if (ridx != ridx_) {
        scale = scaleEx.valueBetween(ridx, ridx_);
        inter = interEx.valueBetween(ridx, ridx_);
      }
    }
    else { // Neumann BC
      if (a == 0)
        continue;

      real3 m2 = m2Field.vectorAt(idx);
      real3 Gamma1 = getGamma(dmiTensor, idx, normal, m);

      real delta = dot(rel_coo, system.cellsize);

      real3 d_m2{0, 0, 0};
      int3 other_neighbor_coo = mastergrid.wrap(coo - rel_coo);
      if(hField.cellInGeometry(other_neighbor_coo)) {
        // Approximate normal derivative of sister sublattice by taking
        // the bulk derivative closest to the edge.
        real3 m2__ = m2Field.vectorAt(other_neighbor_coo);
        d_m2 = (m2 - m2__) / delta;
      }

      m_ = m + (an * cross(cross(d_m2, m), m) + Gamma1) * delta / (2*a);
      a_ = a;
    }

    Aex = (inter != 0) ? inter : harmonicMean(a, a_);
    Aex *= scale;
    h += 2 * Aex * dot(normal, w) * (m_ - m);
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
  auto interEx = magnet->interExch.cu();
  auto scaleEx = magnet->scaleExch.cu();

  if (!magnet->isSublattice() || magnet->enableOpenBC)
    cudaLaunch(ncells, k_exchangeField, hField.cu(), mag,
              aex, msat, w, grid, magnet->enableOpenBC, dmiTensor,
              interEx, scaleEx);
  else {
    // In case `magnet` is a sublattice, it's sister sublattice affects
    // the Neumann BC. There are no open boundaries when in this scope.
    auto mag2 = magnet->hostMagnet()->getOtherSublattice(magnet)->magnetization()->field().cu();
    auto afmex_nn = magnet->hostMagnet()->afmex_nn.cu();
    cudaLaunch(ncells, k_exchangeField, hField.cu(), mag,
              mag2, aex, afmex_nn, msat, w, grid, dmiTensor,
              interEx, scaleEx);
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
                           const CuParameter msat,
                           const Grid mastergrid) {
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
    const int3 coo_ = mastergrid.wrap(coo + relcoo);
    if (!mField.cellInGeometry(coo_))
      continue;
    const int idx_ = grid.coord2index(coo_);
    if (msat.valueAt(idx_) != 0) {
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
             magnet->magnetization()->field().cu(), magnet->aex.cu(),
             magnet->msat.cu(), magnet->world()->mastergrid());
  return maxAbsValue(maxAngleField);
}

FM_ScalarQuantity maxAngle(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, evalMaxAngle, "max_angle", "rad");
}
