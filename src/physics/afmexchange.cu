#include "altermagnet.hpp" // to recycle angle field calculation
#include "antiferromagnet.hpp"
#include "cudalaunch.hpp"
#include "dmi.hpp" // used for Neumann BC
#include "energy.hpp"
#include "afmexchange.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "inter_parameter.hpp"
#include "parameter.hpp"
#include "reduce.hpp"
#include "world.hpp"

bool inHomoAfmExchangeAssuredZero(const Ferromagnet* magnet) {
  if (!magnet->hostMagnet()) { return true; }
  if (magnet->hostMagnet()->afmex_nn.assuredZero() ||
      magnet->msat.assuredZero()) { return true; }

  for (auto sub : magnet->hostMagnet()->getOtherSublattices(magnet)) {
    if (!sub->msat.assuredZero())
      return false;
  }
  return true;
}

bool homoAfmExchangeAssuredZero(const Ferromagnet* magnet) {
  if (!magnet->hostMagnet()) { return true; }
  if (magnet->hostMagnet()->afmex_cell.assuredZero() ||
      magnet->hostMagnet()->latcon.assuredZero() ||
      magnet->msat.assuredZero()) {
        return true;
  }
  for (auto sub : magnet->hostMagnet()->getOtherSublattices(magnet)) {
    if (!sub->msat.assuredZero())
      return false;
  }
  return true;
}

// AFM exchange at a single site
__global__ void k_afmExchangeFieldSite(CuField hField,
                                const CuField mField,
                                const CuParameter msat,
                                const CuParameter msat2,
                                const CuParameter afmex_cell,
                                const CuParameter latcon) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (!hField.cellInGeometry(idx)) {
      if (hField.cellInGrid(idx))
        hField.setVectorInCell(idx, real3{0, 0, 0});
      return;
    }
    if (msat.valueAt(idx) == 0.) {  // total field is 0
      hField.setVectorInCell(idx, real3{0, 0, 0});
      return;
    }
    if (msat2.valueAt(idx) == 0.) {  // no addition to the field
      return;
    }

  const real l = latcon.valueAt(idx);
  real3 h0 = hField.vectorAt(idx);
  hField.setVectorInCell(idx, h0 + 4 * afmex_cell.valueAt(idx) * mField.vectorAt(idx) / (l * l * msat.valueAt(idx)));
  }

/**
 * AFM exchange between NN cells
 * 
 * It is possible to set msat, msat2 or even msat3 to 0 inside the geometry.
 * This leads to loads of possible scenarios where cells or neighbouring cells are
 * nonmagnetic or ferromagnetic instead of antiferromagnetic. This function should be
 * robust against these scenarios on a software level, but this is not necessarily the
 * correct physical treatment.*/
__global__ void k_afmExchangeFieldNN(CuField hField,
                                const CuField m1Field,
                                const CuField m2Field,
                                const CuParameter aex,
                                const CuParameter afmex_nn,
                                const CuInterParameter interExch,
                                const CuInterParameter scaleExch,
                                const CuParameter msat,
                                const CuParameter msat2,
                                const Grid mastergrid,
                                const CuDmiTensor dmiTensor,
                                bool openBC) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto system = hField.system;

  // When outside the geometry or not AFM, set to zero and return early
  // TODO: what if FM, but neighbour is AFM?
  // msat is only evaluated when inside geometry and is thus safe
  if (!hField.cellInGeometry(idx) ||
      msat.valueAt(idx) == 0 || msat2.valueAt(idx) == 0) {
    if (hField.cellInGrid(idx))
      hField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  const Grid grid = m2Field.system.grid;
  const int3 coo = grid.index2coord(idx);
  const real3 m2 = m2Field.vectorAt(idx);
  const real a = aex.valueAt(idx);
  const real ann = afmex_nn.valueAt(idx);
  const unsigned int ridx = system.getRegionIdx(idx);
  
  // If there is no FM-exchange at the boundary, open BC are assumed
  openBC = (a == 0) ? true : openBC;

  // accumulate exchange field in h for cell at idx, divide by msat at the end
  real3 h{0, 0, 0};

  // neighbor parameters
  int3 coo_;
  int idx_;
  bool outside;
  real3 m2_;
  real delta;
  real ann_;
  real exch_nn;
  real inter, scale;
  
  // AFM exchange in NN cells
#pragma unroll
  for (int3 rel_coo : {int3{-1, 0, 0}, int3{1, 0, 0}, int3{0, -1, 0},
                            int3{0, 1, 0}, int3{0, 0, -1}, int3{0, 0, 1}}) {
    coo_ = mastergrid.wrap(coo + rel_coo);

    outside = false;
    if (hField.cellInGeometry(coo_)) {
      idx_ = grid.coord2index(coo_);  // safe to set now
      real msat_ = msat.valueAt(idx_);
      real msat2_ = msat2.valueAt(idx_);
      
      if (msat_ == 0 && msat2_ == 0) {  // non-magnetic neighbour, treat as outside
        // TODO: what about 3+ sublattice magnets. msat3 may be non-zero?
        outside = true;
      } else if (msat_ == 0 || msat2_ == 0) {  // neighbour is not antiferromagnetic
        // TODO: special boundary conditions?
        continue;
      }
    } else {
      outside = true;
    }

    delta = dot(rel_coo, system.cellsize);

    if (!outside) {  // Bulk (AFM)
      m2_ = m2Field.vectorAt(idx_);
      ann_ = afmex_nn.valueAt(idx_);

      // get scaled exchange between coo and coo_
      inter = 0;
      scale = 1;
      unsigned int ridx_ = system.getRegionIdx(idx_);
      if (ridx != ridx_) {
        scale = scaleExch.valueBetween(ridx, ridx_);
        inter = interExch.valueBetween(ridx, ridx_);
      }
      exch_nn = getExchangeStiffness(inter, scale, ann, ann_);

    } else {  // Boundary
      if (openBC)
        continue;

      // Neumann BC
      int3 coo__ = mastergrid.wrap(coo - rel_coo);
      int idx__;  // not yet safe to set

      // TODO: same concerns as above
      if (hField.cellInGeometry(coo__)) {
        idx__ = grid.coord2index(coo__);  // safe to set
        if (msat.valueAt(idx__) == 0 || msat2.valueAt(idx__) == 0)
          // coo__ is outside or not antiferromagnetic
          continue;
      } else {  // outside: Neumann BC on both sides compensate to 0
        continue;
      }

      int3 normal = rel_coo * rel_coo;
      real3 Gamma2 = getGamma(dmiTensor, idx, normal, m2);

      // Approximate normal derivative of sister sublattice by taking
      // the bulk derivative closest to the edge.
      real3 m1__ = m1Field.vectorAt(idx__);
      real3 m1 = m1Field.vectorAt(idx);
      real3 d_m1 = (m1 - m1__) / delta;

      // get scaled exchange between coo and coo__
      inter = 0;
      scale = 1;
      unsigned int ridx__ = system.getRegionIdx(idx__);
      if (ridx != ridx__) {
        scale = scaleExch.valueBetween(ridx, ridx__);
        inter = interExch.valueBetween(ridx, ridx__);
      }
      real ann__ = afmex_nn.valueAt(idx__);
      real afmex_nn__ = getExchangeStiffness(inter, scale, ann, ann__);

      // fill in ghost neighboring magnetization
      m2_ = m2 + (afmex_nn__ * cross(cross(d_m1, m2), m2) + Gamma2) * delta / (2*a);
      exch_nn = ann;
    }

    h += exch_nn * (m2_ - m2) / (delta * delta);
  }

  real3 h0 = hField.vectorAt(idx);
  hField.setVectorInCell(idx, h0 + h / msat.valueAt(idx));
}

Field evalHomogeneousAfmExchangeField(const Ferromagnet* magnet) {
  Field hField(magnet->system(), 3, real3{0, 0, 0});
  if (homoAfmExchangeAssuredZero(magnet))
    return hField;

  auto host = magnet->hostMagnet();
  auto afmex_cell = host->afmex_cell.cu();
  auto latcon = host->latcon.cu();
  auto msat = magnet->msat.cu();

  for (auto sub : host->getOtherSublattices(magnet)) {
    // Accumulate seperate sublattice contributions
    auto mag2 = sub->magnetization()->field().cu();
    auto msat2 = sub->msat.cu();
    cudaLaunch(hField.grid().ncells(), k_afmExchangeFieldSite, hField.cu(),
               mag2, msat, msat2, afmex_cell, latcon);
  }
  return hField;
}

Field evalInHomogeneousAfmExchangeField(const Ferromagnet* magnet) {
  Field hField(magnet->system(), 3, real3{0, 0, 0});

  if (inHomoAfmExchangeAssuredZero(magnet))
    return hField;

  auto aex = magnet->aex.cu();
  auto dmiTensor = magnet->dmiTensor.cu();
  auto BC = magnet->enableOpenBC;
  auto mag = magnet->magnetization()->field().cu();
  auto msat = magnet->msat.cu();

  auto host = magnet->hostMagnet();
  auto afmex_nn = host->afmex_nn.cu();
  auto inter = host->interAfmExchNN.cu();
  auto scale = host->scaleAfmExchNN.cu();

  for (auto sub : host->getOtherSublattices(magnet)) {
    // Accumulate seperate sublattice contributions
    auto mag2 = sub->magnetization()->field().cu();
    auto msat2 = sub->msat.cu();
    cudaLaunch(hField.grid().ncells(), k_afmExchangeFieldNN, hField.cu(),
              mag, mag2, aex, afmex_nn, inter, scale, msat, msat2,
              magnet->world()->mastergrid(), dmiTensor, BC);
  }
  return hField;
}

Field evalInHomoAfmExchangeEnergyDensity(const Ferromagnet* magnet) {
  if (inHomoAfmExchangeAssuredZero(magnet))
    return Field(magnet->system(), 1, 0.0);
  return evalEnergyDensity(magnet, evalInHomogeneousAfmExchangeField(magnet), 0.5);
}

Field evalHomoAfmExchangeEnergyDensity(const Ferromagnet* magnet) {
  if (homoAfmExchangeAssuredZero(magnet))
    return Field(magnet->system(), 1, 0.0);
  return evalEnergyDensity(magnet, evalHomogeneousAfmExchangeField(magnet), 0.5);
}

real evalInHomoAfmExchangeEnergy(const Ferromagnet* magnet) {
  if (inHomoAfmExchangeAssuredZero(magnet))
    return 0;

  real edens = inHomoAfmExchangeEnergyDensityQuantity(magnet).average()[0];
  return energyFromEnergyDensity(magnet, edens);
}

real evalHomoAfmExchangeEnergy(const Ferromagnet* magnet) {
  if (homoAfmExchangeAssuredZero(magnet))
    return 0;

  real edens = homoAfmExchangeEnergyDensityQuantity(magnet).average()[0];
  return energyFromEnergyDensity(magnet, edens);
}

FM_FieldQuantity inHomoAfmExchangeFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalInHomogeneousAfmExchangeField, 3,
                          "inhomogeneous_exchange_field", "T");
}

FM_FieldQuantity homoAfmExchangeFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalHomogeneousAfmExchangeField, 3,
                          "homogeneous_exchange_field", "T");
}

FM_FieldQuantity inHomoAfmExchangeEnergyDensityQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalInHomoAfmExchangeEnergyDensity, 1,
                          "inhomogeneous_exchange_energy_density", "J/m3");
}

FM_FieldQuantity homoAfmExchangeEnergyDensityQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalHomoAfmExchangeEnergyDensity, 1,
                          "homogeneous_exchange_energy_density", "J/m3");
}

FM_ScalarQuantity inHomoAfmExchangeEnergyQuantity(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, evalInHomoAfmExchangeEnergy,
                          "inhomogeneous_exchange_energy", "J");
}

FM_ScalarQuantity homoAfmExchangeEnergyQuantity(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, evalHomoAfmExchangeEnergy,
                          "homogeneous_exchange_energy", "J");
}

__global__ void k_angle(CuField angleField,
                        const CuField mField1,
                        const CuField mField2,
                        const CuParameter afmex,
                        const CuParameter msat1,
                        const CuParameter msat2) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!angleField.cellInGeometry(idx)) {
    if (angleField.cellInGrid(idx)) 
      angleField.setValueInCell(idx, 0, 0);
    return;
  }

  if (msat1.valueAt(idx) == 0 || msat2.valueAt(idx) == 0 || afmex.valueAt(idx) == 0) {
    angleField.setValueInCell(idx, 0, 0);
    return;
  }

  angleField.setValueInCell(idx, 0, acos(copysign(1.0, afmex.valueAt(idx))
                                            * dot(mField1.vectorAt(idx),
                                                  mField2.vectorAt(idx))));
}

Field evalAngleField(const HostMagnet* magnet) {
  if (magnet->sublattices().size() != 2)
    throw std::runtime_error("Cannot compute the angle field if the magnet has no two sublattices");

  Field angleField(magnet->system(), 1);

  cudaLaunch(angleField.grid().ncells(), k_angle, angleField.cu(),
            magnet->sublattices()[0]->magnetization()->field().cu(),
            magnet->sublattices()[1]->magnetization()->field().cu(),
            magnet->afmex_cell.cu(),
            magnet->sublattices()[0]->msat.cu(), magnet->sublattices()[1]->msat.cu());
  return angleField;
}

real evalMaxAngle(const HostMagnet* magnet) {
  return maxAbsValue(evalAngleField(magnet));
}

AFM_FieldQuantity angleFieldQuantity(const Antiferromagnet* magnet) {
  return AFM_FieldQuantity(magnet, evalAngleField, 1, "angle_field", "rad");
}

ATM_FieldQuantity angleFieldQuantity(const Altermagnet* magnet) {
  return ATM_FieldQuantity(magnet, evalAngleField, 1, "angle_field", "rad");
}

AFM_ScalarQuantity maxAngle(const Antiferromagnet* magnet) {
  return AFM_ScalarQuantity(magnet, evalMaxAngle, "max_angle", "rad");
}

ATM_ScalarQuantity maxAngle(const Altermagnet* magnet) {
  return ATM_ScalarQuantity(magnet, evalMaxAngle, "max_angle", "rad");
}