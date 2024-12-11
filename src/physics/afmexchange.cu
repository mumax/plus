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
  if (!magnet->isSublattice()) { return true; }

  return ( magnet->hostMagnet()->afmex_nn.assuredZero() ||
           magnet->hostMagnet()->getOtherSublattice(magnet)->msat.assuredZero());
}

bool homoAfmExchangeAssuredZero(const Ferromagnet* magnet) {
  if (!magnet->isSublattice()) { return true; }

  return (magnet->hostMagnet()->afmex_cell.assuredZero() ||
          magnet->hostMagnet()->getOtherSublattice(magnet)->msat.assuredZero());
}

// AFM exchange at a single site
__global__ void k_afmExchangeFieldSite(CuField hField,
                                const CuField mField,
                                const CuParameter msat,
                                const CuParameter afmex_cell,
                                const CuParameter latcon) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (!hField.cellInGeometry(idx)) {
      if (hField.cellInGrid(idx))
        hField.setVectorInCell(idx, real3{0, 0, 0});
      return;
    }
    if (msat.valueAt(idx) == 0.) {
      hField.setVectorInCell(idx, real3{0, 0, 0});
      return;
    }

  const real l = latcon.valueAt(idx);
  real3 h = 4 * afmex_cell.valueAt(idx) * mField.vectorAt(idx) / (l * l);
  hField.setVectorInCell(idx, h / msat.valueAt(idx));
  }

// AFM exchange between NN cells
__global__ void k_afmExchangeFieldNN(CuField hField,
                                const CuField m1Field,
                                const CuField m2Field,
                                const CuParameter aex,
                                const CuParameter afmex_nn,
                                const CuInterParameter interExch,
                                const CuInterParameter scaleExch,
                                const CuParameter msat2,
                                const Grid mastergrid,
                                const CuDmiTensor dmiTensor,
                                bool openBC) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto system = hField.system;

  // When outside the geometry, set to zero and return early
  if (!hField.cellInGeometry(idx)) {
    if (hField.cellInGrid(idx))
      hField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  const Grid grid = m2Field.system.grid;

  if (!grid.cellInGrid(idx))
    return;

  if (msat2.valueAt(idx) == 0) {
    hField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  const int3 coo = grid.index2coord(idx);
  const real3 m2 = m2Field.vectorAt(idx);
  const real a = aex.valueAt(idx);
  const real ann = afmex_nn.valueAt(idx);
  
  // If there is no FM-exchange at the boundary, open BC are assumed
  openBC = (a == 0) ? true : openBC;

  // accumulate exchange field in h for cell at idx, divide by msat at the end
  real3 h{0, 0, 0};
  
  // AFM exchange in NN cells
#pragma unroll
  for (int3 rel_coo : {int3{-1, 0, 0}, int3{1, 0, 0}, int3{0, -1, 0},
                            int3{0, 1, 0}, int3{0, 0, -1}, int3{0, 0, 1}}) {
    int3 coo_ = mastergrid.wrap(coo + rel_coo);

    if(!hField.cellInGeometry(coo_) && openBC)
      continue;
    
    const int idx_ = grid.coord2index(coo_);
    real delta = dot(rel_coo, system.cellsize);

    if(msat2.valueAt(idx_) != 0 || !openBC) {
      real3 m2_;
      real ann_;
      int3 normal = rel_coo * rel_coo;

      real inter = 0;
      real scale = 1;
      real Aex;

      if(hField.cellInGeometry(coo_)) {
        m2_ = m2Field.vectorAt(idx_);
        ann_ = afmex_nn.valueAt(idx_);

        unsigned int ridx = system.getRegionIdx(idx);
        unsigned int ridx_ = system.getRegionIdx(idx_);

        if (ridx != ridx_) {
          scale = scaleExch.valueBetween(ridx, ridx_);
          inter = interExch.valueBetween(ridx, ridx_);
        }
      }
      else { // Neumann BC
        real3 Gamma2 = getGamma(dmiTensor, idx, normal, m2);

        real3 d_m1{0, 0, 0};
        int3 other_neighbor_coo = mastergrid.wrap(coo - rel_coo);
        if(hField.cellInGeometry(other_neighbor_coo)){
          // Approximate normal derivative of sister sublattice by taking
          // the bulk derivative closest to the edge.
          real3 m1__ = m1Field.vectorAt(other_neighbor_coo);
          real3 m1 = m1Field.vectorAt(idx);
          d_m1 = (m1 - m1__) / delta;
        }

        m2_ = m2 + (ann * cross(cross(d_m1, m2), m2) + Gamma2) * delta / (2*a);
        ann_ = ann;
      }
      Aex = (inter != 0) ? inter : harmonicMean(ann, ann_);
      Aex *= scale;

      h += Aex * (m2_ - m2) / (delta * delta);
    }
  }
  hField.setVectorInCell(idx, h / msat2.valueAt(idx));
}

Field evalHomogeneousAfmExchangeField(const Ferromagnet* magnet) {
  Field hField(magnet->system(), 3);
  
  if (homoAfmExchangeAssuredZero(magnet)) {
    hField.makeZero();
    return hField;
  }

  auto otherSub = magnet->hostMagnet()->getOtherSublattice(magnet);
  auto otherMag = otherSub->magnetization()->field().cu();
  auto msat2 = otherSub->msat.cu();
  auto afmex_cell = magnet->hostMagnet()->afmex_cell.cu();
  auto latcon = magnet->hostMagnet()->latcon.cu();

  cudaLaunch(hField.grid().ncells(), k_afmExchangeFieldSite, hField.cu(),
            otherMag, msat2, afmex_cell, latcon);
  return hField;
}

Field evalInHomogeneousAfmExchangeField(const Ferromagnet* magnet) {

  Field hField(magnet->system(), 3);
  
  if (inHomoAfmExchangeAssuredZero(magnet)) {
    hField.makeZero();
    return hField;
  }
  
  auto otherSub = magnet->hostMagnet()->getOtherSublattice(magnet);
  auto mag = magnet->magnetization()->field().cu();
  auto otherMag = otherSub->magnetization()->field().cu();
  auto msat2 = otherSub->msat.cu();
  auto aex = magnet->aex.cu();
  auto afmex_nn = magnet->hostMagnet()->afmex_nn.cu();
  auto BC = magnet->enableOpenBC;
  auto dmiTensor = magnet->dmiTensor.cu();
  auto inter = magnet->hostMagnet()->interAfmExchNN.cu();
  auto scale = magnet->hostMagnet()->scaleAfmExchNN.cu();

  cudaLaunch(hField.grid().ncells(), k_afmExchangeFieldNN, hField.cu(),
            mag, otherMag, aex, afmex_nn, inter, scale, msat2,
            magnet->world()->mastergrid(), dmiTensor, BC);
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

  int ncells = magnet->grid().ncells();
  real cellVolume = magnet->world()->cellVolume();
  return ncells * edens * cellVolume;
}

real evalHomoAfmExchangeEnergy(const Ferromagnet* magnet) {
  if (homoAfmExchangeAssuredZero(magnet))
    return 0;
    
  real edens = homoAfmExchangeEnergyDensityQuantity(magnet).average()[0];

  int ncells = magnet->grid().ncells();
  real cellVolume = magnet->world()->cellVolume();
  return ncells * edens * cellVolume;
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

Field evalAngleField(const Antiferromagnet* magnet) {
  Field angleField(magnet->system(), 1);

  cudaLaunch(angleField.grid().ncells(), k_angle, angleField.cu(),
            magnet->sub1()->magnetization()->field().cu(),
            magnet->sub2()->magnetization()->field().cu(),
            magnet->afmex_cell.cu(),
            magnet->sub1()->msat.cu(), magnet->sub2()->msat.cu());
  return angleField;
}

real evalMaxAngle(const Antiferromagnet* magnet) {
  return maxAbsValue(evalAngleField(magnet));
}

AFM_FieldQuantity angleFieldQuantity(const Antiferromagnet* magnet) {
  return AFM_FieldQuantity(magnet, evalAngleField, 1, "angle_field", "rad");
}

AFM_ScalarQuantity maxAngle(const Antiferromagnet* magnet) {
  return AFM_ScalarQuantity(magnet, evalMaxAngle, "max_angle", "rad");
}