#include "antiferromagnet.hpp"
#include "cudalaunch.hpp"
#include "datatypes.hpp"
#include "dmi.hpp"
#include "dmitensor.hpp"
#include "energy.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "world.hpp"

bool dmiAssuredZero(const Ferromagnet* magnet) {
  if (magnet->msat.assuredZero()) { return true; }
  if (magnet->isSublattice())
    return magnet->dmiTensor.assuredZero() && magnet->hostMagnet()->dmiTensor.assuredZero();
  return magnet->dmiTensor.assuredZero();
}

__global__ void k_dmiFieldFM(CuField hField,
                           const CuField mField,
                           const CuDmiTensor dmiTensor,
                           const CuParameter msat,
                           const Grid mastergrid,
                           const CuParameter aex,
                           bool openBC) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto system = hField.system;
 
  if (!system.grid.cellInGrid(idx))
    return;

  // When outside the geometry or msat=0, set to zero and return early
  if (!system.inGeometry(idx) || (msat.valueAt(idx) == 0)) {
    hField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

	real3 m = mField.vectorAt(idx);
  const int3 coo = system.grid.index2coord(idx);

  // Accumulate DMI field of cell at idx in h. Divide by msat at the end.
  real3 h{0, 0, 0};

// Loop over the 6 nearest neighbors using the neighbor's relative coordinate.
// Compute for each neighbor the DMI effective field term.
#pragma unroll
  for (int3 relative_coo : {int3{-1, 0, 0}, int3{1, 0, 0}, int3{0, -1, 0},
                            int3{0, 1, 0}, int3{0, 0, -1}, int3{0, 0, 1}}) {
    int3 neighbor_coo = mastergrid.wrap(coo + relative_coo);
    int neighbor_idx;
    if (!system.inGeometry(neighbor_coo)) { neighbor_idx = idx; }
    else { neighbor_idx = system.grid.coord2index(neighbor_coo); }

    // If there is no FM-exchange at the boundary, open BC are assumed
    real a = aex.valueAt(idx);
    openBC = (a == 0) ? true : openBC;

    // If we assume open boundary conditions and if there is no neighbor, 
    // then simply continue without adding to the effective field.    
    if (openBC && (!system.inGeometry(neighbor_coo)
               || msat.valueAt(neighbor_idx) == 0))
      continue;
    
    // Get the dmi strengths between the center cell and the neighbor, which are
    // the harmonic means of the dmi strengths of both cells.
    real Dxz, Dxy, Dyz, Dzx, Dyx, Dzy;
    
    if (relative_coo.x) {  // derivative along x
      Dxz = dmiTensor.xxz.harmonicMean(idx, neighbor_idx);
      Dxy = dmiTensor.xxy.harmonicMean(idx, neighbor_idx);
      Dyz = dmiTensor.xyz.harmonicMean(idx, neighbor_idx);
    } else if (relative_coo.y) {  // derivative along y
      Dxz = dmiTensor.yxz.harmonicMean(idx, neighbor_idx);
      Dxy = dmiTensor.yxy.harmonicMean(idx, neighbor_idx);
      Dyz = dmiTensor.yyz.harmonicMean(idx, neighbor_idx);
    } else if (relative_coo.z) {  // derivative along z
      Dxz = dmiTensor.zxz.harmonicMean(idx, neighbor_idx);
      Dxy = dmiTensor.zxy.harmonicMean(idx, neighbor_idx);
      Dyz = dmiTensor.zyz.harmonicMean(idx, neighbor_idx);
    }

    Dzx = -Dxz;  // dmi tensor is assymetric
    Dyx = -Dxy;
    Dzy = -Dyz;

    // Distance between neighbors (the finite difference)
    real delta = relative_coo.x * system.cellsize.x +
                 relative_coo.y * system.cellsize.y +
                 relative_coo.z * system.cellsize.z;

    real3 m_;
    if (!system.inGeometry(neighbor_coo) && !openBC) { // Neumann BC
      int3 n = relative_coo * relative_coo;
      real3 Gamma = getGamma(dmiTensor, idx, n, m);
      m_ = m + (Gamma / (2*a)) * delta;
    }
    else {
      m_ = mField.vectorAt(neighbor_idx);
    }

    // Compute the effective field contribution of the DMI with the neighbor
    h.x += (Dxy * m_.y + Dxz * m_.z) / delta;
    h.y += (Dyx * m_.x + Dyz * m_.z) / delta;
    h.z += (Dzx * m_.x + Dzy * m_.y) / delta;

  }  // end loop over neighbors

  h /= msat.valueAt(idx);
  hField.setVectorInCell(idx, h);
}

__global__ void k_dmiFieldAFM(CuField hField,
                           const CuField m1Field,
                           const CuField m2Field,
                           const CuDmiTensor dmiTensor,
                           const CuDmiTensor interDmiTensor,
                           const CuParameter msat,
                           const CuParameter msat2,
                           Grid mastergrid,
                           const CuParameter aex,
                           const CuParameter afmex_nn,
                           bool openBC) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto system = hField.system;

  if (!system.grid.cellInGrid(idx))
    return;

  // When outside the geometry or msat=0, set to zero and return early
  if (!system.inGeometry(idx) || (msat.valueAt(idx) == 0)) {
    hField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  real3 m1 = m1Field.vectorAt(idx);
  real3 m2 = m2Field.vectorAt(idx);
  const int3 coo = system.grid.index2coord(idx);

  // Accumulate DMI field of cell at idx in h. Divide by msat at the end.
  real3 h{0, 0, 0};

// Loop over the 6 nearest neighbors using the neighbor's relative coordinate.
// Compute for each neighbor the DMI effective field term.
#pragma unroll
  for (int3 relative_coo : {int3{-1, 0, 0}, int3{1, 0, 0}, int3{0, -1, 0},
                            int3{0, 1, 0}, int3{0, 0, -1}, int3{0, 0, 1}}) {
    int3 neighbor_coo = mastergrid.wrap(coo + relative_coo);
    int neighbor_idx;

    if (!system.inGeometry(neighbor_coo)) { neighbor_idx = idx; }
    else { neighbor_idx = system.grid.coord2index(neighbor_coo); }
    
    // If there is no FM-exchange at the boundary, open BC are assumed
    real a = aex.valueAt(idx);
    openBC = (a == 0) ? true : openBC;

    // If we assume open boundary conditions and if there is no neighbor,
    // then simply continue without adding to the effective field.
    if (openBC && (!system.inGeometry(neighbor_coo)
               || (msat.valueAt(neighbor_idx) == 0 && msat2.valueAt(neighbor_idx) == 0)))
      continue;

    // Get the dmi strengths between the center cell and the neighbor, which are
    // the harmonic means of the dmi strengths of both cells.
    real Dxz, Dxy, Dyz, Dzx, Dyx, Dzy;
    real Dixz, Dixy, Diyz, Dizx, Diyx, Dizy;
    
    if (relative_coo.x) {  // derivative along x
      Dxz = dmiTensor.xxz.harmonicMean(idx, neighbor_idx);
      Dxy = dmiTensor.xxy.harmonicMean(idx, neighbor_idx);
      Dyz = dmiTensor.xyz.harmonicMean(idx, neighbor_idx);
      Dixz = interDmiTensor.xxz.harmonicMean(idx, neighbor_idx);
      Dixy = interDmiTensor.xxy.harmonicMean(idx, neighbor_idx);
      Diyz = interDmiTensor.xyz.harmonicMean(idx, neighbor_idx);
    } else if (relative_coo.y) {  // derivative along y
      Dxz = dmiTensor.yxz.harmonicMean(idx, neighbor_idx);
      Dxy = dmiTensor.yxy.harmonicMean(idx, neighbor_idx);
      Dyz = dmiTensor.yyz.harmonicMean(idx, neighbor_idx);
      Dixz = interDmiTensor.yxz.harmonicMean(idx, neighbor_idx);
      Dixy = interDmiTensor.yxy.harmonicMean(idx, neighbor_idx);
      Diyz = interDmiTensor.yyz.harmonicMean(idx, neighbor_idx);
    } else if (relative_coo.z) {  // derivative along z
      Dxz = dmiTensor.zxz.harmonicMean(idx, neighbor_idx);
      Dxy = dmiTensor.zxy.harmonicMean(idx, neighbor_idx);
      Dyz = dmiTensor.zyz.harmonicMean(idx, neighbor_idx);
      Dixz = interDmiTensor.zxz.harmonicMean(idx, neighbor_idx);
      Dixy = interDmiTensor.zxy.harmonicMean(idx, neighbor_idx);
      Diyz = interDmiTensor.zyz.harmonicMean(idx, neighbor_idx);
    }

    Dzx = -Dxz;  // dmi tensor is assymetric
    Dyx = -Dxy;
    Dzy = -Dyz;
    Dizx = -Dixz;
    Diyx = -Dixy;
    Dizy = -Diyz;

    // Distance between neighbors (the finite difference)
    real delta = relative_coo.x * system.cellsize.x +
                 relative_coo.y * system.cellsize.y +
                 relative_coo.z * system.cellsize.z;

    real3 m1_, m2_;
    if (!system.inGeometry(neighbor_coo)) { // Neumann BC
      int3 n = relative_coo * relative_coo;
      real3 Gamma1 = getGamma(dmiTensor, idx, n, m1);
      real3 Gamma2 = getGamma(dmiTensor, idx, n, m2);

      real an = afmex_nn.valueAt(idx);

      real3 d_m2{0, 0, 0};
      int3 other_neighbor_coo = mastergrid.wrap(coo - relative_coo);
      if(hField.cellInGeometry(other_neighbor_coo)) {
        // Approximate normal derivative of sister sublattice by taking
        // the bulk derivative closest to the edge.
        real3 m2__ = m2Field.vectorAt(other_neighbor_coo);
        d_m2 = (m2 - m2__) / delta;
      }

      m1_ = m1 + (an * cross(cross(d_m2, m1), m1) + Gamma1) * delta / (2*a);
      m2_ = m2 + (an * cross(cross((m1_ - m1)/delta, m2), m2) + Gamma2) * delta / (2*a);
    }
    else {
      m1_ = m1Field.vectorAt(neighbor_idx);
      m2_ = m2Field.vectorAt(neighbor_idx);
    }
    bool ms1 = (msat.valueAt(neighbor_idx) != 0);
    bool ms2 = (msat2.valueAt(neighbor_idx) != 0);
    // Compute the effective field contribution of the DMI with the neighbor
    h.x += (ms1 * (Dxy * m1_.y + Dxz * m1_.z) - ms2 * (Dixy * m2_.y - Dixz * m2_.z)) / delta;
    h.y += (ms1 * (Dyx * m1_.x + Dyz * m1_.z) - ms2 * (Diyx * m2_.x - Diyz * m2_.z)) / delta;
    h.z += (ms1 * (Dzx * m1_.x + Dzy * m1_.y) - ms2 * (Dizx * m2_.x - Dizy * m2_.y)) / delta;

  }  // end loop over neighbors

  // TODO: DMI exchange at a single site ???

  h /= msat.valueAt(idx);
  hField.setVectorInCell(idx, h);
}

Field evalDmiField(const Ferromagnet* magnet) {
  Field hField(magnet->system(), 3);
  if (dmiAssuredZero(magnet)) {
    hField.makeZero();
    return hField;
  }

  int ncells = hField.grid().ncells();
  auto grid = magnet->world()->mastergrid();
  auto mag = magnet->magnetization()->field().cu();
  auto msat = magnet->msat.cu();
  auto aex = magnet->aex.cu();
  auto dmiTensor = magnet->dmiTensor.cu();
  auto BC = magnet->enableOpenBC;
  
  if (!magnet->isSublattice())
    cudaLaunch(ncells, k_dmiFieldFM, hField.cu(),
              mag, dmiTensor, msat, grid, aex, BC);
  else {
    auto mag2 = magnet->hostMagnet()->getOtherSublattice(magnet)->magnetization()->field().cu();
    auto msat2 = magnet->hostMagnet()->getOtherSublattice(magnet)->msat.cu();
    auto afmex_nn = magnet->hostMagnet()->afmex_nn.cu();
    auto interDmiTensor = magnet->hostMagnet()->dmiTensor.cu();
    cudaLaunch(ncells, k_dmiFieldAFM, hField.cu(), mag, mag2,
              dmiTensor, interDmiTensor, msat, msat2, grid, aex, afmex_nn, BC);
  }
  return hField;
}

Field evalDmiEnergyDensity(const Ferromagnet* magnet) {
  if (dmiAssuredZero(magnet))
    return Field(magnet->system(), 1, 0.0);

  return evalEnergyDensity(magnet, evalDmiField(magnet), 0.5);
}

real evalDmiEnergy(const Ferromagnet* magnet) {
  if (dmiAssuredZero(magnet))
    return 0;

  real edens = dmiEnergyDensityQuantity(magnet).average()[0];

  int ncells = magnet->grid().ncells();
  real cellVolume = magnet->world()->cellVolume();
  return ncells * edens * cellVolume;
}

FM_FieldQuantity dmiFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalDmiField, 3, "dmi_field", "T");
}

FM_FieldQuantity dmiEnergyDensityQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalDmiEnergyDensity, 1, "dmi_emergy_density",
                          "J/m3");
}

FM_ScalarQuantity dmiEnergyQuantity(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, evalDmiEnergy, "dmi_energy", "J");
}
