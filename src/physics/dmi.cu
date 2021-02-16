#include "cudalaunch.hpp"
#include "dmi.hpp"
#include "dmitensor.hpp"
#include "energy.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "world.hpp"

bool dmiAssuredZero(const Ferromagnet* magnet) {
  return magnet->dmiTensor.assuredZero() || magnet->msat.assuredZero();
}

__device__ static inline real harmonicMean(real a, real b) {
  if (a + b == 0.0)
    return 0.0;
  return 2 * a * b / (a + b);
}

__device__ static inline real harmonicMean(const CuParameter& param,
                                           int idx1,
                                           int idx2) {
  return harmonicMean(param.valueAt(idx1), param.valueAt(idx2));
}

__global__ void k_dmiField(CuField hField,
                           const CuField mField,
                           const CuDmiTensor dmiTensor,
                           const CuParameter msat,
                           Grid mastergrid) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto system = hField.system;

  if (!system.grid.cellInGrid(idx))
    return;

  // When outside the geometry or msat=0, set to zero and return early
  if (!system.inGeometry(idx) || msat.valueAt(idx) == 0) {
    hField.setVectorInCell(idx, {0, 0, 0});
    return;
  }

  const int3 coo = system.grid.index2coord(idx);

  // Accumulate exchange field of cell at idx in h. Divide by msat at the end.
  real3 h{0, 0, 0};

// Loop over the 6 nearest neighbors using the neighbor's relative coordinate.
// Compute for each neighbor the DMI effective field term.
#pragma unroll
  for (int3 relative_coo : {int3{-1, 0, 0}, int3{1, 0, 0}, int3{0, -1, 0},
                            int3{0, 1, 0}, int3{0, 0, -1}, int3{0, 0, 1}}) {
    int3 neighbor_coo = mastergrid.wrap(coo + relative_coo);
    int neighbor_idx = system.grid.coord2index(neighbor_coo);

    // If there is no neighbor, then simply continue without adding to the
    // effective field. By doing this, we assume open boundary conditions.
    if (!system.inGeometry(neighbor_coo) || msat.valueAt(neighbor_idx) == 0)
      continue;

    // Get the dmi strengths between the center cell and the neighbor, which are
    // the harmonic means of the dmi strengths of both cells.
    real Dxz, Dxy, Dyz, Dzx, Dyx, Dzy;
    if (relative_coo.x) {  // derivative along x
      Dxz = harmonicMean(dmiTensor.xxz, idx, neighbor_idx);
      Dxy = harmonicMean(dmiTensor.xxy, idx, neighbor_idx);
      Dyz = harmonicMean(dmiTensor.xyz, idx, neighbor_idx);
    } else if (relative_coo.y) {  // derivative along y
      Dxz = harmonicMean(dmiTensor.yxz, idx, neighbor_idx);
      Dxy = harmonicMean(dmiTensor.yxy, idx, neighbor_idx);
      Dyz = harmonicMean(dmiTensor.yyz, idx, neighbor_idx);
    } else if (relative_coo.z) {  // derivative along z
      Dxz = harmonicMean(dmiTensor.zxz, idx, neighbor_idx);
      Dxy = harmonicMean(dmiTensor.zxy, idx, neighbor_idx);
      Dyz = harmonicMean(dmiTensor.zyz, idx, neighbor_idx);
    }
    Dzx = -Dxz;  // dmi tensor is assymetric
    Dyx = -Dxy;
    Dzy = -Dyz;

    // Distance between neighbors (the finite difference)
    real delta = relative_coo.x * system.cellsize.x +
                 relative_coo.y * system.cellsize.y +
                 relative_coo.z * system.cellsize.z;

    // Compute the effective field contribution of the DMI with the neighbor
    real3 m_ = mField.vectorAt(neighbor_idx);
    h.x += (Dxy * m_.y + Dxz * m_.z) / delta;
    h.y += (Dyx * m_.x + Dyz * m_.z) / delta;
    h.z += (Dzx * m_.x + Dzy * m_.y) / delta;
  }  // end loop over neighbors

  h /= msat.valueAt(idx);
  hField.setVectorInCell(idx, h);
}

Field evalDmiField(const Ferromagnet* magnet) {
  Field hField(magnet->system(), 3);

  if (dmiAssuredZero(magnet)) {
    hField.makeZero();
    return hField;
  }

  cudaLaunch(hField.grid().ncells(), k_dmiField, hField.cu(),
             magnet->magnetization()->field().cu(), magnet->dmiTensor.cu(),
             magnet->msat.cu(), magnet->world()->mastergrid());

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
