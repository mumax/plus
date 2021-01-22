#include "bulkdmi.hpp"
#include "cudalaunch.hpp"
#include "energy.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "world.hpp"

bool bulkDmiAssuredZero(const Ferromagnet* magnet) {
  return magnet->bdmi.assuredZero() || magnet->msat.assuredZero();
}

__device__ static inline real harmonicMean(real a, real b) {
  if (a + b == 0.0)
    return 0.0;
  return 2 * a * b / (a + b);
}

__global__ static void k_bulkDmiField(CuField hField,
                                      const CuField mField,
                                      const CuParameter bdmi,
                                      const CuParameter msat,
                                      Grid mastergrid) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!hField.cellInGeometry(idx)) {
    if (hField.cellInGrid(idx))
      hField.setVectorInCell(idx, {0, 0, 0});
    return;
  }

  const Grid grid = hField.system.grid;
  const real3 cellsize = hField.system.cellsize;

  if (msat.valueAt(idx) == 0) {
    hField.setVectorInCell(idx, {0, 0, 0});
    return;
  }

  const int3 coo = grid.index2coord(idx);
  const real d = bdmi.valueAt(idx);

  real3 h{0, 0, 0};  // accumulate exchange field of cell at idx. Devide by msat
                     // at the end

  const int3 neighborRelativeCoordinates[6] = {int3{-1, 0, 0}, int3{0, -1, 0},
                                               int3{0, 0, -1}, int3{1, 0, 0},
                                               int3{0, 1, 0},  int3{0, 0, 1}};

  for (int3 relcoo : neighborRelativeCoordinates) {
    const int3 coo_ = mastergrid.wrap(coo + relcoo);
    const int idx_ = grid.coord2index(coo_);

    if (hField.cellInGeometry(coo_) && msat.valueAt(idx_) != 0) {
      // unit vector from cell to neighbor
      real3 dr{(real)relcoo.x, (real)relcoo.y, (real)relcoo.z};

      // cellsize in direction of the neighbor
      real cs = abs(dot(dr, cellsize));

      real d_ = bdmi.valueAt(idx_);
      real3 m_ = mField.vectorAt(idx_);

      h += harmonicMean(d, d_) * cross(m_, dr) / cs;
      // real3 dmivec = harmonicMean(d, d_) * cross(interfaceNormal, dr);

      // h += cross(dmivec, m_) / cs;
    }
  }

  h /= msat.valueAt(idx);
  hField.setVectorInCell(idx, h);
}

Field evalBulkDmiField(const Ferromagnet* magnet) {
  Field hField(magnet->system(), 3);
  if (bulkDmiAssuredZero(magnet)) {
    hField.makeZero();
    return hField;
  }
  cudaLaunch(hField.grid().ncells(), k_bulkDmiField, hField.cu(),
             magnet->magnetization()->field().cu(), magnet->bdmi.cu(),
             magnet->msat.cu(), magnet->world()->mastergrid());
  return hField;
}

Field evalBulkDmiEnergyDensity(const Ferromagnet* magnet) {
  if (bulkDmiAssuredZero(magnet))
    return Field(magnet->system(), 1, 0.0);
  return evalEnergyDensity(magnet, evalBulkDmiField(magnet), 0.5);
}

real evalBulkDmiEnergy(const Ferromagnet* magnet) {
  if (bulkDmiAssuredZero(magnet))
    return 0;
  real edens = bulkDmiEnergyDensityQuantity(magnet).average()[0];
  int ncells = magnet->grid().ncells();
  real cellVolume = magnet->world()->cellVolume();
  return ncells * edens * cellVolume;
}

FM_FieldQuantity bulkDmiFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalBulkDmiField, 3, "bulkdmi_field", "T");
}

FM_FieldQuantity bulkDmiEnergyDensityQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalBulkDmiEnergyDensity, 1,
                          "bulkdmi_emergy_density", "J/m3");
}

FM_ScalarQuantity bulkDmiEnergyQuantity(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, evalBulkDmiEnergy, "bulkdmi_energy", "J");
}
