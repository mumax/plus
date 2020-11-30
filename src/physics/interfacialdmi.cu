#include "cudalaunch.hpp"
#include "energy.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "interfacialdmi.hpp"
#include "parameter.hpp"
#include "world.hpp"

// At the moment we only suport interfacially induced dmi in the xy plane
const real3 interfaceNormal{0, 0, 1};

bool interfacialDmiAssuredZero(const Ferromagnet* magnet) {
  return magnet->idmi.assuredZero() || magnet->msat.assuredZero();
}

__device__ static inline real harmonicMean(real a, real b) {
  if (a + b == 0.0)
    return 0.0;
  return 2 * a * b / (a + b);
}

__global__ void k_interfacialDmiField(CuField hField,
                                      const CuField mField,
                                      const CuParameter idmi,
                                      const CuParameter msat,
                                      const real3 interfaceNormal,
                                      Grid mastergrid) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  const Grid grid = hField.system.grid;
  const real3 cellsize = hField.system.cellsize;

  if (!grid.cellInGrid(idx))
    return;

  if (msat.valueAt(idx) == 0) {
    hField.setVectorInCell(idx, {0, 0, 0});
    return;
  }

  const int3 coo = grid.index2coord(idx);
  const real d = idmi.valueAt(idx);

  real3 h{0, 0, 0};  // accumulate exchange field of cell at idx. Devide by msat
                     // at the end

  const int3 neighborRelativeCoordinates[6] = {int3{-1, 0, 0}, int3{0, -1, 0},
                                               int3{0, 0, -1}, int3{1, 0, 0},
                                               int3{0, 1, 0},  int3{0, 0, 1}};

  for (int3 relcoo : neighborRelativeCoordinates) {
    const int3 coo_ = mastergrid.wrap(coo + relcoo);
    const int idx_ = grid.coord2index(coo_);

    if (grid.cellInGrid(coo_) && msat.valueAt(idx_) != 0) {
      // unit vector from cell to neighbor
      real3 dr{(real)relcoo.x, (real)relcoo.y, (real)relcoo.z};

      // cellsize in direction of the neighbor
      real cs = abs(dot(dr, cellsize));

      real d_ = idmi.valueAt(idx_);
      real3 m_ = mField.vectorAt(idx_);
      real3 dmivec = harmonicMean(d, d_) * cross(interfaceNormal, dr);

      h += cross(dmivec, m_) / cs;
    }
  }

  h /= msat.valueAt(idx);
  hField.setVectorInCell(idx, h);
}

Field evalInterfacialDmiField(const Ferromagnet* magnet) {
  Field hField(magnet->system(), 3);
  if (interfacialDmiAssuredZero(magnet)) {
    hField.makeZero();
    return hField;
  }
  cudaLaunch(hField.grid().ncells(), k_interfacialDmiField, hField.cu(),
             magnet->magnetization()->field().cu(), magnet->idmi.cu(),
             magnet->msat.cu(), interfaceNormal, magnet->world()->mastergrid());
  return hField;
}

Field evalInterfacialDmiEnergyDensity(const Ferromagnet* magnet) {
  if (interfacialDmiAssuredZero(magnet))
    return Field(magnet->system(), 1, 0.0);
  return evalEnergyDensity(magnet, evalInterfacialDmiField(magnet), 0.5);
}

real evalInterfacialDmiEnergy(const Ferromagnet* magnet) {
  if (interfacialDmiAssuredZero(magnet))
    return 0;
  real edens = interfacialDmiEnergyDensityQuantity(magnet).average()[0];
  int ncells = magnet->grid().ncells();
  real cellVolume = magnet->world()->cellVolume();
  return ncells * edens * cellVolume;
}

FM_FieldQuantity interfacialDmiFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalInterfacialDmiField, 3,
                          "interfacialdmi_field", "T");
}

FM_FieldQuantity interfacialDmiEnergyDensityQuantity(
    const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalInterfacialDmiEnergyDensity, 1,
                          "interfacialdmi_emergy_density", "J/m3");
}

FM_ScalarQuantity interfacialDmiEnergyQuantity(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, evalInterfacialDmiEnergy,
                           "interfacialdmi_energy", "J");
}
