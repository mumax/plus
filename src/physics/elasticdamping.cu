#include "cudalaunch.hpp"
#include "elasticdamping.hpp"
#include "magnet.hpp"
#include "field.hpp"
#include "parameter.hpp"


bool elasticDampingAssuredZero(const Magnet* magnet) {
    return ((!magnet->enableElastodynamics()) || magnet->eta.assuredZero());
}

// Dedicated kernel function for -1 * eta * v; otherwise need two kernel calls.
__global__ void k_elasticDamping(CuField fField,
                                 const CuField vField,
                                 const CuParameter eta) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = fField.system;

  // When outside the geometry, set to zero and return early
  if (!system.inGeometry(idx)) {
    if (system.grid.cellInGrid(idx)) {
      fField.setVectorInCell(idx, real3{0, 0, 0});
    }
    return;
  }

  fField.setVectorInCell(idx, -eta.valueAt(idx) * vField.vectorAt(idx));
}

Field evalElasticDamping(const Magnet* magnet) {
    Field fField(magnet->system(), 3);
    if (elasticDampingAssuredZero(magnet)) {
        fField.makeZero();
        return fField;
    }

    int ncells = fField.grid().ncells();
    CuField vField = magnet->elasticVelocity()->field().cu();
    CuParameter eta = magnet->eta.cu();

    cudaLaunch(ncells, k_elasticDamping, fField.cu(), vField, eta);

    return fField;
}

M_FieldQuantity elasticDampingQuantity(const Magnet* magnet) {
  return M_FieldQuantity(magnet, evalElasticDamping, 3,
                         "elastic_damping", "N/m3");
}
