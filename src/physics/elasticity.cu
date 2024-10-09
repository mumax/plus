// TODO: check if these includes are really all necessary
#include "cudalaunch.hpp"
#include "elasticforce.hpp"
#include "elasticity.hpp"
#include "energy.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "magnetoelasticfield.hpp"
#include "magnetoelasticforce.hpp"
#include "parameter.hpp"
#include "world.hpp"


// ========== Elastic Damping ==========

bool elasticDampingAssuredZero(const Ferromagnet* magnet) {
    return ((!magnet->getEnableElastodynamics()) || magnet->eta.assuredZero());
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

Field evalElasticDamping(const Ferromagnet* magnet) {
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

FM_FieldQuantity elasticDampingQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalElasticDamping, 3,
                          "elastic_damping", "N/m3");
}

// ========== Effective Body Force ==========

Field evalEffectiveBodyForce(const Ferromagnet* magnet) {
  Field fField(magnet->system(), 3, 0.0);

  if (!elasticForceAssuredZero(magnet))
    fField += evalElasticForce(magnet);
  if (!magnetoelasticAssuredZero(magnet))
    fField += evalMagnetoelasticForce(magnet);
  if (!magnet->externalBodyForce.assuredZero())
    fField += magnet->externalBodyForce;

  return fField;
}

FM_FieldQuantity effectiveBodyForceQuantity(const Ferromagnet* magnet) {
    return FM_FieldQuantity(magnet, evalEffectiveBodyForce, 3,
                            "effective_body_force", "N/m3");
}

// ========== Elastic Accelleration ==========

__global__ void k_divideByParam(CuField field, const CuParameter param) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = field.system;
  const Grid grid = system.grid;

  // When outside the geometry, set to zero and return early
  if (!system.inGeometry(idx)) {
    if (grid.cellInGrid(idx)) {
      field.setVectorInCell(idx, real3{0, 0, 0});
    }
    return;
  }

  real p = param.valueAt(idx);
  if (p != 0) {
    field.setVectorInCell(idx, field.vectorAt(idx) / p);
  } else {
    field.setVectorInCell(idx, real3{0, 0, 0});  // substitue for infinity
  }
}

Field evalElasticAcceleration(const Ferromagnet* magnet) {
  Field aField = evalEffectiveBodyForce(magnet) + evalElasticDamping(magnet);
  
  // divide by rho if possible
  if (!magnet->rho.assuredZero())
    cudaLaunch(aField.grid().ncells(), k_divideByParam,
               aField.cu(), magnet->rho.cu());

  return aField;
}

FM_FieldQuantity elasticAccelerationQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalElasticAcceleration, 3,
                          "elastic_acceleration", "m/s2");
}

// ========== Elastic Velocity Quantity ==========

FM_FieldQuantity elasticVelocityQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet,  // use lambda function
       [](const Ferromagnet* magnet){return magnet->elasticVelocity()->eval();},
                          3, "elastic_velocity", "m/s");
}
