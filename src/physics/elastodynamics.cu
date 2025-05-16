#include "antiferromagnet.hpp"
#include "cudalaunch.hpp"
#include "elasticdamping.hpp"
#include "elasticforce.hpp"
#include "elastodynamics.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "magnet.hpp"
#include "magnetoelasticfield.hpp"
#include "magnetoelasticforce.hpp"
#include "parameter.hpp"


bool elasticityAssuredZero(const Magnet* magnet) {
  return ((!magnet->enableElastodynamics()) ||
          (magnet->C11.assuredZero() && magnet->C12.assuredZero() &&
           magnet->C44.assuredZero()));
}

// ========== Effective Body Force ==========

Field evalEffectiveBodyForce(const Magnet* magnet) {
  Field fField = evalElasticForce(magnet);  // safely 0 if assuredZero

  if (!magnet->externalBodyForce.assuredZero())
    fField += magnet->externalBodyForce;

  if (const Antiferromagnet* afm = magnet->asAFM()) {
    // add magnetoelastic force of both sublattices
    for (const Ferromagnet* sub : afm->sublattices()) {
      if (!magnetoelasticAssuredZero(sub))
        fField += evalMagnetoelasticForce(sub);
    }
  } else {
    // add magnetoelastic force of independent ferromagnet
    const Ferromagnet* fm = magnet->asFM();
    if (!magnetoelasticAssuredZero(fm))
      fField += evalMagnetoelasticForce(fm);
  }

  return fField;
}

M_FieldQuantity effectiveBodyForceQuantity(const Magnet* magnet) {
    return M_FieldQuantity(magnet, evalEffectiveBodyForce, 3,
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

Field evalElasticAcceleration(const Magnet* magnet) {
  Field aField = evalEffectiveBodyForce(magnet) + evalElasticDamping(magnet);
  
  // divide by rho if possible
  if (!magnet->rho.assuredZero())
    cudaLaunch(aField.grid().ncells(), k_divideByParam,
               aField.cu(), magnet->rho.cu());

  return aField;
}

M_FieldQuantity elasticAccelerationQuantity(const Magnet* magnet) {
  return M_FieldQuantity(magnet, evalElasticAcceleration, 3,
                         "elastic_acceleration", "m/s2");
}

// ========== Elastic Velocity Quantity ==========

M_FieldQuantity elasticVelocityQuantity(const Magnet* magnet) {
  return M_FieldQuantity(magnet,
       [](const Magnet* magnet){return magnet->elasticVelocity()->eval();},
                         3, "elastic_velocity", "m/s");
}
