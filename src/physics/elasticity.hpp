#pragma once

#include "ferromagnetquantity.hpp"

class Ferromagnet;
class Field;


bool elasticDampingAssuredZero(const Ferromagnet*);

Field evalElasticDamping(const Ferromagnet*);
Field evalEffectiveBodyForce(const Ferromagnet*);
Field evalElasticAcceleration(const Ferromagnet*);

// Elastic damping proportional to η and velocity: -ηv.
FM_FieldQuantity elasticDampingQuantity(const Ferromagnet*);

// Elastic effective body force is the sum of elastic, magnetoelastic and
// external body forces. Elastic damping is not included.
FM_FieldQuantity effectiveBodyForceQuantity(const Ferromagnet*);

// Translate const Variable* elasticVelocity to usable FM_fieldQuantity
FM_FieldQuantity elasticVelocityQuantity(const Ferromagnet*);

// Elastic acceleration includes all effects that influence the elastic velocity
// including elastic, magnetoelastic and external body forces, and elastic damping.
FM_FieldQuantity elasticAccelerationQuantity(const Ferromagnet*);