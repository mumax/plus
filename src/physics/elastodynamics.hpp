#pragma once

#include "quantityevaluator.hpp"

class Magnet;
class Field;


// Should only be checked for host-magnets: AFM or independent FM
bool elasticityAssuredZero(const Magnet*);

Field evalEffectiveBodyForce(const Magnet*);  // TODO: or overload?
Field evalElasticAcceleration(const Magnet*);

// Elastic effective body force is the sum of elastic, magnetoelastic and
// external body forces. Elastic damping is not included.
M_FieldQuantity effectiveBodyForceQuantity(const Magnet*);

// Translate const Variable* elasticVelocity to usable FM_fieldQuantity
M_FieldQuantity elasticVelocityQuantity(const Magnet*);

// Elastic acceleration includes all effects that influence the elastic velocity
// including elastic, magnetoelastic and external body forces, and elastic damping.
M_FieldQuantity elasticAccelerationQuantity(const Magnet*);
