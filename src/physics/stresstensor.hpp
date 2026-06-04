#pragma once

#include "quantityevaluator.hpp"

class Magnet;
class Field;

bool viscousDampingAssuredZero(const Magnet*);
bool viscousDampingTensorAssuredZero(const Magnet*);
bool viscousDampingRayleighAssuredZero(const Magnet*);
bool stressTensorAssuredZero(const Magnet*);

Field evalElasticStress(const Magnet*);
Field evalViscousStress(const Magnet*);
Field evalStressTensor(const Magnet*);

// Elastic stress tensor quantity with 6 symmetric stress components  
// calculated according to σ = c:ε. 
M_FieldQuantity elasticStressQuantity(const Magnet*);
// Viscous stress tensor quantity with 6 symmetric stress components  
// calculated according to σ = η : dε/dt.
M_FieldQuantity viscousStressQuantity(const Magnet*);
// Total stress tensor quantity with 6 symmetric stress components  
// [σxx, σyy, σzz, σxy, σxz, σyz]
M_FieldQuantity stressTensorQuantity(const Magnet*);
