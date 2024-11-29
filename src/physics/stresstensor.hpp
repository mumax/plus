#pragma once

#include "quantityevaluator.hpp"

class Magnet;
class Field;


Field evalStressTensor(const Magnet*);

// Stress tensor quantity with 6 symmetric stress components  
// [σxx, σyy, σzz, σxy, σxz, σyz],  
// calculated according to σ = c ε. 
M_FieldQuantity stressTensorQuantity(const Magnet*);
