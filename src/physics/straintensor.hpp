#pragma once

#include "quantityevaluator.hpp"

class Magnet;
class Field;


bool strainTensorAssuredZero(const Magnet*);

Field evalStrainTensor(const Magnet*);

// Strain tensor quantity with 6 symmetric strain components
// [εxx, εyy, εzz, εxy, εxz, εyz],
// calculated according to ε = 1/2 (∇u + (∇u)^T).
M_FieldQuantity strainTensorQuantity(const Magnet*);
