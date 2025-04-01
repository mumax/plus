#pragma once

#include "quantityevaluator.hpp"

class Magnet;
class Field;


bool strainTensorAssuredZero(const Magnet*);

Field evalStrainTensor(const Magnet*);

// Strain tensor quantity with 6 symmetric strain components
// [εxx, εyy, εzz, εxy, εxz, εyz],
// calculated according to ε = C⁻¹ : σ.
M_FieldQuantity strainTensorQuantity(const Magnet*);
