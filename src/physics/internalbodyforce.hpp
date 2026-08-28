#pragma once

#include "quantityevaluator.hpp"

class Magnet;
class Field;


Field evalInternalBodyForce(const Magnet*);

// Internal body force due to stress divergence f = ∇·σ
M_FieldQuantity internalBodyForceQuantity(const Magnet*);
