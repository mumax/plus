#pragma once

#include "quantityevaluator.hpp"

class Magnet;
class Field;


Field evalElasticForce(const Magnet*);

// Elastic body force due to mechanical stress gradients f = ∇σ = ∇(cε)
M_FieldQuantity elasticForceQuantity(const Magnet*);
