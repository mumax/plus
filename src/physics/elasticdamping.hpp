#pragma once

#include "quantityevaluator.hpp"

class Magnet;
class Field;


bool elasticDampingAssuredZero(const Magnet*);

Field evalElasticDamping(const Magnet*);

// Elastic damping proportional to η and velocity: -ηv.
M_FieldQuantity elasticDampingQuantity(const Magnet*);
