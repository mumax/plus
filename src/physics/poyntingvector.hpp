#pragma once

#include "quantityevaluator.hpp"

class Magnet;
class Field;


Field evalPoyntingVector(const Magnet*);

// poynting vector
M_FieldQuantity poyntingVectorQuantity(const Magnet*);
