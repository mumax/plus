#pragma once

#include "quantityevaluator.hpp"

class Magnet;
class Field;


Field evalStressRate(const Magnet*);

M_FieldQuantity stressRateQuantity(const Magnet*);
