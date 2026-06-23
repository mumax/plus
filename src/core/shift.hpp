#pragma once

#include "datatypes.hpp"

class Field;

int calculateShiftDirection(const Field& field, int comp, int axis, real3 leftValue, real3 rightValue);
Field shift(const Field& field, int dir, int comp, int axis, real3 leftValue, real3 RightValue);