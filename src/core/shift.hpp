#pragma once

#include "datatypes.hpp"

class Field;

int calculateShiftDirection(const Field& field, real3 leftValue, real3 rightValue, int comp, int axis);
Field shift(const Field& field, int dir, int axis, int comp, real3 leftValue, real3 RightValue);