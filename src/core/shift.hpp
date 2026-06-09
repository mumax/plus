#pragma once

#include "datatypes.hpp"
#include "gpubuffer.hpp"

class Field;

int calculateShiftDirection(const Field& field, real3 leftValue, real3 rightValue, int comp, int axis);
Field shift(const Field& field, int dir, int axis, int comp, real3 leftValue, real3 RightValue);

template <typename T>
GpuBuffer<T> shift(const GpuBuffer<T>& data, int dir, int axis, int ncells, int nx, int ny, T left, T right);