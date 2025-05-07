#pragma once

#include "datatypes.hpp"
#include "gpubuffer.hpp"

class Field;

int calculateShiftDirection(const Field& field, int comp);
Field shift(const Field& field, int dir, int comp, real3 leftValue, real3 RightValue);

template <typename T>
GpuBuffer<T> shift(const GpuBuffer<T>& data, int dir, int ncells, T left, T right);