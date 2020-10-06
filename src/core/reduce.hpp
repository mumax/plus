#pragma once

#include <vector>

#include "datatypes.hpp"

class Field;

real maxAbsValue(const Field&);
real maxVecNorm(const Field&);
real dotSum(const Field&, const Field&);
real fieldComponentAverage(const Field&, int comp);
std::vector<real> fieldAverage(const Field&);
real geodesicDistance(const Field&, const Field&);
