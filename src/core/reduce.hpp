#pragma once

#include <vector>

#include "datatypes.hpp"

class Field;
class Grid;

real maxAbsValue(const Field&);
real maxVecNorm(const Field&);
real dotSum(const Field&, const Field&);
real fieldComponentAverage(const Field&, int comp);
std::vector<real> fieldAverage(const Field&);
bool idxInRegions(GpuBuffer<unsigned int>, unsigned int idx);
bool isUniformFieldComponent(const Field&, int);
bool isUniformField(const Field&);
bool geometriesOverlap(Grid grid1, bool const* geometry1,
                       Grid grid2, bool const* geometry2,
                       int3 lo, int3 n);