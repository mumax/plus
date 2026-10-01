#pragma once
#include "gpubuffer.hpp"

#define SINGLE 1
#define DOUBLE 2

#ifndef LS_FP_PRECISION
#define LS_FP_PRECISION DOUBLE
#endif

#if LS_FP_PRECISION == SINGLE
using lsReal = float;
#elif LS_FP_PRECISION == DOUBLE
using lsReal = double;
#endif

using GVec = GpuBuffer<lsReal>;

GVec add(const GVec&, const GVec&);
GVec add(lsReal a1, const GVec& v1, lsReal a2, const GVec& v2);
lsReal dotSum(const GVec&, const GVec&);
lsReal maxAbsValue(const GVec&);
