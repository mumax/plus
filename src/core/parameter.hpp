#pragma once

#include "datatypes.hpp"
#include "fieldquantity.hpp"
#include "grid.hpp"

class Field;
class CuParameter;
class System;

class Parameter : public FieldQuantity {
 public:
  Parameter(System*, real value = 0.0);
  ~Parameter();

  void set(real value);
  void set(const Field& values);

  bool isUniform() const;
  bool assuredZero() const;
  int ncomp() const;
  System* system() const;
  Field eval() const;

  CuParameter cu() const;

 private:
  System* system_;
  real uniformValue_;
  Field* field_;
};

class CuParameter {
 public:
  const Grid grid;
  const real uniformValue;

 private:
  real* valuesPtr;

 public:
  CuParameter(Grid grid, real uniformValue)
      : grid(grid), uniformValue(uniformValue), valuesPtr(nullptr){};
  CuParameter(Grid grid, real* valuesPtr)
      : grid(grid), uniformValue(0), valuesPtr(valuesPtr){};

  __device__ bool isUniform() const;
  __device__ real valueAt(int idx) const;
  __device__ real valueAt(int3 coo) const;
};

__device__ inline bool CuParameter::isUniform() const {
  return !valuesPtr;
}

__device__ inline real CuParameter::valueAt(int idx) const {
  if (isUniform())
    return uniformValue;
  return valuesPtr[idx];
}

__device__ inline real CuParameter::valueAt(int3 coo) const {
  return valueAt(grid.coord2index(coo));
}

class CuVectorParameter;

class VectorParameter : public FieldQuantity {
 public:
  VectorParameter(System* system, real3 value = {0.0, 0.0, 0.0});
  ~VectorParameter();

  void set(real3 value);
  void set(const Field& values);

  bool isUniform() const;
  bool assuredZero() const;
  int ncomp() const;
  System* system() const;
  Field eval() const;

  CuVectorParameter cu() const;

 private:
  System* system_;
  real3 uniformValue_;
  Field* field_;
};

struct CuVectorParameter {
 public:
  const Grid grid;
  const real3 uniformValue;

 private:
  real* xValuesPtr;
  real* yValuesPtr;
  real* zValuesPtr;

 public:
  CuVectorParameter(Grid grid, real3 uniformValue);
  CuVectorParameter(Grid grid, real* xPtr, real* yPtr, real* zPtr);

  __device__ bool isUniform() const;
  __device__ real3 vectorAt(int idx) const;
  __device__ real3 vectorAt(int3 coo) const;
};

inline CuVectorParameter::CuVectorParameter(Grid grid, real3 uniformValue)
    : grid(grid),
      uniformValue(uniformValue),
      xValuesPtr(nullptr),
      yValuesPtr(nullptr),
      zValuesPtr(nullptr) {}

inline CuVectorParameter::CuVectorParameter(Grid grid,
                                            real* xPtr,
                                            real* yPtr,
                                            real* zPtr)
    : grid(grid),
      uniformValue({0, 0, 0}),
      xValuesPtr(xPtr),
      yValuesPtr(yPtr),
      zValuesPtr(zPtr) {}

__device__ inline bool CuVectorParameter::isUniform() const {
  return !xValuesPtr;
}

__device__ inline real3 CuVectorParameter::vectorAt(int idx) const {
  if (isUniform())
    return uniformValue;
  return {xValuesPtr[idx], yValuesPtr[idx], zValuesPtr[idx]};
}

__device__ inline real3 CuVectorParameter::vectorAt(int3 coo) const {
  return vectorAt(grid.coord2index(coo));
}
