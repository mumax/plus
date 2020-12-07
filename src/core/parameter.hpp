#pragma once

#include <memory>

#include "datatypes.hpp"
#include "fieldquantity.hpp"
#include "grid.hpp"

class Field;
class CuParameter;
class System;

class Parameter : public FieldQuantity {
 public:
  explicit Parameter(std::shared_ptr<const System> system, real value = 0.0);
  ~Parameter();

  void set(real value);
  void set(const Field& values);

  bool isUniform() const;
  bool assuredZero() const;
  int ncomp() const;
  std::shared_ptr<const System> system() const;
  Field eval() const;

  CuParameter cu() const;

 private:
  std::shared_ptr<const System> system_;
  real uniformValue_;
  Field* field_;

  friend CuParameter;
};

class CuParameter {
 public:
  const CuSystem system;
  const real uniformValue;

 private:
  real* valuesPtr;

 public:
  explicit CuParameter(const Parameter* p);
  __device__ bool isUniform() const;
  __device__ real valueAt(int idx) const;
  __device__ real valueAt(int3 coo) const;
};

inline CuParameter::CuParameter(const Parameter* p)
    : system(p->system()->cu()),
      uniformValue(p->uniformValue_),
      valuesPtr(nullptr) {
  if (p->field_) {
    valuesPtr = p->field_->devptr(0);
  }
}

__device__ inline bool CuParameter::isUniform() const {
  return !valuesPtr;
}

__device__ inline real CuParameter::valueAt(int idx) const {
  if (isUniform())
    return uniformValue;
  return valuesPtr[idx];
}

__device__ inline real CuParameter::valueAt(int3 coo) const {
  return valueAt(system.grid.coord2index(coo));
}

class CuVectorParameter;

class VectorParameter : public FieldQuantity {
 public:
  VectorParameter(std::shared_ptr<const System> system,
                  real3 value = {0.0, 0.0, 0.0});
  ~VectorParameter();

  void set(real3 value);
  void set(const Field& values);

  bool isUniform() const;
  bool assuredZero() const;
  int ncomp() const;
  std::shared_ptr<const System> system() const;
  Field eval() const;

  CuVectorParameter cu() const;

 private:
  std::shared_ptr<const System> system_;
  real3 uniformValue_;
  Field* field_;

  friend CuVectorParameter;
};

struct CuVectorParameter {
 public:
  const CuSystem system;
  const real3 uniformValue;

 private:
  real* xValuesPtr;
  real* yValuesPtr;
  real* zValuesPtr;

 public:
  explicit CuVectorParameter(const VectorParameter*);
  __device__ bool isUniform() const;
  __device__ real3 vectorAt(int idx) const;
  __device__ real3 vectorAt(int3 coo) const;
};

inline CuVectorParameter::CuVectorParameter(const VectorParameter* p)
    : system(p->system()->cu()),
      uniformValue(p->uniformValue_),
      xValuesPtr(nullptr),
      yValuesPtr(nullptr),
      zValuesPtr(nullptr) {
  if (p->field_) {
    xValuesPtr = p->field_->devptr(0);
    yValuesPtr = p->field_->devptr(1);
    zValuesPtr = p->field_->devptr(2);
  }
}

__device__ inline bool CuVectorParameter::isUniform() const {
  return !xValuesPtr;
}

__device__ inline real3 CuVectorParameter::vectorAt(int idx) const {
  if (isUniform())
    return uniformValue;
  return {xValuesPtr[idx], yValuesPtr[idx], zValuesPtr[idx]};
}

__device__ inline real3 CuVectorParameter::vectorAt(int3 coo) const {
  return vectorAt(system.grid.coord2index(coo));
}
