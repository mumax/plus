#pragma once

#include <functional>
#include <memory>
#include <utility>

#include "datatypes.hpp"
#include "dynamic_parameter.hpp"
#include "field.hpp"
#include "fieldquantity.hpp"
#include "grid.hpp"
#include "system.hpp"

class CuParameter;

class Parameter : public FieldQuantity, public DynamicParameter<real> {
 public:
  explicit Parameter(std::shared_ptr<const System> system, real value = 0.0);
  ~Parameter();

  void set(real value);
  void set(const Field& values);

  bool isUniform() const;
  bool assuredZero() const;
  int ncomp() const;
  std::shared_ptr<const System> system() const;
  /** Evaluate parameter on its field. */
  Field eval() const;
  /** Get value of uniform parameter.*/
  real getUniformValue() const;
  /** Send parameter data to the device. */
  CuParameter cu() const;

 private:
  std::shared_ptr<const System> system_;
  real uniformValue_;
  /** Store time-independent values. */
  Field* staticField_;

  friend CuParameter;
};

class CuParameter {
 public:
  const CuSystem system;
  const real uniformValue;

 private:
  real* valuesPtr;
  real* dynamicValuesPtr;

 public:
  explicit CuParameter(const Parameter* p);
  __device__ bool isUniform() const;
  __device__ real valueAt(int idx) const;
  __device__ real valueAt(int3 coo) const;
};

inline CuParameter::CuParameter(const Parameter* p)
    : system(p->system()->cu()),
      uniformValue(p->uniformValue_),
      valuesPtr(nullptr),
      dynamicValuesPtr(nullptr) {
  if (p->staticField_) {
    valuesPtr = p->staticField_->device_ptr(0);
  }

  if (p->dynamicField_) {
    dynamicValuesPtr = p->dynamicField_->device_ptr(0);
  }
}

__device__ inline bool CuParameter::isUniform() const {
  return !valuesPtr;
}

__device__ inline real CuParameter::valueAt(int idx) const {
  if (isUniform()) {
    if (dynamicValuesPtr) {
      return uniformValue + dynamicValuesPtr[idx];
    }

    return uniformValue;
  } else {
    if (dynamicValuesPtr) {
      return valuesPtr[idx] + dynamicValuesPtr[idx];
    }

    return valuesPtr[idx];
  }
}

__device__ inline real CuParameter::valueAt(int3 coo) const {
  return valueAt(system.grid.coord2index(coo));
}

class FM_CuVectorParameter;

class FM_VectorParameter : public FieldQuantity, public DynamicParameter<real3> {
 public:
  FM_VectorParameter(std::shared_ptr<const System> system,
                  real3 value = {0.0, 0.0, 0.0});
  ~FM_VectorParameter();

  void set(real3 value);
  void set(const Field& values);

  bool isUniform() const;
  bool assuredZero() const;
  int ncomp() const;
  std::shared_ptr<const System> system() const;
  Field eval() const;

  FM_CuVectorParameter cu() const;

 private:
  std::shared_ptr<const System> system_;
  real3 uniformValue_;
  Field* staticField_;

  friend FM_CuVectorParameter;
};

class AFM_CuVectorParameter;

class AFM_VectorParameter : public FieldQuantity, public DynamicParameter<real6> {
 public:
  AFM_VectorParameter(std::shared_ptr<const System> system,
                  real6 value = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  ~AFM_VectorParameter();

  void set(real6 value);
  void set(const Field& values);

  bool isUniform() const;
  bool assuredZero() const;
  int ncomp() const;
  std::shared_ptr<const System> system() const;
  Field eval() const;

  AFM_CuVectorParameter cu() const;

 private:
  std::shared_ptr<const System> system_;
  real6 uniformValue_;
  Field* staticField_;

  friend AFM_CuVectorParameter;
};

struct FM_CuVectorParameter {
 public:
  const CuSystem system;
  const real3 uniformValue;

 private:
  real* xValuesPtr;
  real* yValuesPtr;
  real* zValuesPtr;
  real* xDynamicValuesPtr;
  real* yDynamicValuesPtr;
  real* zDynamicValuesPtr;

 public:
  explicit FM_CuVectorParameter(const FM_VectorParameter*);
  __device__ bool isUniform() const;
  __device__ real3 FM_vectorAt(int idx) const;
  __device__ real3 FM_vectorAt(int3 coo) const;
};

struct AFM_CuVectorParameter {
 public:
  const CuSystem system;
  const real6 uniformValue;

 private:
  real* x1ValuesPtr;
  real* y1ValuesPtr;
  real* z1ValuesPtr;
  real* x1DynamicValuesPtr;
  real* y1DynamicValuesPtr;
  real* z1DynamicValuesPtr;
  real* x2ValuesPtr;
  real* y2ValuesPtr;
  real* z2ValuesPtr;
  real* x2DynamicValuesPtr;
  real* y2DynamicValuesPtr;
  real* z2DynamicValuesPtr;

 public:
  explicit AFM_CuVectorParameter(const AFM_VectorParameter*);
  __device__ bool isUniform() const;
  __device__ real6 AFM_vectorAt(int idx) const;
  __device__ real6 AFM_vectorAt(int3 coo) const;
};

inline FM_CuVectorParameter::FM_CuVectorParameter(const FM_VectorParameter* p)
    : system(p->system()->cu()),
      uniformValue(p->uniformValue_),
      xValuesPtr(nullptr),
      yValuesPtr(nullptr),
      zValuesPtr(nullptr),
      xDynamicValuesPtr(nullptr),
      yDynamicValuesPtr(nullptr),
      zDynamicValuesPtr(nullptr) {
  if (p->staticField_) {
    xValuesPtr = p->staticField_->device_ptr(0);
    yValuesPtr = p->staticField_->device_ptr(1);
    zValuesPtr = p->staticField_->device_ptr(2);
  }

  if (p->dynamicField_) {
    xDynamicValuesPtr = p->dynamicField_->device_ptr(0);
    yDynamicValuesPtr = p->dynamicField_->device_ptr(1);
    zDynamicValuesPtr = p->dynamicField_->device_ptr(2);
  }
}

inline AFM_CuVectorParameter::AFM_CuVectorParameter(const AFM_VectorParameter* p)
    : system(p->system()->cu()),
      uniformValue(p->uniformValue_),
      x1ValuesPtr(nullptr),
      y1ValuesPtr(nullptr),
      z1ValuesPtr(nullptr),
      x1DynamicValuesPtr(nullptr),
      y1DynamicValuesPtr(nullptr),
      z1DynamicValuesPtr(nullptr),
      x2ValuesPtr(nullptr),
      y2ValuesPtr(nullptr),
      z2ValuesPtr(nullptr),
      x2DynamicValuesPtr(nullptr),
      y2DynamicValuesPtr(nullptr),
      z2DynamicValuesPtr(nullptr)  {
  if (p->staticField_) {
    x1ValuesPtr = p->staticField_->device_ptr(0);
    y1ValuesPtr = p->staticField_->device_ptr(1);
    z1ValuesPtr = p->staticField_->device_ptr(2);
    x2ValuesPtr = p->staticField_->device_ptr(3);
    y2ValuesPtr = p->staticField_->device_ptr(4);
    z2ValuesPtr = p->staticField_->device_ptr(5);
  }

  if (p->dynamicField_) {
    x1DynamicValuesPtr = p->dynamicField_->device_ptr(0);
    y1DynamicValuesPtr = p->dynamicField_->device_ptr(1);
    z1DynamicValuesPtr = p->dynamicField_->device_ptr(2);
    x2DynamicValuesPtr = p->dynamicField_->device_ptr(3);
    y2DynamicValuesPtr = p->dynamicField_->device_ptr(4);
    z2DynamicValuesPtr = p->dynamicField_->device_ptr(5);
  }
}

__device__ inline bool FM_CuVectorParameter::isUniform() const {
  return !xValuesPtr;
}

__device__ inline bool AFM_CuVectorParameter::isUniform() const {
  return !x1ValuesPtr && !x2ValuesPtr;
}

__device__ inline real3 FM_CuVectorParameter::FM_vectorAt(int idx) const {
  if (isUniform()) {
    if (xDynamicValuesPtr && yDynamicValuesPtr && zDynamicValuesPtr) {
      real3 dynamic_value{xValuesPtr[idx], yValuesPtr[idx], zValuesPtr[idx]};

      return uniformValue + dynamic_value;
    }

    return uniformValue;
  } else {
    real3 static_value{xValuesPtr[idx], yValuesPtr[idx], zValuesPtr[idx]};
    if (xDynamicValuesPtr && yDynamicValuesPtr && zDynamicValuesPtr) {
      real3 dynamic_value{xValuesPtr[idx], yValuesPtr[idx], zValuesPtr[idx]};

      return static_value + dynamic_value;
    }

    return static_value;
  }
}

__device__ inline real6 AFM_CuVectorParameter::AFM_vectorAt(int idx) const {
  if (isUniform()) {
    if (x1DynamicValuesPtr && y1DynamicValuesPtr && z1DynamicValuesPtr &&
        x2DynamicValuesPtr && y2DynamicValuesPtr && z2DynamicValuesPtr) {
      real6 dynamic_value{x1ValuesPtr[idx], y1ValuesPtr[idx], z1ValuesPtr[idx],
                          x2ValuesPtr[idx], y2ValuesPtr[idx], z2ValuesPtr[idx]};

      return uniformValue + dynamic_value;
    }

    return uniformValue;
  } else {
    real6 static_value{x1ValuesPtr[idx], y1ValuesPtr[idx], z1ValuesPtr[idx],
                       x2ValuesPtr[idx], y2ValuesPtr[idx], z2ValuesPtr[idx]};
    if (x1DynamicValuesPtr && y1DynamicValuesPtr && z1DynamicValuesPtr && 
        x2DynamicValuesPtr && y2DynamicValuesPtr && z2DynamicValuesPtr) {
      real6 dynamic_value{x1ValuesPtr[idx], y1ValuesPtr[idx], z1ValuesPtr[idx],
                          x2ValuesPtr[idx], y2ValuesPtr[idx], z2ValuesPtr[idx]};

      return static_value + dynamic_value;
    }

    return static_value;
  }
}

__device__ inline real3 FM_CuVectorParameter::FM_vectorAt(int3 coo) const {
  return FM_vectorAt(system.grid.coord2index(coo));
}

__device__ inline real6 AFM_CuVectorParameter::AFM_vectorAt(int3 coo) const {
  return AFM_vectorAt(system.grid.coord2index(coo));
}