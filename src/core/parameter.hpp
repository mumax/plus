#pragma once

#include <functional>
#include <memory>
#include <utility>

#include "datatypes.hpp"
#include "fieldquantity.hpp"
#include "grid.hpp"
#include "system.hpp"

class Field;
class CuParameter;

class Parameter : public FieldQuantity {
 public:
  /** Declare a short name for a time-dependent function */
  typedef std::function<real(real)> time_function;

  explicit Parameter(std::shared_ptr<const System> system, real value = 0.0);
  ~Parameter();

  void set(real value);
  void set(const Field& values);
  /** Add time-dependent function that is the same for every grid cell.
  * 
  * Parameter values will be evaluated as:
  * a) uniform_value + term(t)
  * b) cell_value + term(t)
  * 
  * @param term time-dependent function.
  */
  void addTimeDependentTerm(const time_function& term);
  /** Add time-dependent function that is the same for every grid cell.
  * 
  * Parameter values will be evaluated as:
  * a) uniform_value + term(t) * mask
  * b) cell_value + term(t) * cell_mask_value
  *
  * @param term time-dependent function.
  * @param mask define how the magnitude of the time-dependent function should
  *             depend on cell coordinates. The input value will be copied.
  */
  void addTimeDependentTerm(const time_function& term, const Field& mask);
   /** Remove all time-dependet terms and their masks. */
  void removeAllTimeDependentTerms();

  bool isUniform() const;
  bool assuredZero() const;
  int ncomp() const;
  std::shared_ptr<const System> system() const;
  /** Evaluate parameter on its field. */
  Field eval() const;

  /** Send parameter data to the device. */
  CuParameter cu() const;

 private:
  std::shared_ptr<const System> system_;
  /** List of all time dependent terms */
  std::vector<std::pair<time_function, Field*>> time_dep_terms;
  real uniformValue_;
  Field* field_;

  Field evalTimeDependentTerms(real t) const;

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
