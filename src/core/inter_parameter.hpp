#pragma once

#include "datatypes.hpp"
#include "gpubuffer.hpp"
#include "reduce.hpp"
#include "system.hpp"



class System;

class CuInterParameter;

class InterParameter {
 public:
  explicit InterParameter(std::shared_ptr<const System> system,
                           real value,
                           std::string name = "",
                           std::string unit = "");
  ~InterParameter() {};

  std::string name() const { return name_; }
  std::string unit() const { return unit_; }
  int ncomp() const { return 1; }
  bool isUniform() const { return valuesBuffer_.size() == 0; }
  bool assuredZero() const { return isUniform() && uniformValue_ == 0.0; };

  const std::vector<real> eval() const;

  // set to uniform value
  void set(real value);
  void setBetween(unsigned int i, unsigned int j, real value);
  real getUniformValue() const;
  real getBetween(unsigned int i, unsigned int j) const;

  CuInterParameter cu() const;

  // TODO: these user-convenience functions should probably move
  const std::vector<unsigned int> uniqueRegions() const { return system_->uniqueRegions; }
  int numberOfRegions() const { return system_->uniqueRegions.size(); }

 private:
  /** Set full buffer to uniform value. */
  void setBuffer(real value);

  std::string name_;
  std::string unit_;

  std::shared_ptr<const System> system_;
  real uniformValue_;
  GpuBuffer<real> valuesBuffer_;
  size_t valuesLimit_;  // N(N-1)/2 with N = maxRegionIdx+1

  friend CuInterParameter;
};

class CuInterParameter {
 private:
  real* valuesPtr_;
  real uniformValue_;

 public:
  explicit CuInterParameter(const InterParameter* p);
  __device__ bool isUniform() const { return !valuesPtr_; }
  __device__ real valueBetween(unsigned int i, unsigned int j) const;
};

inline CuInterParameter::CuInterParameter(const InterParameter* p)
    : uniformValue_(p->uniformValue_),
      valuesPtr_(nullptr) {
  if (!p->isUniform()) {
    valuesPtr_ = p->valuesBuffer_.get();
  }
}

/** Look-up Table index
 * This transforms a bottom triangular matrix row-first with 0-diagonal
 * to a 1D array index, or an upper triangular matrix column-first.
 * No built-in safety checks for the sake of performance.
 */
__device__ __host__ inline int getLutIndex(int i, int j) {
  if (i > j)
    return i * (i - 1) / 2 + j;
  return j * (j - 1) / 2 + i;
}

__device__ inline real CuInterParameter::valueBetween(unsigned int idx1, unsigned int idx2) const {
  if (isUniform())
    return uniformValue_;
  return valuesPtr_[getLutIndex(idx1, idx2)];
}