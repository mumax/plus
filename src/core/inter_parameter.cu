#include <algorithm>
#include <string>

#include "cudalaunch.hpp"
#include "inter_parameter.hpp"
#include "reduce.hpp"

InterParameter::InterParameter(std::shared_ptr<const System> system,
                               real value,
                               std::string name,
                               std::string unit)
    : system_(system), name_(name), unit_(unit), uniformValue_(value), valuesBuffer_() {
  size_t N = 1;  // at least 1 region: default 0
  std::vector<unsigned int> uni = system->uniqueRegions;
  if (!uni.empty()) {
    N = *std::max_element(uni.begin(), uni.end()) + 1;
  }
  valuesLimit_ = N * (N - 1) / 2;
}

const std::vector<real> InterParameter::eval() const {
  // TODO: eval should include intraRegionValue_
  if (isUniform())
    return std::vector<real>(valuesLimit_, uniformValue_);
  return valuesBuffer_.getData();
}

__global__ void k_set(real* values, real value, size_t valuesLimit) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 0 || idx >= valuesLimit) {
    return;
  }
  values[idx] = value;
}

void InterParameter::setBuffer(real value) {
  cudaLaunch(valuesLimit_, k_set, valuesBuffer_.get(), value, valuesLimit_);
}

void InterParameter::set(real value) {
  uniformValue_ = value;
  if (valuesBuffer_.size() > 0)
    valuesBuffer_.recycle();
}

void InterParameter::setIntraRegionValue(real value) {
  if (isUniform()) {
    if (value == uniformValue_)
      return;  // nothing to update

    valuesBuffer_.allocate(valuesLimit_);  // make memory available
    setBuffer(uniformValue_);              // keep old uniform value
  }
  intraRegionValue_ = value;
}

void InterParameter::setBetween(unsigned int i, unsigned int j, real value) {
  system_->checkIdxInRegions(i);
  system_->checkIdxInRegions(j);
  if (i == j) {
    throw std::invalid_argument(
        "Can not set " + name_ + ", region indexes must be different: " +
        std::to_string(i) + "." + "Set intra-region value instead.");
  }

  if (isUniform()) {
    if (value == uniformValue_)
      return;  // nothing to update

    valuesBuffer_.allocate(valuesLimit_);  // make memory available
    setBuffer(uniformValue_);              // keep old uniform value
    intraRegionValue_ = uniformValue_;     // also on diagonal
  }

  // Update GPU memory directly at (i, j)
  checkCudaError(cudaMemcpy(&valuesBuffer_.get()[getLutIndex(i, j)], &value,
                            sizeof(real), cudaMemcpyHostToDevice));
  return;
}

real InterParameter::getUniformValue() const {
  if (!isUniform()) {
    throw std::invalid_argument(
        "Cannot get uniform value of non-uniform InterParameter " + name_ + ".");
  }
  return uniformValue_;
}

real InterParameter::getIntraRegionValue() const {
  if (isUniform())
    return uniformValue_;
  return intraRegionValue_;
}

real InterParameter::getBetween(unsigned int i, unsigned int j) const {
  system_->checkIdxInRegions(i);
  system_->checkIdxInRegions(j);

  if (isUniform())
    return uniformValue_;

  if (i == j)
    return intraRegionValue_;

  // legally copy single value from device to host
  real hostValue;
  checkCudaError(cudaMemcpy(&hostValue, &valuesBuffer_.get()[getLutIndex(i, j)],
                            sizeof(real), cudaMemcpyDeviceToHost));
  return hostValue;
}

CuInterParameter InterParameter::cu() const {
  return CuInterParameter(this);
}
