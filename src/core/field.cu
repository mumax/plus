#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "cudaerror.hpp"
#include "cudalaunch.hpp"
#include "cudastream.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "fieldquantity.hpp"
#include "gpubuffer.hpp"
#include "system.hpp"

Field::Field() : system_(nullptr), ncomp_(0) {}

Field::Field(std::shared_ptr<const System> system, int nComponents)
    : system_(system), ncomp_(nComponents) {
  allocate();
}

Field::Field(std::shared_ptr<const System> system, int nComponents, real value)
    : Field(system, nComponents) {
  for (int comp = 0; comp < nComponents; comp++)
    setUniformComponent(comp, value);
}

Field::Field(const Field& other)
    : system_(other.system_), ncomp_(other.ncomp_) {
  buffers_ = other.buffers_;
  updateDevicePointersBuffer();
}

Field::Field(Field&& other) : system_(other.system_), ncomp_(other.ncomp_) {
  buffers_ = std::move(other.buffers_);
  bufferPtrs_ = std::move(other.bufferPtrs_);
  other.clear();
}

Field& Field::operator=(const Field& other) {
  if (this == &other)
    return *this;
  return *this = std::move(Field(other));  // moves a copy of other to this
}

Field& Field::operator=(const FieldQuantity& q) {
  return *this = std::move(q.eval());
}

Field& Field::operator=(Field&& other) {
  system_ = other.system_;
  ncomp_ = other.ncomp_;
  buffers_ = std::move(other.buffers_);
  bufferPtrs_ = std::move(other.bufferPtrs_);
  other.clear();
  return *this;
}

void Field::clear() {
  system_ = nullptr;
  ncomp_ = 0;
  free();
}

std::shared_ptr<const System> Field::system() const {
  return system_;
}

void Field::updateDevicePointersBuffer() {
  std::vector<real*> bufferPtrsOnHost(ncomp_);
  std::transform(buffers_.begin(), buffers_.end(), bufferPtrsOnHost.begin(),
                 [](auto& buf) { return buf.get(); });
  bufferPtrs_ = GpuBuffer<real*>(bufferPtrsOnHost);
}

void Field::allocate() {
  free();

  if (empty())
    return;

  buffers_ =
      std::vector<GpuBuffer<real>>(ncomp_, GpuBuffer<real>(grid().ncells()));

  updateDevicePointersBuffer();
}

void Field::free() {
  buffers_.clear();
  bufferPtrs_.recycle();
}

CuField Field::cu() const {
  return CuField(grid(), ncomp_, bufferPtrs_.get());
}

void Field::getData(real* buffer) const {
  for (int c = 0; c < ncomp_; c++) {
    real* bufferComponent = buffer + c * grid().ncells();
    checkCudaError(cudaMemcpyAsync(bufferComponent, buffers_[c].get(),
                                   grid().ncells() * sizeof(real),
                                   cudaMemcpyDeviceToHost, getCudaStream()));
  }
}

void Field::setData(real* buffer) {
  for (int c = 0; c < ncomp_; c++) {
    real* bufferComponent = buffer + c * grid().ncells();
    checkCudaError(cudaMemcpyAsync(buffers_[c].get(), bufferComponent,
                                   grid().ncells() * sizeof(real),
                                   cudaMemcpyHostToDevice, getCudaStream()));
  }
}

__global__ void k_setComponent(CuField f, real value, int comp) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (!f.cellInGrid(idx))
    return;
  f.setValueInCell(idx, comp, value);
}

void Field::setUniformComponent(int comp, real value) {
  cudaLaunch(grid().ncells(), k_setComponent, cu(), value, comp);
}

void Field::makeZero() {
  for (int comp = 0; comp < ncomp_; comp++)
    setUniformComponent(comp, 0.0);
}

Field& Field::operator+=(const Field& other) {
  addTo(*this, 1, other);
  return *this;
}

Field& Field::operator-=(const Field& other) {
  addTo(*this, -1, other);
  return *this;
}

Field& Field::operator+=(const FieldQuantity& q) {
  addTo(*this, 1, q.eval());
  return *this;
}

Field& Field::operator-=(const FieldQuantity& q) {
  addTo(*this, -1, q.eval());
  return *this;
}
