#pragma once

#include <iostream>
#include <memory>
#include <vector>

#include "datatypes.hpp"
#include "fieldquantity.hpp"
#include "gpubuffer.hpp"
#include "grid.hpp"
#include "system.hpp"

class CuField;

class Field : public FieldQuantity {
  int ncomp_;
  std::shared_ptr<const System> system_;
  std::vector<GpuBuffer<real>> buffers_;
  GpuBuffer<real*> bufferPtrs_;

 public:
  Field();
  Field(std::shared_ptr<const System> system, int nComponents);
  Field(std::shared_ptr<const System> system, int nComponents, real value);
  Field(const Field&);   // copies gpu field data
  Field(Field&& other);  // moves gpu field data

  ~Field() {}

  Field eval() const { return Field(*this); }

  Field& operator=(Field&& other);               // moves gpu field data
  Field& operator=(const Field& other);          // copies gpu field data
  Field& operator=(const FieldQuantity& other);  // evaluates quantity on this

  Field& operator+=(const Field& other);
  Field& operator-=(const Field& other);
  Field& operator+=(const FieldQuantity& other);
  Field& operator-=(const FieldQuantity& other);

  void clear();

  bool empty() const { return grid().ncells() == 0 || ncomp_ == 0; }
  std::shared_ptr<const System> system() const;
  int ncomp() const { return ncomp_; }
  real* devptr(int comp) const { return buffers_[comp].get(); }

  CuField cu() const;

  void getData(real* buffer) const;
  void setData(real* buffer);
  void setUniformComponent(int comp, real value);
  void makeZero();

 private:
  void updateDevicePointersBuffer();
  void allocate();
  void free();

  friend CuField;
};

struct CuField {
 public:
  const CuSystem system;
  const int ncomp;

 private:
  real** ptrs;

 public:
  explicit CuField(const Field* f)
      : system(f->system()->cu()),
        ncomp(f->ncomp()),
        ptrs(f->bufferPtrs_.get()) {}

  __device__ bool cellInGrid(int) const;
  __device__ bool cellInGrid(int3) const;

  __device__ real valueAt(int idx, int comp = 0) const;
  __device__ real valueAt(int3 coo, int comp = 0) const;

  __device__ real3 vectorAt(int idx) const;
  __device__ real3 vectorAt(int3 coo) const;

  __device__ void setValueInCell(int idx, int comp, real value);
  __device__ void setVectorInCell(int idx, real3 vec);
};

__device__ inline bool CuField::cellInGrid(int idx) const {
  return system.grid.cellInGrid(idx);
}

__device__ inline bool CuField::cellInGrid(int3 coo) const {
  return system.grid.cellInGrid(coo);
}

__device__ inline real CuField::valueAt(int idx, int comp) const {
  return ptrs[comp][idx];
}

__device__ inline real CuField::valueAt(int3 coo, int comp) const {
  return valueAt(system.grid.coord2index(coo), comp);
}

__device__ inline real3 CuField::vectorAt(int idx) const {
  return real3{ptrs[0][idx], ptrs[1][idx], ptrs[2][idx]};
}

__device__ inline real3 CuField::vectorAt(int3 coo) const {
  return vectorAt(system.grid.coord2index(coo));
}

__device__ inline void CuField::setValueInCell(int idx, int comp, real value) {
  ptrs[comp][idx] = value;
}

__device__ inline void CuField::setVectorInCell(int idx, real3 vec) {
  ptrs[0][idx] = vec.x;
  ptrs[1][idx] = vec.y;
  ptrs[2][idx] = vec.z;
}
