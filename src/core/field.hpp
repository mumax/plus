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
  Field(std::shared_ptr<const System> system, int nComponents, real3 value);
  Field(const Field&);   // copies gpu field data
  Field(Field&& other);  // moves gpu field data

  ~Field() = default;

  Field eval() const { return Field(*this); }

  Field& operator=(Field&& other);               // moves gpu field data
  Field& operator=(const Field& other);          // copies gpu field data
  Field& operator=(const FieldQuantity& other);  // evaluates quantity on this

  Field& operator+=(const Field& other);
  Field& operator-=(const Field& other);
  Field& operator+=(const FieldQuantity& other);
  Field& operator-=(const FieldQuantity& other);

  void clear();

  bool empty() const { return !system_ || grid().ncells() == 0 || ncomp_ == 0; }
  std::shared_ptr<const System> system() const;
  int ncomp() const { return ncomp_; }
  real* device_ptr(int comp) const { return buffers_[comp].get(); }

  CuField cu() const;

  /** Copy field values into a C-style array from the device to host memory.
   *
   * @param buffer a pointer to an array of size number of cells by number of
   * components.
   */
  void getData(real* buffer) const;
  /** Copy field values into a vector from the device to host memory. */
  std::vector<real> getData() const;
  /** Set field values using a C-style array.
   *
   * Values should be provided for every cell and every component.
   * The buffer content will be copied from the host to device memory.
   *
   * @param buffer a pointer to an array of size number of cells by number of
   * components.
   */
  void setData(const real* buffer);
  /** Set field values using a vector instance.
   *
   * Values should be provided for every cell and every component.
   * The buffer content will be copied from the host to device memory.
   *
   * @param buffer a vector of size number of cells by number of components.
   */
  void setData(const std::vector<real>& buffer);
  void setUniformComponent(int comp, real value);
  void setUniformComponentInRegion(unsigned int regionIdx, int comp, real value);
  void setUniformValue(real value);
  void setUniformValue(real3 value);
  void setUniformValueInRegion(unsigned int regionIdx, real value);
  void setUniformValueInRegion(unsigned int regionIdx, real3 value);


  void makeZero();

  void setZeroOutsideGeometry();

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

  __device__ bool cellInGeometry(int) const;
  __device__ bool cellInGeometry(int3) const;

  __device__ bool cellInRegion(unsigned int, int) const;
  __device__ bool cellInRegion(unsigned int, int3) const;

  __device__ real valueAt(int idx, int comp = 0) const;
  __device__ real valueAt(int3 coo, int comp = 0) const;

  __device__ real3 vectorAt(int idx) const;
  __device__ real3 vectorAt(int3 coo) const;

  __device__ real3 vectorAt(int idx, int3 comps) const;
  __device__ real3 vectorAt(int3 coo, int3 comps) const;

  __device__ void setValueInCell(int idx, int comp, real value);
  __device__ void setVectorInCell(int idx, real3 vec);
};

__device__ inline bool CuField::cellInGrid(int idx) const {
  return system.grid.cellInGrid(idx);
}

__device__ inline bool CuField::cellInGrid(int3 coo) const {
  return system.grid.cellInGrid(coo);
}

__device__ inline bool CuField::cellInGeometry(int idx) const {
  return system.inGeometry(idx);
}

__device__ inline bool CuField::cellInGeometry(int3 coo) const {
  return system.inGeometry(coo);
}

__device__ inline bool CuField::cellInRegion(unsigned int region_idx, int idx) const {
  return system.inRegion(region_idx, idx);
}

__device__ inline bool CuField::cellInRegion(unsigned int region_idx, int3 coo) const {
  return system.inRegion(region_idx, coo);
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

__device__ inline real3 CuField::vectorAt(int idx, int3 comps) const {
  return real3{ptrs[comps.x][idx], ptrs[comps.y][idx], ptrs[comps.z][idx]};
}

__device__ inline real3 CuField::vectorAt(int3 coo, int3 comps) const {
  return vectorAt(system.grid.coord2index(coo), comps);
}

__device__ inline void CuField::setValueInCell(int idx, int comp, real value) {
  ptrs[comp][idx] = value;
}

__device__ inline void CuField::setVectorInCell(int idx, real3 vec) {
  ptrs[0][idx] = vec.x;
  ptrs[1][idx] = vec.y;
  ptrs[2][idx] = vec.z;
}
