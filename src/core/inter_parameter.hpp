#pragma once

#include <unordered_map>

#include "datatypes.hpp"
#include "gpubuffer.hpp"
#include "reduce.hpp"
#include "system.hpp"



class System;

class CuInterParameter;

class InterParameter {
 public:
   explicit InterParameter(std::shared_ptr<const System> system);

   ~InterParameter();

   GpuBuffer<uint> regions() const;
   GpuBuffer<uint> uniqueRegions() const;
   GpuBuffer<real> values() const;
   size_t numberOfRegions() const;

   // TODO: implement get()

   void checkIdxInRegions(uint) const; // CAN THIS FUNCTION BE REMOVED???
   void setBetween(uint i, uint j, real value);

   CuInterParameter cu() const;

 public:
   std::unordered_map<uint, uint> indexMap;

 private:
    GpuBuffer<uint> uniqueRegions_;
    std::shared_ptr<const System> system_;
    GpuBuffer<uint> regions_;
    GpuBuffer<real> valuesbuffer_;
    size_t numRegions_; // TODO: cast into (u)int

};

class CuInterParameter {
 private:
   uint* regPtr_;
   real* valuePtr_;
   size_t numRegions_;

 public:
   explicit CuInterParameter(const InterParameter* p);
   __device__ real valueBetween(uint i, uint j) const;
   __device__ real valueBetween(int3 coo1, int3 coo2) const;
};

inline CuInterParameter::CuInterParameter(const InterParameter* p)
   : regPtr_(p->uniqueRegions().get()),
     valuePtr_(p->values().get()),
     numRegions_(p->numberOfRegions()) {}


__device__ __host__ inline int getLutIndex(int i, int j) {
  // Look-up Table index
  if (i <= j)
    return j * (j + 1) / 2 + i;
  return i * (i + 1) / 2 + j;
}

__device__ inline real CuInterParameter::valueBetween(uint idx1, uint idx2) const {
  int i = getIdxOnThread(regPtr_, numRegions_, idx1);
  int j = getIdxOnThread(regPtr_, numRegions_, idx2);
  return valuePtr_[getLutIndex(i, j)];
}

__device__ inline real CuInterParameter::valueBetween(int3 coo1, int3 coo2) const {
   // TODO: implement this
   //       let system be class member, then system->grid->coord2index
   return coo1.x*coo2.x;
}