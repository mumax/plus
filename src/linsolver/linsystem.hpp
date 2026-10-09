#pragma once

#include "gpubuffer.hpp"
#include "vec.hpp"

/** Represent a system of linear equations Ax = b with A a sparse square matrix.
 */
class LinearSystem {
 public:
  LinearSystem(int nrows, int nonZerosInRow);

  LinearSystem() = default;
  LinearSystem(const LinearSystem&) = default;
  LinearSystem(LinearSystem&&) = default;

  LinearSystem& operator=(const LinearSystem&) = default;
  LinearSystem& operator=(LinearSystem&&) = default;

  int nRows() const { return b_.size() == 0 ? 0 : b_.size() / nnz_; }
  bool empty() const { return nRows() == 0; }

  GVec matrixmul(const GVec& x) const; /** Return Ax */
  GVec residual(const GVec& x) const;  /** Return b - Ax */

  struct CuData;
  CuData cu() const;

 private:
  GVec b_;
  int nnz_;
  GVec matrixval_;
  GpuBuffer<int> matrixidx_;
};

struct LinearSystem::CuData {
  int nrows;
  int nnz;
  int* idx_;
  lsReal* val_;
  lsReal* b;

  __device__ int& matrixIdx(int row, int elem) { return idx_[row + nrows * elem]; }

  __device__ lsReal& matrixVal(int row, int elem) { return val_[row + nrows * elem]; }
};
