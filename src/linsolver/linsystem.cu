#include <stdexcept>

#include "cudalaunch.hpp"
#include "linsystem.hpp"
#include "vec.hpp"

LinearSystem::LinearSystem(int nrows, int maxNonZerosInRow)
    : nnz_(maxNonZerosInRow),
      b_(maxNonZerosInRow * nrows),
      matrixval_(GVec(maxNonZerosInRow * nrows)),
      matrixidx_(GpuBuffer<int>(maxNonZerosInRow * nrows)) {}

// For a linear system Ax=b, this kernel computes y= ka * A*x + kb * b
__global__ static void k_apply(lsReal* y,
                               LinearSystem::CuData linsys,
                               lsReal* x,
                               lsReal ka,
                               lsReal kb) {
  const int rowidx = blockIdx.x * blockDim.x + threadIdx.x;

  if (rowidx >= linsys.nrows)
    return;

  lsReal ax = 0.0;  // A*x accumulator for this row

  for (int i = 0; i < linsys.nnz; i++) {
    const int colidx = linsys.matrixIdx(rowidx, i);
    if (colidx >= 0)
      ax += linsys.matrixVal(rowidx, i) * x[colidx];
  }

  y[rowidx] = ka * ax + kb * linsys.b[rowidx];
}

// For a linear system Ax=b, this function returns y= ka * A*x + kb * b
static GVec apply(const LinearSystem& sys, const GVec& x, lsReal ka, lsReal kb) {
  if (x.size() != sys.nRows()) {
    throw std::invalid_argument(
        "The numbers of rows in the linear system does not match the number of "
        "cells of the field");
  }
  GVec y(x.size());
  cudaLaunch(x.size(), k_apply, y.get(), sys.cu(), x.get(), ka, kb);
  return y;
}

GVec LinearSystem::matrixmul(const GVec& x) const {
  return apply(*this, x, 1.0, 0.0);  // A*x
}

GVec LinearSystem::residual(const GVec& x) const {
  return apply(*this, x, -1.0, 1.0);  // b - A*x
}

LinearSystem::CuData LinearSystem::cu() const {
  return CuData{nRows(), nnz_, matrixidx_.get(), matrixval_.get(), b_.get()};
}
