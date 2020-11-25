#include <stdexcept>
#include <vector>

#include "cudalaunch.hpp"
#include "field.hpp"
#include "fieldops.hpp"

__global__ void k_addFields(CuField y,
                            real a1,
                            const CuField x1,
                            real a2,
                            const CuField x2) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!y.cellInGrid(idx))
    return;
  for (int c = -0; c < y.ncomp; c++) {
    real term1 = a1 * x1.valueAt(idx, c);
    real term2 = a2 * x2.valueAt(idx, c);
    y.setValueInCell(idx, c, term1 + term2);
  }
}

inline void add(Field& y, real a1, const Field& x1, real a2, const Field& x2) {
  if (x1.system() != y.system() || x2.system() != y.system()) {
    throw std::invalid_argument(
        "Fields can not be added together because they belong to different "
        "systems)");
  }
  if (x1.ncomp() != y.ncomp() || x1.ncomp() != y.ncomp()) {
    throw std::invalid_argument(
        "Fields can not be added because they do not have the same number of "
        "components");
  }
  int ncells = y.grid().ncells();
  cudaLaunch(ncells, k_addFields, y.cu(), a1, x1.cu(), a2, x2.cu());
}

Field add(real a1, const Field& x1, real a2, const Field& x2) {
  Field y(x1.system(), x1.ncomp());
  add(y, a1, x1, a2, x2);
  return y;
}

Field add(const Field& x1, const Field& x2) {
  return add(1, x1, 1, x2);
}

void addTo(Field& y, real a, const Field& x) {
  add(y, 1, y, a, x);
}

// TODO: this can be done much more efficient
Field add(std::vector<const Field*> x, std::vector<real> weights) {
  // TODO:: throw error if inputs are not compatible
  Field y = weights.at(0) * (*x.at(0));
  if (x.size() == 1) {
    return y;
  }

  for (int n = 1; n < x.size(); n++) {
    if (weights.at(n) != 0.0) {
      addTo(y, weights.at(n), *x.at(n));
    }
  }
  return y;
}

Field operator*(real a, const Field& x) {
  return add(0, x, a, x);
}

__global__ void k_addConstant(CuField y,
                              const CuField x,
                              real value,
                              int comp) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!y.cellInGrid(idx))
    return;
  y.setValueInCell(idx, comp, x.valueAt(idx, comp) + value);
}

__global__ void k_normalize(CuField dst, const CuField src) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!dst.cellInGrid(idx))
    return;
  real norm2 = 0.0;
  for (int c = 0; c < src.ncomp; c++) {
    real v = src.valueAt(idx, c);
    norm2 += v * v;
  }
  real invnorm = rsqrt(norm2);
  for (int c = 0; c < src.ncomp; c++) {
    real value = src.valueAt(idx, c) * invnorm;
    dst.setValueInCell(idx, c, value);
  }
}

Field normalized(const Field& src) {
  Field dst(Field(src.system(), src.ncomp()));
  cudaLaunch(dst.grid().ncells(), k_normalize, dst.cu(), src.cu());
  return dst;
}

void normalize(Field& f) {
  cudaLaunch(f.grid().ncells(), k_normalize, f.cu(), f.cu());
}