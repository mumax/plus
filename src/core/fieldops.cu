#include <stdexcept>
#include <vector>

#include "cudalaunch.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "reduce.hpp"

__global__ void k_addFields(CuField y,
                            real a1,
                            const CuField x1,
                            real a2,
                            const CuField x2) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!y.cellInGeometry(idx))
    return;
  for (int c = 0; c < y.ncomp; c++) {
    real term1 = a1 * x1.valueAt(idx, c);
    real term2 = a2 * x2.valueAt(idx, c);
    y.setValueInCell(idx, c, term1 + term2);
  }
}

__global__ void k_addFields(CuField y,
                            real3 a1,
                            const CuField x1,
                            real3 a2,
                            const CuField x2) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!y.cellInGeometry(idx))
    return;

  real3 term1 = a1 * x1.vectorAt(idx);
  real3 term2 = a2 * x2.vectorAt(idx);
  y.setVectorInCell(idx, term1 + term2);
}

__global__ void k_addFields(CuField y,
                            const CuField a1,
                            const CuField x1,
                            const CuField a2,
                            const CuField x2) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!y.cellInGeometry(idx))
    return;

  for (int c = 0; c < y.ncomp; c++) {
    int c_a1 = (a1.ncomp == y.ncomp) ? c : 0;  // follow comp or be scalar
    real term1 = a1.valueAt(idx, c_a1) * x1.valueAt(idx, c);
    int c_a2 = (a2.ncomp == y.ncomp) ? c : 0;
    real term2 = a2.valueAt(idx, c_a2) * x2.valueAt(idx, c);
    y.setValueInCell(idx, c, term1 + term2);
  }
}

// same as above, but without a1. Otherwise need to make a whole onefield first
__global__ void k_addFields(CuField y,
                            const CuField x1,
                            const CuField a2,
                            const CuField x2) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!y.cellInGeometry(idx))
    return;

  for (int c = 0; c < y.ncomp; c++) {
    int c_a2 = (a2.ncomp == y.ncomp) ? c : 0;
    y.setValueInCell(idx, c, x1.valueAt(idx, c)
                            + a2.valueAt(idx, c_a2) * x2.valueAt(idx, c));
  }
}

inline void add(Field& y, real a1, const Field& x1, real a2, const Field& x2) {
  if (x1.system() != y.system() || x2.system() != y.system()) {
    throw std::invalid_argument(
        "Fields can not be added together because they belong to different "
        "systems.");
  }
  if ((x1.ncomp() != y.ncomp() || x1.ncomp() != y.ncomp()) ) {
    throw std::invalid_argument(
        "Fields can not be added because they do not have the same number of "
        "components.");
  }
  int ncells = y.grid().ncells();
  cudaLaunch(ncells, k_addFields, y.cu(), a1, x1.cu(), a2, x2.cu());
}

inline void add(Field& y,
                real3 a1,
                const Field& x1,
                real3 a2,
                const Field& x2) {
  if (x1.system() != y.system() || x2.system() != y.system()) {
    throw std::invalid_argument(
        "Fields can not be added together because they belong to different "
        "systems.");
  }
  if (x1.ncomp() != y.ncomp() || x1.ncomp() != y.ncomp()) {
    throw std::invalid_argument(
        "Fields can not be added because they do not have the same number of "
        "components.");
  }
  if (x1.ncomp() != 3) {
    throw std::invalid_argument("Fields should have 3 components.");
  }
  int ncells = y.grid().ncells();
  cudaLaunch(ncells, k_addFields, y.cu(), a1, x1.cu(), a2, x2.cu());
}

inline void add(Field& y,
                const Field& a1,
                const Field& x1,
                const Field& a2,
                const Field& x2) {
  if (x1.system() != y.system() || x2.system() != y.system() ||
      a1.system() != y.system() || a2.system() != y.system()) {
    throw std::invalid_argument(
        "Fields can not be multiplied and added together because they belong to "
        "different systems.");
  }
  if (x1.ncomp() != y.ncomp() || x2.ncomp() != y.ncomp()) {
    throw std::invalid_argument(
        "Fields can not be added because they do not have the same number of "
        "components.");
  }
  if (!(a1.ncomp() == 1 || a1.ncomp() == x1.ncomp()) ||
      !(a2.ncomp() == 1 || a2.ncomp() == x2.ncomp())) {
    throw std::invalid_argument(
        "Weights need to be scalar (1 component) or have the same number of "
        "components as the fields."
    );
  }

  int ncells = y.grid().ncells();
  cudaLaunch(ncells, k_addFields, y.cu(), a1.cu(), x1.cu(), a2.cu(), x2.cu());
}

Field add(real a1, const Field& x1, real a2, const Field& x2) {
  Field y(x1.system(), x1.ncomp());
  add(y, a1, x1, a2, x2);
  return y;
}

Field add(real3 a1, const Field& x1, real3 a2, const Field& x2) {
  Field y(x1.system(), x1.ncomp());
  add(y, a1, x1, a2, x2);
  return y;
}

Field add(const Field& a1, const Field& x1, const Field& a2, const Field& x2) {
  Field y(x1.system(), x1.ncomp());
  add(y, a1, x1, a2, x2);
  return y;
}

Field add(const Field& x1, const Field& x2) {
  return add(1, x1, 1, x2);
}

Field operator+(const Field& x1, const Field& x2) {
  return add(x1, x2);
}

void addTo(Field& y, real a, const Field& x) {
  add(y, 1, y, a, x);
}

void addTo(Field& y, real3 a, const Field& x) {
  real3 a0 = real3{1, 1, 1};
  add(y, a0, y, a, x);
}

void addTo(Field& y, const Field& a, const Field& x) {
  if (x.system() != y.system() || a.system() != y.system()) {
    throw std::invalid_argument(
        "Fields can not be multiplied and added together because they belong to "
        "different systems.");
  }
  if (x.ncomp() != y.ncomp()) {
    throw std::invalid_argument(
        "Fields can not be added because they do not have the same number of "
        "components.");
  }
  if (!(a.ncomp() == 1 || a.ncomp() == x.ncomp())) {
    throw std::invalid_argument(
        "Weights need to be scalar (1 component) or have the same number of "
        "components as the fields."
    );
  }

  int ncells = y.grid().ncells();
  cudaLaunch(ncells, k_addFields, y.cu(), y.cu(), a.cu(), x.cu());  // x1 = y
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
  if (!y.cellInGeometry(idx))
    return;
  y.setValueInCell(idx, comp, x.valueAt(idx, comp) + value);
}

Field addConstant(const Field& x, real value) {
  Field y(x.system(), x.ncomp());
  for (int i = 0; i < x.ncomp(); i++) {
    cudaLaunch(y.grid().ncells(), k_addConstant, y.cu(), x.cu(), value, i);
  }
  return y;
}

__global__ void k_normalize(CuField dst, const CuField src) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!dst.cellInGeometry(idx))
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

Field operator*(real3 a, const Field& x) {
  Field y(x.system(), x.ncomp());
  real3 a0 = real3{0, 0, 0};
  add(y, a0, x, a, x);
  return y;
}

__global__ void k_multiplyFields(CuField y,
                            const CuField a,
                            const CuField x) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!y.cellInGeometry(idx))
    return;
  for (int c = 0; c < y.ncomp; c++) {
    int c_a = (a.ncomp == y.ncomp) ? c : 0;  // follow comp or be scalar
    y.setValueInCell(idx, c, a.valueAt(idx, c_a) * x.valueAt(idx, c));
  }
}

inline void multiply(Field& y, const Field& a, const Field& x) {
  if (x.system() != y.system() || a.system() != y.system()) {
    throw std::invalid_argument(
        "Fields can not be added together because they belong to different "
        "systems.");
  }
  if (x.ncomp() != y.ncomp()) {
    throw std::invalid_argument(
        "Fields can not be added because they do not have the same number of "
        "components.");
  }
  if (!(a.ncomp() == 1 || a.ncomp() == x.ncomp())) {
    throw std::invalid_argument(
        "Weights need to be scalar (1 component) or have the same number of "
        "components as the fields."
    );
  }
  int ncells = y.grid().ncells();
  cudaLaunch(ncells, k_multiplyFields, y.cu(), a.cu(), x.cu());
}

Field multiply(const Field& a, const Field& x) {
  Field y(x.system(), x.ncomp());
  multiply(y, a, x);
  return y;
}

Field operator*(const Field& a, const Field& x) {
  return multiply(a, x);
}

// --------------------------------------------------
// fieldGetRGB

const float pi = 3.1415926535897931f;

/// Transform 3D vector with norm<=1 to its RGB representation
__device__ real3 getRGB(real3 vec) {
  // This function uses float arithmatic, as there is no need for
  // double precision colors.

  // HSL
  float H = atan2f(vec.y, vec.x);
  float S = norm(vec);
  float L = 0.5f + 0.5f * vec.z;

  // HSL to RGB
  float Hp = 3.f * H/pi;
  if (Hp < 0.f) {Hp += 6.f;}  // in [0, 6)
  else if (Hp >= 6.f) {Hp -= 6.f;}
  float C = (L<=0.5f) ? 2.f*L*S : 2.f*(1.f-L)*S;
  float X = C * (1.f - fabs(fmodf(Hp, 2.f) - 1.f));
  float m = L - C / 2.f;

  float R = m, G = m, B = m;
  if (Hp < 1.f) {
    R += C;
    G += X;
  } else if (Hp < 2.f) {
    R += X;
    G += C;
  } else if (Hp < 3.f) {
    G += C;
    B += X;
  } else if (Hp < 4.f) {
    G += X;
    B += C;
  } else if (Hp < 5.f) {
    R += X;
    B += C;
  } else {  // Hp < 6
    R += C;
    B += X;
  }

  // clip RGB values to be in [0,1]
  R = fminf(fmaxf(R, 0.f), 1.f);
  G = fminf(fmaxf(G, 0.f), 1.f);
  B = fminf(fmaxf(B, 0.f), 1.f);

  return real3{R, G, B};  // convert to real3 for Field
}

/// Map 3D vector field (with norm<=1) to RGB
__global__ void k_fieldGetRGB(CuField dst, const CuField src) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!dst.cellInGeometry(idx)) {
    if (dst.cellInGrid(idx))
    // not in geometry, so make grey instead
      dst.setVectorInCell(idx, real3{0.5, 0.5, 0.5});
  } else {
    dst.setVectorInCell(idx, getRGB(src.vectorAt(idx)));
  }
}

Field fieldGetRGB(const Field& src) {
  if (src.ncomp() == 3) {  // 3D
    Field dst =  (1./maxVecNorm(src)) * src;  // rescale to make maximum norm 1
    cudaLaunch(dst.grid().ncells(), k_fieldGetRGB, dst.cu(), dst.cu());  // src is dst
    return dst;
  } else {
    throw std::invalid_argument(
            "getRGB can only operate on vector fields with 3 components.");
  }
}
