#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "cudaerror.hpp"
#include "cudalaunch.hpp"
#include "cudastream.hpp"
#include "field.hpp"
#include "gpubuffer.hpp"
#include "magnet.hpp"
#include "parameter.hpp"
#include "rigidbodymodes4.hpp"
#include "rigidbodymodes_common.hpp"
#include "rigidbodymodes_common.cuh"
#include "system.hpp"


// Horn's method for quaternion alignment, used to perform a full rigid body rotation/translation removal when the rotation is large
// see: Horn, JOSA A 4, 629 (1987)
//
// 1. Accumulate C = Σ_i w_i*(p_i (x) u_i^T), the cross-covariance between reference position (centered at COM0) and displacement.
// 2. H = S0 + C, the full cross-covariance between reference and current position (S0 = Σ_i w_i*(p_i (x) p_i^T)).
// 3. Build a symmetric 4x4 matrix N, from H. N comes from: minimizing the least-squares fit error
//    Σ_i w_i*|current_i - R*reference_i|^2 over rotations R reduces to maximizing trace(R^T * H).
//    Writing R in terms of a unit quaternion q and expanding trace(R^T*H), terms are quadratic in
//    q's components, so trace(R^T*H) = q^T*N*q. N formed by collecting the
//    coefficient of each quadratic monomial (q0^2, q0*q1, q1*q2, ...) using row/col order (q0,q1,q2,q3), with
//    tr = Hxx+Hyy+Hzz and Dx=Hyz-Hzy, Dy=Hzx-Hxz, Dz=Hxy-Hyx:
//    N = [ tr           Dx           Dy           Dz        ]
//        [ Dx        2Hxx-tr        Hxy+Hyx      Hzx+Hxz    ]
//        [ Dy        Hxy+Hyx       2Hyy-tr       Hyz+Hzy    ]
//        [ Dz        Hzx+Hxz       Hyz+Hzy      2Hzz-tr     ]
// 4. The unit quaternion maximizing a quadratic form q^T*N*q is exactly the eigenvector of N's largest eigenvalue (Jacobi eigensolve).
// 5. Convert that quaternion to a rotation matrix R.
// 6. Recover translation T from matching centroids: T = com - R*com0, where com = com0 + mean(u) is the current centroid.
// 7. Subtract the rigid motion (R*r0 + T - r0) from u at every cell, where R is a full arbitrary rotation matrix

namespace {

// reference shape covariance S0 = Σ w_i * (r0_i - COM0) * (r0_i - COM0)^T
// unweighted: w_i=1 for every in-geometry cell instead of rho. 7th field is ncellsInGeometry.
// two-stage warp-shuffle -> double-combine reduction
struct S0Accum  { real xx, yy, zz, xy, xz, yz, count; };
struct S0AccumD { double xx, yy, zz, xy, xz, yz, count; };

// level 1
__global__ void k_S0SumsPartial(S0Accum* blockPartials, CuSystem system, CuParameter rho,
                             real3 com0, bool unweighted) {
  int ncells = system.grid.ncells();
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  real acc[7] = {0,0,0,0,0,0,0};
  real comp[7] = {0,0,0,0,0,0,0};

  for (int i = gid; i < ncells; i += stride) {
    if (!system.inGeometry(i)) {
      continue;
    }
    real w;
    if (unweighted) {
      w = real(1.0);
    } else {
      w = rho.valueAt(i);
    }
    real3 p = cellPositionDevice(system, i) - com0;
    kahanAdd(acc[0], comp[0], w * p.x * p.x);
    kahanAdd(acc[1], comp[1], w * p.y * p.y);
    kahanAdd(acc[2], comp[2], w * p.z * p.z);
    kahanAdd(acc[3], comp[3], w * p.x * p.y);
    kahanAdd(acc[4], comp[4], w * p.x * p.z);
    kahanAdd(acc[5], comp[5], w * p.y * p.z);
    kahanAdd(acc[6], comp[6], real(1.0));
  }
  blockReduceSum<real, 7>(acc);
  if (threadIdx.x == 0){
    blockPartials[blockIdx.x] = S0Accum{acc[0],acc[1],acc[2],acc[3],acc[4],acc[5],acc[6]};
  }
}

//level 2
__global__ void k_S0CombineDouble(S0AccumD* out, const S0Accum* blockPartials, int numPartials) {
  double acc[7] = {0,0,0,0,0,0,0};
  for (int i = threadIdx.x; i < numPartials; i += blockDim.x) {
    S0Accum p = blockPartials[i];
    acc[0]+=p.xx; acc[1]+=p.yy; acc[2]+=p.zz; acc[3]+=p.xy; acc[4]+=p.xz; acc[5]+=p.yz; acc[6]+=p.count;
  }
  blockReduceSum<double, 7>(acc);
  if (threadIdx.x == 0){
    *out = S0AccumD{acc[0],acc[1],acc[2],acc[3],acc[4],acc[5],acc[6]};
  }
}

// Host launcher
S0AccumD computeS0Sums(const CuSystem& cusys, const CuParameter& rho, real3 com0,
                       int ncells, bool unweighted) {
  int numBlocks = numReductionBlocks(ncells, k_S0SumsPartial);
  GpuBuffer<S0Accum> d_partials(numBlocks);
  GpuBuffer<S0AccumD> d_accum(1);

  k_S0SumsPartial<<<numBlocks, BLOCKDIM, 0, getCudaStream()>>>(d_partials.get(), cusys, rho, com0, unweighted);
  checkCudaError(cudaPeekAtLastError());
  k_S0CombineDouble<<<1, BLOCKDIM, 0, getCudaStream()>>>(d_accum.get(), d_partials.get(), numBlocks);
  checkCudaError(cudaPeekAtLastError());

  S0AccumD result;
  checkCudaError(cudaMemcpyAsync(&result, d_accum.get(), sizeof(S0AccumD), cudaMemcpyDeviceToHost, getCudaStream()));
  checkCudaError(cudaStreamSynchronize(getCudaStream()));
  return result;
}

//C = Σ_i w_i*(r0_i-com0)*u_i^T and weighted displacement sum Σ_i w_i*u_i
struct CMatAccum  { real C0,C1,C2,C3,C4,C5,C6,C7,C8, ux,uy,uz; };
struct CMatAccumD { double C0,C1,C2,C3,C4,C5,C6,C7,C8, ux,uy,uz; };

__global__ void k_CMatPartials(CMatAccum* blockPartials, CuField u, CuSystem system,
                              CuParameter rho, real3 com0, bool unweighted) {
  int ncells = system.grid.ncells();
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  real acc[12] = {0,0,0,0,0,0,0,0,0,0,0,0};
  real comp[12] = {0,0,0,0,0,0,0,0,0,0,0,0};

  for (int i = gid; i < ncells; i += stride) {
    if (!u.cellInGeometry(i)) {
      continue;
    }
    real w;
    if (unweighted) {
      w = real(1.0);
    } else {
      w = rho.valueAt(i);
    }
    real3 p = cellPositionDevice(system, i) - com0;
    real3 uv = u.vectorAt(i);

    kahanAdd(acc[0], comp[0], w*p.x*uv.x); kahanAdd(acc[1], comp[1], w*p.x*uv.y); kahanAdd(acc[2], comp[2], w*p.x*uv.z);
    kahanAdd(acc[3], comp[3], w*p.y*uv.x); kahanAdd(acc[4], comp[4], w*p.y*uv.y); kahanAdd(acc[5], comp[5], w*p.y*uv.z);
    kahanAdd(acc[6], comp[6], w*p.z*uv.x); kahanAdd(acc[7], comp[7], w*p.z*uv.y); kahanAdd(acc[8], comp[8], w*p.z*uv.z);
    kahanAdd(acc[9],  comp[9],  w*uv.x);
    kahanAdd(acc[10], comp[10], w*uv.y);
    kahanAdd(acc[11], comp[11], w*uv.z);
  }
  blockReduceSum<real, 12>(acc);
  if (threadIdx.x == 0){
    blockPartials[blockIdx.x] = CMatAccum{acc[0],acc[1],acc[2],acc[3],acc[4],acc[5],
                                          acc[6],acc[7],acc[8],acc[9],acc[10],acc[11]};
  }
}

__global__ void k_CmatCombineDouble(CMatAccumD* out, const CMatAccum* blockPartials, int numPartials) {
  double acc[12] = {0,0,0,0,0,0,0,0,0,0,0,0};
  for (int i = threadIdx.x; i < numPartials; i += blockDim.x) {
    CMatAccum p = blockPartials[i];
    acc[0]+=p.C0; acc[1]+=p.C1; acc[2]+=p.C2; acc[3]+=p.C3; acc[4]+=p.C4; acc[5]+=p.C5;
    acc[6]+=p.C6; acc[7]+=p.C7; acc[8]+=p.C8; acc[9]+=p.ux; acc[10]+=p.uy; acc[11]+=p.uz;
  }
  blockReduceSum<double, 12>(acc);
  if (threadIdx.x == 0){
    *out = CMatAccumD{acc[0],acc[1],acc[2],acc[3],acc[4],acc[5],
                      acc[6],acc[7],acc[8],acc[9],acc[10],acc[11]};
  }
}

CMatAccumD computeQSums(const CuSystem& cusys, CuField u, const CuParameter& rho,
                          real3 com0, bool unweighted, int ncells) {
  int numBlocks = numReductionBlocks(ncells, k_CMatPartials);
  GpuBuffer<CMatAccum> d_partials(numBlocks);
  GpuBuffer<CMatAccumD> d_accum(1);

  k_CMatPartials<<<numBlocks, BLOCKDIM, 0, getCudaStream()>>>(d_partials.get(), u, cusys, rho, com0, unweighted);
  checkCudaError(cudaPeekAtLastError());
  k_CmatCombineDouble<<<1, BLOCKDIM, 0, getCudaStream()>>>(d_accum.get(), d_partials.get(), numBlocks);
  checkCudaError(cudaPeekAtLastError());

  CMatAccumD result;
  checkCudaError(cudaMemcpyAsync(&result, d_accum.get(), sizeof(CMatAccumD), cudaMemcpyDeviceToHost, getCudaStream()));
  checkCudaError(cudaStreamSynchronize(getCudaStream()));
  return result;
}

// host: Horn's N matrix
void buildHornMatrix(const double H[3][3], double N[4][4]) {
  double Sxx = H[0][0], Sxy = H[0][1], Sxz = H[0][2];
  double Syx = H[1][0], Syy = H[1][1], Syz = H[1][2];
  double Szx = H[2][0], Szy = H[2][1], Szz = H[2][2];

  N[0][0] = Sxx + Syy + Szz;
  N[0][1] = N[1][0] = Syz - Szy;
  N[0][2] = N[2][0] = Szx - Sxz;
  N[0][3] = N[3][0] = Sxy - Syx;

  N[1][1] = Sxx - Syy - Szz;
  N[1][2] = N[2][1] = Sxy + Syx;
  N[1][3] = N[3][1] = Szx + Sxz;

  N[2][2] = -Sxx + Syy - Szz;
  N[2][3] = N[3][2] = Syz + Szy;

  N[3][3] = -Sxx - Syy + Szz;
}

inline void rotate(double a[4][4], double s, double tau,
                    int i1, int j1, int i2, int j2) {
  double g = a[i1][j1];
  double h = a[i2][j2];
  a[i1][j1] = g - s * (h + g * tau);
  a[i2][j2] = h + s * (g - h * tau);
}

// Jacobi eigenvalue/vector solver for 4x4 symmetric matrix
//see https://en.wikipedia.org/wiki/Jacobi_eigenvalue_algorithm or Numerical Recipes, 3rd ed, sec 11.1
void jacobiEigenSymmetric4x4(const double A[4][4], double V[4][4], double eigval[4]) {
  double a[4][4];
  double d[4], b[4], z[4];
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      a[i][j] = A[i][j];
      if (i == j) {
        V[i][j] = 1.0;
      } else {
        V[i][j] = 0.0;
      }
    }
    b[i] = d[i] = A[i][i];  
    z[i] = 0.0;             
  }

  const int pq[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
  const double EPS = std::numeric_limits<double>::epsilon();
  //TODO: test smaller maxsweeps. Normally 6-10, not sure for degenerate cases. low cost.
  const int maxSweeps = 50;

  for (int sweep = 0; sweep < maxSweeps; sweep++) {
    double sm = 0.0;
    for (int k = 0; k < 6; k++) {
      sm += std::abs(a[pq[k][0]][pq[k][1]]);
    }
    if (sm == 0.0) {
      break;
    }

    double tresh = (sweep < 3) ? 0.2 * sm / 16.0 : 0.0;

    for (int pair = 0; pair < 6; pair++) {
      int p = pq[pair][0], q = pq[pair][1];
      double g = 100.0 * std::abs(a[p][q]);

      if (sweep > 3 && g <= EPS * std::abs(d[p]) && g <= EPS * std::abs(d[q])) {
        a[p][q] = 0.0;
        continue;
      }
      if (std::abs(a[p][q]) <= tresh) {
        continue;
      }

      double h = d[q] - d[p];
      double t;
      if (g <= EPS * std::abs(h)) {
        t = a[p][q] / h;
      } else {
        double theta = 0.5 * h / a[p][q];
        t = 1.0 / (std::abs(theta) + std::sqrt(1.0 + theta * theta));
        if (theta < 0.0) {
          t = -t;
        }
      }
      double c = 1.0 / std::sqrt(1.0 + t * t);
      double s = t * c;
      double tau = s / (1.0 + c);

      h = t * a[p][q];
      z[p] -= h;
      z[q] += h;
      d[p] -= h;
      d[q] += h;
      a[p][q] = 0.0;

      for (int j = 0; j < p; j++) {
        rotate(a, s, tau, j, p, j, q);
      }
      for (int j = p + 1; j < q; j++) {
        rotate(a, s, tau, p, j, j, q);
      }
      for (int j = q + 1; j < 4; j++) {
        rotate(a, s, tau, p, j, q, j);
      }
      for (int j = 0; j < 4; j++) {
        rotate(V, s, tau, j, p, j, q);
      }
    }

    for (int p = 0; p < 4; p++) {
      b[p] += z[p];
      d[p] = b[p];
      z[p] = 0.0;
    }
  }

  for (int i = 0; i < 4; i++) {
    eigval[i] = d[i];
  }
}

void QtoRMat(const double q[4], double R[3][3]) {
  double q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3];
  R[0][0] = 1 - 2*(q2*q2 + q3*q3); R[0][1] = 2*(q1*q2 - q0*q3);     R[0][2] = 2*(q1*q3 + q0*q2);
  R[1][0] = 2*(q1*q2 + q0*q3);     R[1][1] = 1 - 2*(q1*q1 + q3*q3); R[1][2] = 2*(q2*q3 - q0*q1);
  R[2][0] = 2*(q1*q3 - q0*q2);     R[2][1] = 2*(q2*q3 + q0*q1);     R[2][2] = 1 - 2*(q1*q1 + q2*q2);
}

__host__ __device__ inline double3 mulMat3Vec3(const double R[3][3], double3 v) {
  return double3{R[0][0]*v.x + R[0][1]*v.y + R[0][2]*v.z,
                 R[1][0]*v.x + R[1][1]*v.y + R[1][2]*v.z,
                 R[2][0]*v.x + R[2][1]*v.y + R[2][2]*v.z};
}

// u -= T + (R*r0 - r0)
__global__ void k_subQModes(CuField f, Mat3x3 R, real3 T) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!f.cellInGrid(idx) || !f.cellInGeometry(idx)) return;

  real3 r0 = cellPositionDevice(f.system, idx);
  double3 Rr0D = mulMat3Vec3(R.m, toDouble3(r0));
  real3 correction = toReal3(Rr0D) + T - r0;

  real3 v = f.vectorAt(idx);
  f.setVectorInCell(idx, v - correction);
}

}  // namespace ends


// Unweighted computes reference geometric centroid, COM0 and shape covariance S0
// rho-weighted currently N/A, preserved in case of future use (dynamics?)
RigidBodyGeometry4 computeRigidBodyGeometry4(const Magnet* magnet) {
  std::shared_ptr<const System> system = magnet->system();
  int ncells = system->grid().ncells();
  CuSystem cusys = system->cu();
  CuParameter rho = magnet->rho.cu();

  ComResult comWeighted   = computeCom(cusys, rho, ncells, false);
  ComResult comUnweighted = computeCom(cusys, rho, ncells, true);

  RigidBodyGeometry4 geom;
  geom.com0            = comWeighted.com;
  geom.com0Unweighted  = comUnweighted.com;
  geom.totalRho        = comWeighted.weightSum;
  geom.ncellsInGeometry = comUnweighted.weightSum;

  S0AccumD s0  = computeS0Sums(cusys, rho, geom.com0, ncells, false);
  S0AccumD s0u = computeS0Sums(cusys, rho, geom.com0Unweighted, ncells, true);

  geom.S0[0][0]=s0.xx; geom.S0[0][1]=s0.xy; geom.S0[0][2]=s0.xz;
  geom.S0[1][0]=s0.xy; geom.S0[1][1]=s0.yy; geom.S0[1][2]=s0.yz;
  geom.S0[2][0]=s0.xz; geom.S0[2][1]=s0.yz; geom.S0[2][2]=s0.zz;

  geom.S0Unweighted[0][0]=s0u.xx; geom.S0Unweighted[0][1]=s0u.xy; geom.S0Unweighted[0][2]=s0u.xz;
  geom.S0Unweighted[1][0]=s0u.xy; geom.S0Unweighted[1][1]=s0u.yy; geom.S0Unweighted[1][2]=s0u.yz;
  geom.S0Unweighted[2][0]=s0u.xz; geom.S0Unweighted[2][1]=s0u.yz; geom.S0Unweighted[2][2]=s0u.zz;

  return geom;
}


// Align the current u to the reference geometry
QuatAlignResult computeQuaternionAlignment(const Field& u, const RigidBodyGeometry4& geom,
                                           const Magnet* magnet, bool unweighted) {
  CuParameter rho = magnet->rho.cu();
  CuSystem cusys = magnet->system()->cu();
  int ncells = cusys.grid.ncells();
  real3 com0;
  const double (*S0ToUse)[3];
  double denom;
  if (unweighted) {
    com0 = geom.com0Unweighted;
    S0ToUse = geom.S0Unweighted;
    denom = double(geom.ncellsInGeometry);
  } else {
    com0 = geom.com0;
    S0ToUse = geom.S0;
    denom = double(geom.totalRho);
  }

  CMatAccumD qs = computeQSums(cusys, u.cu(), rho, com0, unweighted, ncells);

  double C[3][3] = {{qs.C0,qs.C1,qs.C2},{qs.C3,qs.C4,qs.C5},{qs.C6,qs.C7,qs.C8}};
  double3 uSum = {qs.ux, qs.uy, qs.uz};

  double H[3][3];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      H[i][j] = S0ToUse[i][j] + C[i][j];
    }
  }

  double N[4][4];
  buildHornMatrix(H, N);
  double eigvecs[4][4], eigvals[4];
  jacobiEigenSymmetric4x4(N, eigvecs, eigvals);

  int best = 0;
  for (int i = 1; i < 4; i++) {
    if (eigvals[i] > eigvals[best]) {
      best = i;
    }
  }
  int second = -1;
  for (int i = 0; i < 4; i++) {
    if (i != best && (second < 0 || eigvals[i] > eigvals[second])) {
      second = i;
    }
  }

  QuatAlignResult result;
  for (int k = 0; k < 4; k++) {
    result.q[k] = eigvecs[k][best];
  }
  result.eigenGap = eigvals[best] - eigvals[second];

  double kEigenGapWarnRelTol = 1e-6;
  double relGap = result.eigenGap / std::max(std::abs(eigvals[best]), 1e-30);
  if (relGap < kEigenGapWarnRelTol) {
    std::cerr << "warning: computeQuaternionAlignment: nearly degenerate (relative gap of largest eigenvalues = "
              << relGap << "). Rotation axis poorly determined by this geometry/displacement." << std::endl;
  }
  QtoRMat(result.q, result.R);

  real3 uMean = {real(uSum.x/denom), real(uSum.y/denom), real(uSum.z/denom)};
  result.com = com0 + uMean;                              

  double3 Rcom0 = mulMat3Vec3(result.R, toDouble3(com0));   
  double3 comD  = toDouble3(result.com);
  result.T = toReal3(double3{comD.x - Rcom0.x, comD.y - Rcom0.y, comD.z - Rcom0.z});

  return result;
}

void removeRigidBodyModesQuaternion(Field& u, const RigidBodyGeometry4& geom,
                                    const Magnet* magnet, bool unweighted) {
  QuatAlignResult result = computeQuaternionAlignment(u, geom, magnet, unweighted);
  int ncells = u.system()->grid().ncells();

  Mat3x3 R;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      R.m[i][j] = result.R[i][j];
    }
  }

  cudaLaunch(ncells, k_subQModes, u.cu(), R, result.T);
}

double quaternionRotationAngle(const QuatAlignResult& result) {
  double q0clamped = std::max(-1.0, std::min(1.0, std::abs(result.q[0])));
  return 2.0 * std::acos(q0clamped);
}
