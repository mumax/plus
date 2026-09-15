#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <limits>

#include "cudaerror.hpp"
#include "cudalaunch.hpp"
#include "cudastream.hpp"
#include "field.hpp"
#include "gpubuffer.hpp"
#include "magnet.hpp"
#include "parameter.hpp"
#include "rigidbodymodesQ.hpp"
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
//    N = [ Hxx+Hyy+Hzz    Hyz-Hzy        Hzx-Hxz         Hxy-Hyx]
//        [ Hyz-Hzy        Hxx-Hyy-Hzz    Hxy+Hyx         Hzx+Hxz]
//        [ Hzx-Hxz        Hxy+Hyx        -Hxx+Hyy-Hzz    Hyz+Hzy]
//        [ Hxy-Hyx        Hzx+Hxz        Hyz+Hzy         -Hxx-Hyy+Hzz]
// 4. The unit quaternion maximizing a quadratic form q^T*N*q is exactly the eigenvector of N's largest eigenvalue (Jacobi eigensolve).
// 5. Convert that quaternion to a rotation matrix R.
// 6. Recover translation T from matching centroids: T = com - R*com0, where com = com0 + mean(u) is the current centroid.
// 7. Subtract the rigid motion (R*r0 + T - r0) from u at every cell, where R is a full arbitrary rotation matrix

namespace {

// reference shape covariance S0 = Σ_i w_i * (r0_i - COM0) * (r0_i - COM0)^T
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

//level 1
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
    real3 r = cellPositionDevice(system, i) - com0;
    real3 uvec = u.vectorAt(i);

    kahanAdd(acc[0], comp[0], w*r.x*uvec.x); kahanAdd(acc[1], comp[1], w*r.x*uvec.y); kahanAdd(acc[2], comp[2], w*r.x*uvec.z);
    kahanAdd(acc[3], comp[3], w*r.y*uvec.x); kahanAdd(acc[4], comp[4], w*r.y*uvec.y); kahanAdd(acc[5], comp[5], w*r.y*uvec.z);
    kahanAdd(acc[6], comp[6], w*r.z*uvec.x); kahanAdd(acc[7], comp[7], w*r.z*uvec.y); kahanAdd(acc[8], comp[8], w*r.z*uvec.z);
    kahanAdd(acc[9],  comp[9],  w*uvec.x);
    kahanAdd(acc[10], comp[10], w*uvec.y);
    kahanAdd(acc[11], comp[11], w*uvec.z);
  }
  blockReduceSum<real, 12>(acc);
  if (threadIdx.x == 0){
    blockPartials[blockIdx.x] = CMatAccum{acc[0],acc[1],acc[2],acc[3],acc[4],acc[5],
                                          acc[6],acc[7],acc[8],acc[9],acc[10],acc[11]};
  }
}

//level 2
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

// Host launcher
CMatAccumD computeCMatSums(const CuSystem& cusys, CuField u, const CuParameter& rho,
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
  N[0][0] = H[0][0] + H[1][1] + H[2][2];
  N[0][1] = N[1][0] = H[1][2] - H[2][1];
  N[0][2] = N[2][0] = H[2][0] - H[0][2];
  N[0][3] = N[3][0] = H[0][1] - H[1][0];

  N[1][1] = H[0][0] - H[1][1] - H[2][2];
  N[1][2] = N[2][1] = H[0][1] + H[1][0];
  N[1][3] = N[3][1] = H[2][0] + H[0][2];

  N[2][2] = -H[0][0] + H[1][1] - H[2][2];
  N[2][3] = N[3][2] = H[1][2] + H[2][1];

  N[3][3] = -H[0][0] - H[1][1] + H[2][2];
}

inline void rotate(double a[4][4], double Jsin, double Jtau,
                    int i1, int j1, int i2, int j2) {
  double a11_orig = a[i1][j1];
  double a22_orig = a[i2][j2];
  a[i1][j1] = a11_orig - Jsin * (a22_orig + a11_orig * Jtau);
  a[i2][j2] = a22_orig + Jsin * (a11_orig - a22_orig * Jtau);
}

// host Jacobi eigenvalue/vector solver for 4x4 symmetric matrix
// see https://en.wikipedia.org/wiki/Jacobi_eigenvalue_algorithm or Numerical Recipes, 3rd ed, sec 11.1
// Horn uses exact solve of quartic characteristic equation (e.g. Ferrari method), but that fails for symmetric/near-degenerate states
void jacobiEigenSymmetric4x4(double N[4][4], double eigvecs[4][4], double eigvals[4]) {
  double eigvalSweepBase[4], eigvalcorr[4];
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      if (i == j) {
        eigvecs[i][j] = 1.0;
      } else {
        eigvecs[i][j] = 0.0;
      }
    }
    eigvalSweepBase[i] = eigvals[i] = N[i][i];  
    eigvalcorr[i] = 0.0;             
  }

  const int pq[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
  const double EPS = std::numeric_limits<double>::epsilon();
  //TODO: test smaller maxsweeps. Normally 6-10, not sure for degenerate cases. low cost.
  const int maxSweeps = 50;

  for (int sweep = 0; sweep < maxSweeps; sweep++) {
    double offDiagSum = 0.0;
    for (int k = 0; k < 6; k++) {
      offDiagSum += std::abs(N[pq[k][0]][pq[k][1]]);
    }
    // NR uses underflow to 0, pick something more robust
    double trace = std::abs(N[0][0]) + std::abs(N[1][1]) + std::abs(N[2][2]) + std::abs(N[3][3]);
    if (offDiagSum <= EPS * trace) {
      break;
    }

    // 4x4=16
    double thresh;
    if (sweep < 3) {
      thresh = 0.2 * offDiagSum / 16.0;
    } else {
      thresh = 0.0;
    }

    for (int pairIdx = 0; pairIdx < 6; pairIdx++) {
      int p = pq[pairIdx][0], q = pq[pairIdx][1];
      double precisionBuffer = 100.0 * std::abs(N[p][q]);

      if (sweep > 3 && precisionBuffer <= EPS * std::abs(eigvals[p]) && precisionBuffer <= EPS * std::abs(eigvals[q])) {
        N[p][q] = 0.0;
        continue;
      }
      if (std::abs(N[p][q]) <= thresh) {
        continue;
      }

      double diagGap = eigvals[q] - eigvals[p];
      double Jtan;
      if (precisionBuffer <= EPS * std::abs(diagGap)) {
        Jtan = N[p][q] / diagGap;
      } else {
        double theta = 0.5 * diagGap / N[p][q];
        Jtan = 1.0 / (std::abs(theta) + std::sqrt(1.0 + theta * theta));
        if (theta < 0.0) {
          Jtan = -Jtan;
        }
      }
      double Jcos = 1.0 / std::sqrt(1.0 + Jtan * Jtan);
      double Jsin = Jtan * Jcos;
      double Jtau = Jsin / (1.0 + Jcos);

      double correction = Jtan * N[p][q];
      eigvalcorr[p] -= correction;
      eigvalcorr[q] += correction;
      eigvals[p] -= correction;
      eigvals[q] += correction;
      N[p][q] = 0.0;

      for (int j = 0; j < p; j++) {
        rotate(N, Jsin, Jtau, j, p, j, q);
      }
      for (int j = p + 1; j < q; j++) {
        rotate(N, Jsin, Jtau, p, j, j, q);
      }
      for (int j = q + 1; j < 4; j++) {
        rotate(N, Jsin, Jtau, p, j, q, j);
      }
      for (int j = 0; j < 4; j++) {
        rotate(eigvecs, Jsin, Jtau, j, p, j, q);
      }
    }

    for (int p = 0; p < 4; p++) {
      eigvalSweepBase[p] += eigvalcorr[p];
      eigvals[p] = eigvalSweepBase[p];
      eigvalcorr[p] = 0.0;
    }
  }

}

void QtoRMat(const double q[4], double R[3][3]) {
  R[0][0] = 1 - 2*(q[2]*q[2] + q[3]*q[3]); R[0][1] = 2*(q[1]*q[2] - q[0]*q[3]);     R[0][2] = 2*(q[1]*q[3] + q[0]*q[2]);
  R[1][0] = 2*(q[1]*q[2] + q[0]*q[3]);     R[1][1] = 1 - 2*(q[1]*q[1] + q[3]*q[3]); R[1][2] = 2*(q[2]*q[3] - q[0]*q[1]);
  R[2][0] = 2*(q[1]*q[3] - q[0]*q[2]);     R[2][1] = 2*(q[2]*q[3] + q[0]*q[1]);     R[2][2] = 1 - 2*(q[1]*q[1] + q[2]*q[2]);
}

__host__ __device__ inline double3 mulMat3x3Vec3(const double R[3][3], double3 v) {
  return double3{R[0][0]*v.x + R[0][1]*v.y + R[0][2]*v.z,
                 R[1][0]*v.x + R[1][1]*v.y + R[1][2]*v.z,
                 R[2][0]*v.x + R[2][1]*v.y + R[2][2]*v.z};
}

// u -= T + (R*r0 - r0)
__global__ void k_subQModes(CuField f, Mat3x3 R, real3 T) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!f.cellInGrid(idx) || !f.cellInGeometry(idx)) return;

  real3 r0 = cellPositionDevice(f.system, idx);
  double3 Rr0D = mulMat3x3Vec3(R.m, toDouble3(r0));
  real3 correction = toReal3(Rr0D) + T - r0;

  real3 v = f.vectorAt(idx);
  f.setVectorInCell(idx, v - correction);
}

}  // namespace ends


// Unweighted computes reference geometric centroid, COM0 and shape covariance S0
// rho-weighted currently N/A, preserved in case of future use (dynamics?)
RigidBodyGeomQ computeRigidBodyGeometryQ(const Magnet* magnet) {
  std::shared_ptr<const System> system = magnet->system();
  int ncells = system->grid().ncells();
  CuSystem cusys = system->cu();
  CuParameter rho = magnet->rho.cu();

  ComResult comWeighted   = computeCom(cusys, rho, ncells, false);
  ComResult comUnweighted = computeCom(cusys, rho, ncells, true);

  RigidBodyGeomQ geom;
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
QuatAlignResult computeQuaternionAlignment(const Field& u, const RigidBodyGeomQ& geom,
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

  CMatAccumD cMatSums = computeCMatSums(cusys, u.cu(), rho, com0, unweighted, ncells);

  double C[3][3] = {{cMatSums.C0,cMatSums.C1,cMatSums.C2},{cMatSums.C3,cMatSums.C4,cMatSums.C5},{cMatSums.C6,cMatSums.C7,cMatSums.C8}};
  double3 uSum = {cMatSums.ux, cMatSums.uy, cMatSums.uz};

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

  double3 Rcom0 = mulMat3x3Vec3(result.R, toDouble3(com0));   
  double3 comD  = toDouble3(result.com);
  result.T = toReal3(double3{comD.x - Rcom0.x, comD.y - Rcom0.y, comD.z - Rcom0.z});

  return result;
}

void removeRigidBodyModesQ(Field& u, const RigidBodyGeomQ& geom,
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
