#include <stdexcept>

#include "cudaerror.hpp"
#include "cudalaunch.hpp"
#include "cudastream.hpp"
#include "field.hpp"
#include "gpubuffer.hpp"
#include "magnet.hpp"
#include "parameter.hpp"
#include "reduce.hpp"
#include "rigidbodymodes.hpp"
#include "rigidbodymodes_common.hpp"
#include "rigidbodymodes_common.cuh"
#include "system.hpp"

namespace {


// pass 2: inertia tensor about a given COM
// same two-stage warp-shuffle -> double-combine shape as pass 1.

struct InertiaAccum { real xx, yy, zz, xy, xz, yz; };
struct InertiaAccumD { double xx, yy, zz, xy, xz, yz; };

// level 1
__global__ void k_inertiaSumsPartial(InertiaAccum* blockPartials,
                                     CuSystem system, CuParameter rho,
                                     real3 com, bool unweighted) {
  int ncells = system.grid.ncells();
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  real accum6[6] = {0, 0, 0, 0, 0, 0};  // xx, yy, zz, xy, xz, yz
  real comp6[6] = {0, 0, 0, 0, 0, 0};   // Kahan compensation terms

  for (int i = gid; i < ncells; i += stride) {
    if (!system.inGeometry(i)){
      continue;
    }
    real w;
    if (unweighted) {
      w = real(1.0);
    } else {
      w = rho.valueAt(i);
    }
    real3 r = cellPositionDevice(system, i) - com;
    kahanAdd(accum6[0], comp6[0], w * (r.y * r.y + r.z * r.z));
    kahanAdd(accum6[1], comp6[1], w * (r.x * r.x + r.z * r.z));
    kahanAdd(accum6[2], comp6[2], w * (r.x * r.x + r.y * r.y));
    kahanAdd(accum6[3], comp6[3], w * (-r.x * r.y));
    kahanAdd(accum6[4], comp6[4], w * (-r.x * r.z));
    kahanAdd(accum6[5], comp6[5], w * (-r.y * r.z));
  }

  blockReduceSum<real, 6>(accum6);

  if (threadIdx.x == 0){
    blockPartials[blockIdx.x] = InertiaAccum{accum6[0], accum6[1], accum6[2], accum6[3], accum6[4], accum6[5]};
  }
}

// level 2: single block, double accumulation
__global__ void k_inertiaSumsCombineDouble(InertiaAccumD* out,
                                           const InertiaAccum* blockPartials,
                                           int numPartials) {
  double accum6_out[6] = {0, 0, 0, 0, 0, 0};

  for (int i = threadIdx.x; i < numPartials; i += blockDim.x) {
    InertiaAccum p = blockPartials[i];
    accum6_out[0] += p.xx; accum6_out[1] += p.yy; accum6_out[2] += p.zz;
    accum6_out[3] += p.xy; accum6_out[4] += p.xz; accum6_out[5] += p.yz;
  }

  blockReduceSum<double, 6>(accum6_out);

  if (threadIdx.x == 0){
    *out = InertiaAccumD{accum6_out[0], accum6_out[1], accum6_out[2], accum6_out[3], accum6_out[4], accum6_out[5]};
  }
}

// Host launcher
InertiaAccumD computeInertiaSums(const CuSystem& cusys, const CuParameter& rho, real3 com,
                                 int ncells, bool unweighted) {
  int numBlocks = numReductionBlocks(ncells, k_inertiaSumsPartial);

  GpuBuffer<InertiaAccum> d_partials(numBlocks);
  GpuBuffer<InertiaAccumD> d_accum(1);

  k_inertiaSumsPartial<<<numBlocks, BLOCKDIM, 0, getCudaStream()>>>(d_partials.get(), cusys, rho, com, unweighted);
  checkCudaError(cudaPeekAtLastError());

  k_inertiaSumsCombineDouble<<<1, BLOCKDIM, 0, getCudaStream()>>>(d_accum.get(), d_partials.get(), numBlocks);
  checkCudaError(cudaPeekAtLastError());

  InertiaAccumD result;
  checkCudaError(cudaMemcpyAsync(&result, d_accum.get(), sizeof(InertiaAccumD),cudaMemcpyDeviceToHost, getCudaStream()));
  checkCudaError(cudaStreamSynchronize(getCudaStream()));
  return result;
}

void invert3x3(const double I[3][3], double out[3][3]) {
  double det =
      I[0][0]*(I[1][1]*I[2][2] - I[1][2]*I[2][1]) -
      I[0][1]*(I[1][0]*I[2][2] - I[1][2]*I[2][0]) +
      I[0][2]*(I[1][0]*I[2][1] - I[1][1]*I[2][0]);

  if (det == 0.0){
    throw std::runtime_error("Rigid-body inertia tensor is singular.");
  }

  double invDet = 1.0 / det;
  out[0][0] = ( I[1][1]*I[2][2] - I[1][2]*I[2][1]) * invDet;
  out[0][1] = (-I[0][1]*I[2][2] + I[0][2]*I[2][1]) * invDet;
  out[0][2] = ( I[0][1]*I[1][2] - I[0][2]*I[1][1]) * invDet;
  out[1][0] = (-I[1][0]*I[2][2] + I[1][2]*I[2][0]) * invDet;
  out[1][1] = ( I[0][0]*I[2][2] - I[0][2]*I[2][0]) * invDet;
  out[1][2] = (-I[0][0]*I[1][2] + I[0][2]*I[1][0]) * invDet;
  out[2][0] = ( I[1][0]*I[2][1] - I[1][1]*I[2][0]) * invDet;
  out[2][1] = (-I[0][0]*I[2][1] + I[0][1]*I[2][0]) * invDet;
  out[2][2] = ( I[0][0]*I[1][1] - I[0][1]*I[1][0]) * invDet;
}


// fused translation/rotation moment reduction, 2 stage reduction again
// level 1: grid-stride load + two-level warp-shuffle tree reduce.
//  Each block writes its own partial to a small global array. 
// level 2: exactly one block. Combines level-1 partials using f64,
//  then the same warp-shuffle tree.

struct TransRotAccum { real moment0_x, moment0_y, moment0_z, moment1_x, moment1_y, moment1_z; };
struct TransRotAccumD { double moment0_x_d, moment0_y_d, moment0_z_d, moment1_x_d, moment1_y_d, moment1_z_d; };

// level 1
__global__ void k_transRotSumsPartial(TransRotAccum* blockPartials, CuField f,
                                      CuParameter rho, real3 com, bool unweighted) {
  int ncells = f.system.grid.ncells();
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  real accum6[6] = {0, 0, 0, 0, 0, 0};  // moment0_x, moment0_y, moment0_z, moment1_x, moment1_y, moment1_z
  real comp6[6] = {0, 0, 0, 0, 0, 0};   // Kahan compensation terms

  for (int i = gid; i < ncells; i += stride) {
    if (!f.cellInGeometry(i)){
       continue;
    }
    real w;
    if (unweighted) {
        w = real(1.0);
    } else {
        w = rho.valueAt(i);
    }
    real3 fvec = f.vectorAt(i);
    kahanAdd(accum6[0], comp6[0], w * fvec.x);
    kahanAdd(accum6[1], comp6[1], w * fvec.y);
    kahanAdd(accum6[2], comp6[2], w * fvec.z);

    real3 r = cellPositionDevice(f.system, i) - com;
    real3 rxf = cross(r, fvec);
    kahanAdd(accum6[3], comp6[3], w * rxf.x);
    kahanAdd(accum6[4], comp6[4], w * rxf.y);
    kahanAdd(accum6[5], comp6[5], w * rxf.z);
  }

  blockReduceSum<real, 6>(accum6);

  if (threadIdx.x == 0){
    blockPartials[blockIdx.x] = TransRotAccum{accum6[0], accum6[1], accum6[2], accum6[3], accum6[4], accum6[5]};
  }
}

// level 2: single block, double accumulation
__global__ void k_transRotSumsCombineDouble(TransRotAccumD* out,
                                            const TransRotAccum* blockPartials,
                                            int numPartials) {
  double accum6_out[6] = {0, 0, 0, 0, 0, 0};

  for (int i = threadIdx.x; i < numPartials; i += blockDim.x) {
    TransRotAccum partials = blockPartials[i];
    accum6_out[0] += partials.moment0_x; accum6_out[1] += partials.moment0_y; accum6_out[2] += partials.moment0_z;
    accum6_out[3] += partials.moment1_x; accum6_out[4] += partials.moment1_y; accum6_out[5] += partials.moment1_z;
  }

  blockReduceSum<double, 6>(accum6_out);

  if (threadIdx.x == 0){
    *out = TransRotAccumD{accum6_out[0], accum6_out[1], accum6_out[2], accum6_out[3], accum6_out[4], accum6_out[5]};
  }
}

// Host launcher
const TransRotAccumD* reduceTransRotSumsDevice(CuField f, CuParameter rho, real3 com,
                                               bool unweighted, int ncells) {
  int numBlocksA = numReductionBlocks(ncells, k_transRotSumsPartial);
  static GpuBuffer<TransRotAccum> d_partials;
  static GpuBuffer<TransRotAccumD> d_accum(1);
  d_partials.allocate(numBlocksA);

  k_transRotSumsPartial<<<numBlocksA, BLOCKDIM, 0, getCudaStream()>>>(d_partials.get(), f, rho, com, unweighted);
  checkCudaError(cudaPeekAtLastError());

  k_transRotSumsCombineDouble<<<1, BLOCKDIM, 0, getCudaStream()>>>(d_accum.get(), d_partials.get(), numBlocksA);
  checkCudaError(cudaPeekAtLastError());

  return d_accum.get();
}


// u -= T + theta x r, or f -= f_trans + tau. least-squares fit
__global__ void k_subtractRigidModesGeneric(CuField f, real3 com,
                                            const TransRotAccumD* accum,
                                            double denom, Mat3x3 Iinv) {
  __shared__ real3 T, theta;

  if (threadIdx.x == 0) {
    TransRotAccumD a = *accum;
    T = {real(a.moment0_x_d / denom), real(a.moment0_y_d / denom), real(a.moment0_z_d / denom)};
    theta = {
        real(Iinv.m[0][0]*a.moment1_x_d + Iinv.m[0][1]*a.moment1_y_d + Iinv.m[0][2]*a.moment1_z_d),
        real(Iinv.m[1][0]*a.moment1_x_d + Iinv.m[1][1]*a.moment1_y_d + Iinv.m[1][2]*a.moment1_z_d),
        real(Iinv.m[2][0]*a.moment1_x_d + Iinv.m[2][1]*a.moment1_y_d + Iinv.m[2][2]*a.moment1_z_d)};
  }
  __syncthreads();

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!f.cellInGrid(idx) || !f.cellInGeometry(idx)) return;

  real3 r = cellPositionDevice(f.system, idx) - com;
  real3 correction = {T.x + (theta.y * r.z - theta.z * r.y),
                      T.y + (theta.z * r.x - theta.x * r.z),
                      T.z + (theta.x * r.y - theta.y * r.x)};

  real3 fvec = f.vectorAt(idx);
  f.setVectorInCell(idx, fvec - correction);
}

}  // namespace ends

// Computes both the rho-weighted and the fully unweighted (geometric)
// versions of com, inertia tensor, and normalization.
//
// if rho-weighted, compute center of mass (COM) COM= (Σrho_i*r_i)/Σrho_i (factor of V cancels off) and moment of Inertia tensor (I)
// about COM I = Σrho_i*(|r_comi|^2*I - r_comi*r_comi^T). r_comi=r_i - COM . (factor of V also cancels off with L in L=I*ω or ω=I^-1*L)
// L= Σrho_i*(r_comi x f_i) . COM, I computed once and reused
RigidBodyGeometry computeRigidBodyGeometry(const Magnet* magnet) {
  std::shared_ptr<const System> system = magnet->system();
  int ncells = system->grid().ncells();
  CuSystem cusys = system->cu();
  CuParameter rho = magnet->rho.cu();

  ComResult comWeighted = computeCom(cusys, rho, ncells, false);
  ComResult comUnweighted = computeCom(cusys, rho, ncells, true);

  InertiaAccumD inertiaP = computeInertiaSums(cusys, rho, comWeighted.com, ncells, false);
  InertiaAccumD inertiaPUnweighted = computeInertiaSums(cusys, rho, comUnweighted.com, ncells, true);

 // I is symmetric, Ixy=Iyx, Ixz=Izx, Iyz=Izy, f64 in case of conditioning for flat/thin geometries
  double I[3][3] = {{inertiaP.xx, inertiaP.xy, inertiaP.xz},
                    {inertiaP.xy, inertiaP.yy, inertiaP.yz},
                    {inertiaP.xz, inertiaP.yz, inertiaP.zz}};
  double Iu[3][3] = {{inertiaPUnweighted.xx, inertiaPUnweighted.xy, inertiaPUnweighted.xz},
                     {inertiaPUnweighted.xy, inertiaPUnweighted.yy, inertiaPUnweighted.yz},
                     {inertiaPUnweighted.xz, inertiaPUnweighted.yz, inertiaPUnweighted.zz}};


  //account for the self-inertia of cuboids, not point masses
  real3 cellsize = system->cellsize();
  double dx2 = double(cellsize.x) * double(cellsize.x);
  double dy2 = double(cellsize.y) * double(cellsize.y);
  double dz2 = double(cellsize.z) * double(cellsize.z);

  double selfFactor = double(comWeighted.weightSum) / 12.0;
  I[0][0] += selfFactor * (dy2 + dz2);
  I[1][1] += selfFactor * (dx2 + dz2);
  I[2][2] += selfFactor * (dx2 + dy2);

  double selfFactorU = double(comUnweighted.weightSum) / 12.0;
  Iu[0][0] += selfFactorU * (dy2 + dz2);
  Iu[1][1] += selfFactorU * (dx2 + dz2);
  Iu[2][2] += selfFactorU * (dx2 + dy2);

  // Tikhonov regularization trick: add a small nonzero term to avoid singularity/instability
  // see Numerical Recipes, 3rd ed, sec 19.5. probably no longer needed with cuboid self-inertia, but leaving just in case
  //double trace = I[0][0] + I[1][1] + I[2][2];
  //double lambda = 1e-10 * trace;
  //for (int d = 0; d < 3; d++) I[d][d] += lambda;

  RigidBodyGeometry geom;
  geom.com = comWeighted.com;
  geom.comUnweighted = comUnweighted.com;
  geom.totalRho = comWeighted.weightSum;
  geom.ncellsInGeometry = comUnweighted.weightSum;
  for (int a = 0; a < 3; a++){
    for (int b = 0; b < 3; b++) {
      geom.I[a][b] = I[a][b];
      geom.IUnweighted[a][b] = Iu[a][b];
    }
  }
  invert3x3(I, geom.Iinv);
  invert3x3(Iu, geom.IinvUnweighted);
  return geom;
}

void removeRigidBodyModes(Field& f, const RigidBodyGeometry& geom,
                          const Magnet* magnet, bool unweighted) {
  CuParameter rho = magnet->rho.cu();
  int ncells = f.system()->grid().ncells();

  real3 com;
  double denom;
  const double (*Iinv)[3];

  if (unweighted) {
      com = geom.comUnweighted;
      denom = double(geom.ncellsInGeometry);
      Iinv = geom.IinvUnweighted;
  } else {
      com = geom.com;
      denom = double(geom.totalRho);
      Iinv = geom.Iinv;
  }

  const TransRotAccumD* d_accum = reduceTransRotSumsDevice(f.cu(), rho, com, unweighted, ncells);

  Mat3x3 IinvArg;
  for (int a = 0; a < 3; a++){
    for (int b = 0; b < 3; b++){
      IinvArg.m[a][b] = Iinv[a][b];
    }
  }

  cudaLaunch(ncells, k_subtractRigidModesGeneric, f.cu(), com, d_accum, denom, IinvArg);
}
