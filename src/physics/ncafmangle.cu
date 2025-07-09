#include "ncafm.hpp"
#include "cudalaunch.hpp"
#include "ncafmangle.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "parameter.hpp"

__global__ void k_angle(CuField angleField,
                        const CuField mField1,
                        const CuField mField2,
                        const CuField mField3,
                        const CuParameter ncafmex,
                        const CuParameter msat1,
                        const CuParameter msat2,
                        const CuParameter msat3) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!angleField.cellInGeometry(idx)) {
    if (angleField.cellInGrid(idx))
      angleField.setVectorInCell(idx, real3{0, 0, 0});
      return;
  }

  if (ncafmex.valueAt(idx) == 0) {
    angleField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  bool b1 = (msat1.valueAt(idx) != 0);
  bool b2 = (msat2.valueAt(idx) != 0);
  bool b3 = (msat3.valueAt(idx) != 0);

  real dev12 = acos(dot(mField1.vectorAt(idx) * b1, mField2.vectorAt(idx) * b2))
                    - (120.0 * M_PI / 180.0);
  real dev13 = acos(dot(mField1.vectorAt(idx) * b1, mField3.vectorAt(idx) * b3))
                    - (120.0 * M_PI / 180.0);
  real dev23 = acos(dot(mField2.vectorAt(idx) * b2, mField3.vectorAt(idx) * b3))
                    - (120.0 * M_PI / 180.0);

  angleField.setVectorInCell(idx, real3{dev12, dev13, dev23});
}

__global__ void k_maxAngle(real* result,
                           CuField sub1,
                           CuField sub2,
                           CuParameter msat1,
                           CuParameter msat2) {
  // Reduce to a block
  __shared__ real sdata[BLOCKDIM];
  int ncells = sub1.system.grid.ncells();
  int tid = threadIdx.x;
  real threadValue = -1.0;

  for (int i = tid; i < ncells; i += BLOCKDIM) {
    if (!sub1.cellInGeometry(i))
      continue;
    if (msat1.valueAt(i) == 0 || msat2.valueAt(i) == 0)
      continue;
    real angle = fabs(acos(dot(sub1.vectorAt(i), sub2.vectorAt(i))) - 2.0 * M_PI / 3);

    threadValue = angle > threadValue ? angle : threadValue;
  }
  sdata[tid] = threadValue;
  __syncthreads();

  // Reduce the block
  for (unsigned int s = BLOCKDIM / 2; s > 0; s >>= 1) {
    if (tid < s)
      if (sdata[tid + s] > sdata[tid])
        sdata[tid] = sdata[tid + s];
    __syncthreads();
  }
  // TODO: check if loop unrolling makes sense here

  // Set the result
  if (tid == 0)
    *result = sdata[0];
}

Field evalAngleField(const NcAfm* magnet) {
  // Three components for the angles between 1-2, 1-3 and 2-3
  Field angleField(magnet->system(), 3);

  cudaLaunch(angleField.grid().ncells(), k_angle, angleField.cu(),
             magnet->sub1()->magnetization()->field().cu(),
             magnet->sub2()->magnetization()->field().cu(),
             magnet->sub3()->magnetization()->field().cu(),
             magnet->afmex_cell.cu(),
             magnet->sub1()->msat.cu(),
             magnet->sub2()->msat.cu(),
             magnet->sub3()->msat.cu());
  return angleField;
}

real evalMaxAngle(const Ferromagnet* sub1, const Ferromagnet* sub2) {
  if (!sub1->hostMagnet()->asNcAfm() || !sub2->hostMagnet()->asNcAfm())
    throw std::invalid_argument("Maximum angle can only be calculated for NcAfm sublattices.");
  if (sub1->hostMagnet()->asNcAfm() != sub2->hostMagnet()->asNcAfm())
    throw std::invalid_argument("Maximum angle can only be calculated for sublattices in the same NcAfm host.");
  if (sub1 == sub2)
    throw std::invalid_argument("Maximum angle can only be calculated for different sublattices.");
  auto m1 = sub1->magnetization()->field().cu();
  auto m2 = sub2->magnetization()->field().cu();
  auto ms1 = sub1->msat.cu();
  auto ms2 = sub2->msat.cu();

  GpuBuffer<real> d_result(1);
  cudaLaunchReductionKernel(k_maxAngle, d_result.get(), m1, m2, ms1, ms2);

  real result;
  checkCudaError(cudaMemcpyAsync(&result, d_result.get(), 1 * sizeof(real),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  return result;
}

NcAfm_FieldQuantity angleFieldQuantity(const NcAfm* magnet) {
  return NcAfm_FieldQuantity(magnet, evalAngleField, 3, "angle_field", "rad");
}