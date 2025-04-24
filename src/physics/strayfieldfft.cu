#include <cufft.h>

#include <memory>
#include <vector>

#include "antiferromagnet.hpp"
#include "constants.hpp"
#include "cudalaunch.hpp"
#include "ferromagnet.hpp"
#include "fieldops.hpp"
#include "magnet.hpp"
#include "quantityevaluator.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "parameter.hpp"
#include "strayfieldfft.hpp"
#include "strayfieldkernel.hpp"
#include "system.hpp"

#if FP_PRECISION == SINGLE
const cufftType FFT = CUFFT_R2C;
const cufftType IFFT = CUFFT_C2R;
const auto& fftExec = cufftExecR2C;
const auto& ifftExec = cufftExecC2R;
#elif FP_PRECISION == DOUBLE
const cufftType FFT = CUFFT_D2Z;
const cufftType IFFT = CUFFT_Z2D;
const auto& fftExec = cufftExecD2Z;
const auto& ifftExec = cufftExecZ2D;
#endif

#define __CUDAOP__ inline __device__ __host__

// No simpel operator overloading due to definition of real2.
__CUDAOP__ complex sum(complex a, complex b) {
#if FP_PRECISION == SINGLE
  return cuCaddf(a, b);
#elif FP_PRECISION == DOUBLE
  return cuCadd(a, b);
#endif
}

__CUDAOP__ complex prod(complex a, complex b) {
#if FP_PRECISION == SINGLE
  return cuCmulf(a, b);
#elif FP_PRECISION == DOUBLE
  return cuCmul(a, b);
#endif
}

__global__ void k_pad(CuField out, CuField in1, CuParameter msat1, CuField in2, CuParameter msat2, real fac) {
  int outIdx = blockIdx.x * blockDim.x + threadIdx.x;
  
  Grid outgrid = out.system.grid;
  Grid ingrid = in1.system.grid;

  if (outIdx >= outgrid.ncells())
    return;

  int3 outCoo = outgrid.index2coord(outIdx);
  int3 inCoo = outCoo - outgrid.origin() + ingrid.origin();
  int inIdx = ingrid.coord2index(inCoo);

  if (in1.cellInGeometry(inCoo)) {
    real Ms1 = msat1.valueAt(inIdx);
    real Ms2 = msat2.valueAt(inIdx);
    for (int c = 0; c < out.ncomp; c++)
      out.setValueInCell(outIdx, c, (Ms1 * in1.valueAt(inIdx, c) + Ms2 * in2.valueAt(inIdx, c)) / fac);
  } else {
    for (int c = 0; c < out.ncomp; c++)
      out.setValueInCell(outIdx, c, 0.0);
  }
}

__global__ void k_unpad(CuField out, CuField in) {
  int outIdx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry of destiny field, set to zero and return
  // early
  if (!out.cellInGeometry(outIdx)) {
    if (out.cellInGrid(outIdx))
        out.setVectorInCell(outIdx, real3{0, 0, 0});
    return;
  }

  Grid outgrid = out.system.grid;
  Grid ingrid = in.system.grid;

  // Output coordinate relative to the origin of the output grid
  int3 outRelCoo = outgrid.index2coord(outIdx) - outgrid.origin();

  // Input coordinate relative to the origin of the input grid
  int3 inRelCoo = ingrid.size() - outgrid.size() + outRelCoo;

  int inIdx = ingrid.coord2index(inRelCoo + ingrid.origin());

  for (int c = 0; c < out.ncomp; c++) {
    out.setValueInCell(outIdx, c, in.valueAt(inIdx, c));
  }
}

static void checkCufftResult(cufftResult result) {
  if (result != CUFFT_SUCCESS)
    throw std::runtime_error("cufft error in demag convolution");
}

__global__ void k_apply_kernel_3d(complex* hx,
                                  complex* hy,
                                  complex* hz,
                                  complex* mx,
                                  complex* my,
                                  complex* mz,
                                  complex* kxx,
                                  complex* kyy,
                                  complex* kzz,
                                  complex* kxy,
                                  complex* kxz,
                                  complex* kyz,
                                  complex preFactor,
                                  int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  hx[i] = prod(preFactor, (sum(sum(prod(kxx[i], mx[i]), prod(kxy[i], my[i])), prod(kxz[i], mz[i]))));
  hy[i] = prod(preFactor, (sum(sum(prod(kxy[i], mx[i]), prod(kyy[i], my[i])), prod(kyz[i], mz[i]))));
  hz[i] = prod(preFactor, (sum(sum(prod(kxz[i], mx[i]), prod(kyz[i], my[i])), prod(kzz[i], mz[i]))));
}

__global__ void k_apply_kernel_2d(complex* hx,
                                  complex* hy,
                                  complex* hz,
                                  complex* mx,
                                  complex* my,
                                  complex* mz,
                                  complex* kxx,
                                  complex* kyy,
                                  complex* kzz,
                                  complex* kxy,
                                  complex preFactor,
                                  int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  hx[i] = prod(preFactor, (sum(prod(kxx[i], mx[i]), prod(kxy[i], my[i]))));
  hy[i] = prod(preFactor, (sum(prod(kxy[i], mx[i]), prod(kyy[i], my[i]))));
  hz[i] = prod(preFactor, prod(kzz[i], mz[i]));
}

StrayFieldFFTExecutor::StrayFieldFFTExecutor(
    const Magnet* magnet,
    std::shared_ptr<const System> system, int order, double switchingradius)
    : StrayFieldExecutor(magnet, system),
      kernel_(system->grid(), magnet_->grid(), magnet->world(), order, switchingradius),
      kfft(6),
      hfft(3),
      mfft(3) {
  int3 size = kernel_.grid().size();
  fftSize = {size.x / 2 + 1, size.y, size.z};
  int ncells = fftSize.x * fftSize.y * fftSize.z;

  for (auto& p : kfft)
    cudaMalloc(reinterpret_cast<void**>(&p), ncells * sizeof(complex));
  for (auto& p : mfft)
    cudaMalloc(reinterpret_cast<void**>(&p), ncells * sizeof(complex));
  for (auto& p : hfft)
    cudaMalloc(reinterpret_cast<void**>(&p), ncells * sizeof(complex));

  checkCufftResult(cufftPlan3d(&forwardPlan, size.z, size.y, size.x, FFT));
  checkCufftResult(cufftPlan3d(&backwardPlan, size.z, size.y, size.x, IFFT));

  cufftSetStream(forwardPlan, getCudaStream());
  cufftSetStream(backwardPlan, getCudaStream());

  for (int comp = 0; comp < 6; comp++)
    checkCufftResult(
        fftExec(forwardPlan, kernel_.field().device_ptr(comp), kfft.at(comp)));
}

StrayFieldFFTExecutor::~StrayFieldFFTExecutor() {
  for (auto p : mfft)
    cudaFree(p);
  for (auto p : kfft)
    cudaFree(p);
  for (auto p : hfft)
    cudaFree(p);
  
  checkCufftResult(cufftDestroy(forwardPlan));
  checkCufftResult(cufftDestroy(backwardPlan));
}

Field StrayFieldFFTExecutor::exec() const {

  // pad m, and multiply with msat
  std::shared_ptr<System> kernelSystem =
      std::make_shared<System>(magnet_->world(), kernel_.grid());
  std::unique_ptr<Field> mpad(new Field(kernelSystem, 3));

  // Launch kernel function in different scopes to avoid unnecessary copies
  real fac;
  if (const Ferromagnet* mag = dynamic_cast<const Ferromagnet*>(magnet_)) {
    auto m = mag->magnetization()->field().cu();
    auto ms = mag->msat.cu();
    fac = 2.0;
    cudaLaunch(mpad->grid().ncells(), k_pad, mpad->cu(), m, ms, m, ms, fac);
  }
  else if (const Antiferromagnet* mag = dynamic_cast<const Antiferromagnet*>(magnet_)) {
    auto m1 = mag->sub1()->magnetization()->field().cu();
    auto m2 = mag->sub2()->magnetization()->field().cu();
    auto ms1 = mag->sub1()->msat.cu();
    auto ms2 = mag->sub2()->msat.cu();
    fac = 1.0;
    cudaLaunch(mpad->grid().ncells(), k_pad, mpad->cu(), m1, ms1, m2, ms2, fac);
  }

  // Forward fourier transforms
  for (int comp = 0; comp < 3; comp++)
    checkCufftResult(
        fftExec(forwardPlan, mpad->device_ptr(comp), mfft.at(comp)));
  
  // apply kernel on m_fft
  int ncells = fftSize.x * fftSize.y * fftSize.z;
  complex preFactor{-MU0 / kernel_.grid().ncells(), 0};
  if (kernel_.grid().size().z == 1 && kernel_.grid().origin().z == 0) {
    // if the h field and m field are two dimensional AND are in the same plane
    // (kernel grid origin at z=0) then the kernel matrix has only 4 relevant
    // components and a more efficient cuda kernel can be used:
    cudaLaunch(ncells, k_apply_kernel_2d, hfft.at(0), hfft.at(1), hfft.at(2),
               mfft.at(0), mfft.at(1), mfft.at(2), kfft.at(0), kfft.at(1),
               kfft.at(2), kfft.at(3), preFactor, ncells);
  } else {
    cudaLaunch(ncells, k_apply_kernel_3d, hfft.at(0), hfft.at(1), hfft.at(2),
               mfft.at(0), mfft.at(1), mfft.at(2), kfft.at(0), kfft.at(1),
               kfft.at(2), kfft.at(3), kfft.at(4), kfft.at(5), preFactor,
               ncells);
  }

  // backward fourier transfrom
  for (int comp = 0; comp < 3; comp++)
    checkCufftResult(
      ifftExec(backwardPlan, hfft.at(comp), mpad->device_ptr(comp)));

  // unpad
  Field h(system_, 3);
  cudaLaunch(h.grid().ncells(), k_unpad, h.cu(), mpad->cu());
  return h;
}
