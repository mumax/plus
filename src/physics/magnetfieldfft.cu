#include <cufft.h>

#include <memory>
#include <vector>

#include "constants.hpp"
#include "cudalaunch.hpp"
#include "field.hpp"
#include "magnetfieldfft.hpp"
#include "magnetfieldkernel.hpp"
#include "parameter.hpp"
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

__CUDAOP__ complex operator+(complex a, complex b) {
#if FP_PRECISION == SINGLE
  return cuCaddf(a, b);
#elif FP_PRECISION == DOUBLE
  return cuCadd(a, b);
#endif
}

__CUDAOP__ complex operator*(complex a, complex b) {
#if FP_PRECISION == SINGLE
  return cuCmulf(a, b);
#elif FP_PRECISION == DOUBLE
  return cuCmul(a, b);
#endif
}

__global__ void k_pad(CuField out, CuField in, CuParameter msat) {
  int outIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (outIdx >= out.grid.ncells())
    return;

  int3 outCoo = out.grid.index2coord(outIdx);
  int3 inCoo = outCoo - out.grid.origin() + in.grid.origin();
  int inIdx = in.grid.coord2index(inCoo);

  if (in.grid.cellInGrid(inCoo)) {
    real Ms = msat.valueAt(inIdx);
    for (int c = 0; c < out.ncomp; c++)
      out.setValueInCell(outIdx, c, Ms * in.valueAt(inIdx, c));
  } else {
    for (int c = 0; c < out.ncomp; c++)
      out.setValueInCell(outIdx, c, 0.0);
  }
}

__global__ void k_unpad(CuField out, CuField in) {
  int outIdx = blockIdx.x * blockDim.x + threadIdx.x;

  if (outIdx >= out.grid.ncells())
    return;

  // Output coordinate relative to the origin of the output grid
  int3 outRelCoo = out.grid.index2coord(outIdx) - out.grid.origin();

  // Input coordinate relative to the origin of the input grid
  int3 inRelCoo = in.grid.size() - out.grid.size() + outRelCoo;

  int inIdx = in.grid.coord2index(inRelCoo + in.grid.origin());

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
  hx[i] = preFactor * (kxx[i] * mx[i] + kxy[i] * my[i] + kxz[i] * mz[i]);
  hy[i] = preFactor * (kxy[i] * mx[i] + kyy[i] * my[i] + kyz[i] * mz[i]);
  hz[i] = preFactor * (kxz[i] * mx[i] + kyz[i] * my[i] + kzz[i] * mz[i]);
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
  hx[i] = preFactor * (kxx[i] * mx[i] + kxy[i] * my[i]);
  hy[i] = preFactor * (kxy[i] * mx[i] + kyy[i] * my[i]);
  hz[i] = preFactor * (kzz[i] * mz[i]);
}

MagnetFieldFFTExecutor::MagnetFieldFFTExecutor(Grid gridOut,
                                               Grid gridIn,
                                               real3 cellsize)
    : kernel_(gridOut, gridIn, cellsize), kfft(6), hfft(3), mfft(3) {
  int3 size = kernel_.grid().size();
  fftSize = {size.x / 2 + 1, size.y, size.z};
  int ncells = fftSize.x * fftSize.y * fftSize.z;

  for (auto& p : kfft)
    cudaMalloc((void**)&p, ncells * sizeof(complex));
  for (auto& p : mfft)
    cudaMalloc((void**)&p, ncells * sizeof(complex));
  for (auto& p : hfft)
    cudaMalloc((void**)&p, ncells * sizeof(complex));

  checkCufftResult(cufftPlan3d(&forwardPlan, size.z, size.y, size.x, FFT));
  checkCufftResult(cufftPlan3d(&backwardPlan, size.z, size.y, size.x, IFFT));

  cufftSetStream(forwardPlan, getCudaStream());
  cufftSetStream(backwardPlan, getCudaStream());

  for (int comp = 0; comp < 6; comp++)
    checkCufftResult(
        fftExec(forwardPlan, kernel_.field().devptr(comp), kfft.at(comp)));
}

MagnetFieldFFTExecutor::~MagnetFieldFFTExecutor() {
  for (auto p : mfft)
    cudaFree(p);
  for (auto p : kfft)
    cudaFree(p);
  for (auto p : hfft)
    cudaFree(p);

  checkCufftResult(cufftDestroy(forwardPlan));
  checkCufftResult(cufftDestroy(backwardPlan));
}

void MagnetFieldFFTExecutor::exec(Field* h,
                                  const Field* m,
                                  const Parameter* msat) const {
  // pad m, and multiply with msat
  System kernelSystem(h->system()->world(), kernel_.grid());
  std::unique_ptr<Field> mpad(new Field(&kernelSystem, 3));
  cudaLaunch(mpad->grid().ncells(), k_pad, mpad->cu(), m->cu(), msat->cu());

  // Forward fourier transforms
  for (int comp = 0; comp < 3; comp++)
    checkCufftResult(fftExec(forwardPlan, mpad->devptr(comp), mfft.at(comp)));

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
    checkCufftResult(ifftExec(backwardPlan, hfft.at(comp), mpad->devptr(comp)));

  // unpad
  cudaLaunch(h->grid().ncells(), k_unpad, h->cu(), mpad->cu());
}