#include <stdexcept>

#include "cudaerror.hpp"
#include "cudalaunch.hpp"
#include "cudastream.hpp"
#include "datatypes.hpp"
#include "field.hpp"
#include "gpubuffer.hpp"
#include "reduce.hpp"

__global__ void k_maxAbsValue(real* result, CuField f) {
  // Reduce to a block
  __shared__ real sdata[BLOCKDIM];
  int ncells = f.system.grid.ncells();
  int tid = threadIdx.x;
  real threadValue = 0.0;
  for (int i = tid; i < ncells; i += BLOCKDIM) {
    if (!f.cellInGeometry(i))
      continue;
    for (int c = 0; c < f.ncomp; c++) {
      real value = abs(f.valueAt(i, c));
      threadValue = value > threadValue ? value : threadValue;
    }
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

real maxAbsValue(const Field& f) {
  GpuBuffer<real> d_result(1);
  cudaLaunchReductionKernel(k_maxAbsValue, d_result.get(), f.cu());

  // copy the result to the host and return
  real result;
  checkCudaError(cudaMemcpyAsync(&result, d_result.get(), 1 * sizeof(real),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  return result;
}

__global__ void k_maxVecNorm(real* result, CuField f) {
  // Reduce to a block
  __shared__ real sdata[BLOCKDIM];
  int ncells = f.system.grid.ncells();
  int tid = threadIdx.x;
  real threadValue = 0.0;
  for (int i = tid; i < ncells; i += BLOCKDIM) {
    if (!f.cellInGeometry(i))
      continue;

    real3 cellVec = f.vectorAt(i);
    real cellNorm = norm(cellVec);
    if (cellNorm > threadValue)
      threadValue = cellNorm;
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

real maxVecNorm(const Field& f) {
  if (f.ncomp() != 3) {
    throw std::runtime_error(
        "the input field of maxVecNorm should have 3 components");
  }
  
  GpuBuffer<real> d_result(1);
  cudaLaunchReductionKernel(k_maxVecNorm, d_result.get(), f.cu());

  // copy the result to the host and return
  real result;
  checkCudaError(cudaMemcpyAsync(&result, d_result.get(), 1 * sizeof(real),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  return result;
}

__global__ void k_average(real* result, CuField f, int comp, int cellsingeo) {
  __shared__ real sdata[BLOCKDIM];
  int tid = threadIdx.x;
  int ncells = f.system.grid.ncells();

  // Reduce to a block
  real threadValue = 0.0;
  for (int i = tid; i < ncells; i += BLOCKDIM) {
    if (!f.cellInGeometry(i))
      continue;
    threadValue += f.valueAt(i, comp);
  }
  sdata[tid] = threadValue;
  __syncthreads();

  // Reduce the block
  for (unsigned int s = BLOCKDIM / 2; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  // TODO: check if loop unrolling makes sense here

  // Set the result
  if (tid == 0)
    *result = sdata[0] / cellsingeo;
}

real fieldComponentAverage(const Field& f, int comp) {
  if (comp >= f.ncomp()) {
    throw std::runtime_error("Can not take the average of component " +
                             std::to_string(comp) +
                             " of a field which has only " +
                             std::to_string(f.ncomp()) + " components");
  }
  
  real result;
  int cellsingeo = f.system()->cellsingeo();
  GpuBuffer<real> d_result(1);
  cudaLaunchReductionKernel(k_average, d_result.get(), f.cu(), comp, cellsingeo);
  checkCudaError(cudaMemcpyAsync(&result, d_result.get(), sizeof(real),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  return result;
}

std::vector<real> fieldAverage(const Field& f) {
  std::vector<real> result;
  for (int c = 0; c < f.ncomp(); c++)
    result.push_back(fieldComponentAverage(f, c));
  return result;
}

__global__ void k_dotSum(real* result, CuField f, CuField g) {
  __shared__ real sdata[BLOCKDIM];
  int ncells = f.system.grid.ncells();
  int tid = threadIdx.x;

  real threadValue = 0.0;
  for (int i = tid; i < ncells; i += BLOCKDIM) {
    if (!f.cellInGeometry(i))
      continue;
    for (int c = 0; c < f.ncomp; c++)
      threadValue += f.valueAt(i, c) * g.valueAt(i, c);
  }

  sdata[tid] = threadValue;
  __syncthreads();

  // Reduce the block
  for (unsigned int s = BLOCKDIM / 2; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  // Set the result
  if (tid == 0)
    *result = sdata[0];
}

real dotSum(const Field& f, const Field& g) {
  if (f.system() != g.system())
    throw std::invalid_argument(
        "Can not take the dot sum of the two fields because they are not "
        "defined on the same system.");

  GpuBuffer<real> d_result(1);
  cudaLaunchReductionKernel(k_dotSum, d_result.get(), f.cu(), g.cu());

  // copy the result to the host and return
  real result;
  checkCudaError(cudaMemcpyAsync(&result, d_result.get(), sizeof(real),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  return result;
}

__global__ void k_idxInRegions(bool* result, unsigned int* regions, size_t size, unsigned int ridx) {
  __shared__ bool sdata[BLOCKDIM];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int thread_id = threadIdx.x;

  bool found = false;

  for (int i = tid; i < size; i += blockDim.x * gridDim.x) {
    if (regions[i] == ridx) {
      found = true;
      break;
    }
  }

  sdata[thread_id] = found;
  __syncthreads();

  // Reduce the block
  for (unsigned int s = BLOCKDIM / 2; s > 0; s >>= 1) {
    if (thread_id < s)
      sdata[thread_id] = sdata[thread_id] || sdata[thread_id + s];
    __syncthreads();
  }

  // Set the result
  if (thread_id == 0)
    *result = sdata[0];
}

bool idxInRegions(GpuBuffer<unsigned int> regions, unsigned int idx) {

  GpuBuffer<bool> d_result(1);
  cudaLaunchReductionKernel(k_idxInRegions, d_result.get(), regions.get(), regions.size(), idx);

  // copy the result to the host and return
  bool result;
  checkCudaError(cudaMemcpyAsync(&result, d_result.get(), sizeof(bool),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  return result;
}

__global__ void k_isUniformComponent(bool* isUniform, CuField f, int c) {
  __shared__ bool sdata[BLOCKDIM];
  int ncells = f.system.grid.ncells();
  int tid = threadIdx.x;

  bool result = true;

  real firstValue = 0.0;
  for (int i = 0; i < ncells; i ++)
    if (f.cellInGeometry(i)) {
      firstValue = f.valueAt(i, c);
      break;
    }

  for (int i = tid; i < ncells; i += BLOCKDIM) {
    if (!f.cellInGeometry(i))
      continue;

    if (f.valueAt(i, c) != firstValue) {
      result = false;
      break;
    }
  }

  sdata[tid] = result;
  __syncthreads();

  for (unsigned int s = BLOCKDIM / 2; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] = sdata[tid] && sdata[tid + s];
    __syncthreads();
  }

  if (tid == 0)
    *isUniform = sdata[0];
}

bool isUniformFieldComponent(const Field& f, int comp) {
  GpuBuffer<bool> d_isUniform(1);

  cudaLaunchReductionKernel(k_isUniformComponent, d_isUniform.get(), f.cu(), comp);

  bool isUniform;
  checkCudaError(cudaMemcpyAsync(&isUniform, d_isUniform.get(), sizeof(bool),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  return isUniform;
}

bool isUniformField(const Field& f) {
  for (int c = 0; c < f.ncomp(); c++) {
    if (!isUniformFieldComponent(f, c))
      return false;
  }
  return true;
}