#include "cudaerror.hpp"
#include "gpubuffer.hpp"

GpuBufferPool bufferPool;

GpuBufferPool::~GpuBufferPool() {
  for (const auto& poolEntry : pool_)
    for (auto& ptr : poolEntry.second)
      checkCudaError(cudaFree(ptr));
}

void* GpuBufferPool::allocate(size_t size) {
  void* ptr;
  if (pool_[size].size() == 0) {
    checkCudaError(cudaMalloc((void**)&ptr, size));
  } else {
    ptr = pool_[size].back();
    pool_[size].pop_back();
  }
  inUse_[ptr] = size;
  return ptr;
}

void GpuBufferPool::free(void** ptr) {
  inUse_.erase(*ptr);
  checkCudaError(cudaFree(*ptr));
  *ptr = nullptr;
}

void GpuBufferPool::recycle(void** ptr) {
  auto inUseIt = inUse_.find(*ptr);
  int size = inUseIt->second;
  inUse_.erase(inUseIt);
  pool_[size].push_back(*ptr);
  *ptr = nullptr;
}