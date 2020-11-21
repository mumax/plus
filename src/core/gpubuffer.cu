#include <iomanip>
#include <iostream>
#include <map>

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

void GpuBufferPool::printInfo() const {
  // bufferuse map:
  //   key           buffer size
  //   value.first   in use count
  //   value.second  in pool count
  std::map<size_t, std::pair<int, int>> bufferuse;

  // count number of used buffers (for each buffersize seperately)
  for (auto u : inUse_)
    bufferuse[u.second].first++;

  // get number of buffers in bufferpool for each buffer size
  for (const auto& p : pool_)
    bufferuse[p.first].second = p.second.size();

  int totalMemUsed = 0;
  for (auto b : bufferuse) {
    totalMemUsed += b.first * (b.second.first + b.second.second);
  }

  std::cout << "GPU BUFFER POOL INFORMATION:" << std::endl;
  int colwidth = 10;
  std::cout << std::setw(colwidth) << "bufsize";
  std::cout << std::setw(colwidth) << "in use";
  std::cout << std::setw(colwidth) << "free" << std::endl;
  for (auto b : bufferuse) {
    std::cout << std::setw(colwidth) << b.first;
    std::cout << std::setw(colwidth) << b.second.first;
    std::cout << std::setw(colwidth) << b.second.second << std::endl;
  }
  std::cout << "Total used GPU memory:  " << totalMemUsed << " bytes"
            << std::endl;
}