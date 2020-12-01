#include <iomanip>
#include <iostream>
#include <map>
#include <utility>

#include "cudaerror.hpp"
#include "gpumemorypool.hpp"

GpuMemoryPool memoryPool;

GpuMemoryPool::~GpuMemoryPool() {
  for (const auto& poolEntry : pool_)
    for (auto& ptr : poolEntry.second)
      checkCudaError(cudaFree(ptr));
}

void* GpuMemoryPool::allocate(size_t size) {
  void* ptr;
  if (pool_[size].size() == 0) {
    checkCudaError(cudaMalloc(reinterpret_cast<void**>(&ptr), size));
  } else {
    ptr = pool_[size].back();
    pool_[size].pop_back();
  }
  inUse_[ptr] = size;
  return ptr;
}

void GpuMemoryPool::free(void** ptr) {
  inUse_.erase(*ptr);
  checkCudaError(cudaFree(*ptr));
  *ptr = nullptr;
}

void GpuMemoryPool::recycle(void** ptr) {
  auto inUseIt = inUse_.find(*ptr);
  int size = inUseIt->second;
  inUse_.erase(inUseIt);
  pool_[size].push_back(*ptr);
  *ptr = nullptr;
}

void GpuMemoryPool::printInfo() const {
  // memoryUse map:
  //   key           memory block size
  //   value.first   in use count
  //   value.second  in pool count
  std::map<size_t, std::pair<int, int>> memoryUse;

  // count number of used memory blocks (for each memory block size seperately)
  for (auto u : inUse_)
    memoryUse[u.second].first++;

  // get number of memory blocks in the pool for each memory block size
  for (const auto& p : pool_)
    memoryUse[p.first].second = p.second.size();

  int totalMemUsed = 0;
  for (auto b : memoryUse) {
    totalMemUsed += b.first * (b.second.first + b.second.second);
  }

  std::cout << "GPU MEMORY POOL INFORMATION:" << std::endl;
  int colwidth = 10;
  std::cout << std::setw(colwidth) << "size(bytes)";
  std::cout << std::setw(colwidth) << "in use";
  std::cout << std::setw(colwidth) << "free" << std::endl;
  for (auto b : memoryUse) {
    std::cout << std::setw(colwidth) << b.first;
    std::cout << std::setw(colwidth) << b.second.first;
    std::cout << std::setw(colwidth) << b.second.second << std::endl;
  }
  std::cout << "Total used GPU memory:  " << totalMemUsed << " bytes"
            << std::endl;
}
