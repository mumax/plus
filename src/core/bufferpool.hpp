#pragma once
#include <map>
#include <vector>

#include "cudaerror.hpp"
#include "cudastream.hpp"

class BufferPool {
 public:
  BufferPool() = default;
  BufferPool(BufferPool const&) = delete;
  void operator=(BufferPool const&) = delete;

  ~BufferPool();

  void* allocate(int size);
  void free(void**);
  void recycle(void**);

 private:
  std::map<int, std::vector<void*>> pool_;
  std::map<void*, int> inUse_;
};

extern BufferPool bufferPool;

template <typename T>
class GpuBuffer {
 public:
  GpuBuffer(size_t N) { allocate(N); }
  GpuBuffer() : ptr_(nullptr) {}
  ~GpuBuffer() { recycle(); }

  // disable copy constructor
  GpuBuffer(const GpuBuffer&) = delete;

  GpuBuffer(const std::vector<T>& other) {
    allocate(other.size());
    checkCudaError(cudaMemcpyAsync(ptr_, &other[0], size_ * sizeof(T),
                                   cudaMemcpyHostToDevice, getCudaStream()));
  }

  // Move constructor
  GpuBuffer(GpuBuffer&& other) {
    ptr_ = other.ptr_;
    size_ = other.size_;
    other.ptr_ = nullptr;
    other.size_ = 0;
  }

  // Move assignment
  GpuBuffer<T>& operator=(GpuBuffer<T>&& other) {
    ptr_ = other.ptr_;
    size_ = other.size_;
    other.ptr_ = nullptr;
    other.size_ = 0;
    return *this;
  }

  // disable assignment operator
  GpuBuffer<T>& operator=(const GpuBuffer<T>&) = delete;

  void allocate(size_t size) {
    recycle();
    if (size != 0) {
      ptr_ = (T*)bufferPool.allocate(size * sizeof(T));
    } else {
      ptr_ = nullptr;
    }
    size_ = size;
  }

  void recycle() {
    if (ptr_)
      bufferPool.recycle((void**)&ptr_);
    ptr_ = nullptr;
    size_ = 0;
  }

  int size() { return size_; }
  T* get() const { return ptr_; }

 private:
  T* ptr_;
  size_t size_;
};