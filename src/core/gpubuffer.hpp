#pragma once
#include <map>
#include <vector>

#include "cudaerror.hpp"
#include "cudastream.hpp"

class GpuBufferPool {
 public:
  GpuBufferPool() = default;
  GpuBufferPool(GpuBufferPool const&) = delete;
  void operator=(GpuBufferPool const&) = delete;

  ~GpuBufferPool();

  void* allocate(size_t size);
  void free(void**);
  void recycle(void**);

 private:
  std::map<size_t, std::vector<void*>> pool_;
  std::map<void*, size_t> inUse_;
};

extern GpuBufferPool bufferPool;

template <typename T>
class GpuBuffer {
 public:
  GpuBuffer(size_t N) { allocate(N); }
  GpuBuffer() {}
  ~GpuBuffer() { recycle(); }

  GpuBuffer(const std::vector<T>& other) {
    allocate(other.size());
    if (size_ > 0) {
      checkCudaError(cudaMemcpyAsync(ptr_, &other[0], size_ * sizeof(T),
                                     cudaMemcpyHostToDevice, getCudaStream()));
    }
  }

  // Copy constructor
  GpuBuffer(const GpuBuffer& other) {
    allocate(other.size());
    if (size_ == 0)
      return;

    checkCudaError(cudaMemcpyAsync(ptr_, other.ptr_, size_ * sizeof(T),
                                   cudaMemcpyDeviceToDevice, getCudaStream()));
  }

  // Move constructor
  GpuBuffer(GpuBuffer&& other) {
    ptr_ = other.ptr_;
    size_ = other.size_;
    other.ptr_ = nullptr;
    other.size_ = 0;
  }

  // Copy assignment
  GpuBuffer<T>& operator=(const GpuBuffer<T>& other) {
    allocate(other.size());
    if (size_ > 0) {
      checkCudaError(cudaMemcpyAsync(ptr_, other.ptr_, size_ * sizeof(T),
                                     cudaMemcpyDeviceToDevice,
                                     getCudaStream()));
    }
    return *this;
  }

  // Move assignment
  GpuBuffer<T>& operator=(GpuBuffer<T>&& other) {
    ptr_ = other.ptr_;
    size_ = other.size_;
    other.ptr_ = nullptr;
    other.size_ = 0;
    return *this;
  }

  void allocate(size_t size) {
    if (size_ == size)
      return;

    recycle();

    if (size > 0) {
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

  size_t size() const { return size_; }
  T* get() const { return ptr_; }

 private:
  T* ptr_ = nullptr;
  size_t size_ = 0;
};