#pragma once
#include <map>
#include <vector>

#include "cudaerror.hpp"
#include "cudastream.hpp"

/**
 * Gpu memory pool.
 *
 * The allocation of large memory blocks on the gpu is an heavy operation.
 * Using a memory pool avoids the need the allocate new memory on the gpu over
 * and over again by recycling the memory blocks.
 *
 * The GpuBufferPool user is responsible for recycling or freeing the allocated
 * memory after use. If you need a gpu buffer, consider to use a GpuBuffer
 * object which does this kind of memory management for you.
 */
class GpuBufferPool {
 public:
  GpuBufferPool() = default;
  GpuBufferPool(GpuBufferPool const&) = delete;
  void operator=(GpuBufferPool const&) = delete;

  ~GpuBufferPool();

  /**
   * Returns a pointer to allocated gpu memory block with the specified size
   * (in bytes).
   *
   * If there is an allocate memoery blcok available in the pool which has the
   * correct size, then a pointer to this memory be returned and removed from
   * the pool which avoids the allocation of new memory on the gpu. If there is
   * no memoery block with the correct size available in the pool, then new gpu
   * memory will be allocated.
   *
   * The caller is responsible to release the memory after use. This should be
   * done by calling the 'free' or 'recycle' method on the GpuBufferPool which
   * is used to allocate the gpu memory.
   */
  void* allocate(size_t size);

  /**
   * Free gpu memory allocated by this buffer pool.
   */
  void free(void**);

  /**
   * Recycle gpu memory allocated by this buffer pool.
   * The allocated memory will become available again in the buffer pool.
   */
  void recycle(void**);

  /**
   * Prints the number of used and available allocate memory by this buffer
   * pool. This function is useful to detect memory leaks.
   */
  void printInfo() const;

 private:
  std::map<size_t, std::vector<void*>> pool_;
  std::map<void*, size_t> inUse_;
};

/**
 * The global GpuBufferPool instance
 */
extern GpuBufferPool bufferPool;

template <typename T>
class GpuBuffer {
 public:
  /**
   * Constructs a gpu buffer for N objects of type T;
   */
  GpuBuffer(size_t N) { allocate(N); }

  /**
   * Constructs an empty gpu buffer.
   */
  GpuBuffer() {}

  /**
   * The GpuBuffer destructor releases the memory back to the bufferPool.
   */
  ~GpuBuffer() { recycle(); }

  /**
   * Construct a gpu buffer from a std::vector.
   * The size of the gpu buffer matches the size of the vector.
   * The values are copied from the vector on the host to the gpu.
   */
  GpuBuffer(const std::vector<T>& other) {
    allocate(other.size());
    if (size_ > 0) {
      checkCudaError(cudaMemcpyAsync(ptr_, &other[0], size_ * sizeof(T),
                                     cudaMemcpyHostToDevice, getCudaStream()));
    }
  }

  /**
   * Copy constructor
   */
  GpuBuffer(const GpuBuffer& other) {
    allocate(other.size());
    if (size_ == 0)
      return;
    checkCudaError(cudaMemcpyAsync(ptr_, other.ptr_, size_ * sizeof(T),
                                   cudaMemcpyDeviceToDevice, getCudaStream()));
  }

  /**
   * Move constructor
   */
  GpuBuffer(GpuBuffer&& other) {
    recycle();
    ptr_ = other.ptr_;
    size_ = other.size_;
    other.ptr_ = nullptr;
    other.size_ = 0;
  }

  /**
   * Copy assignment
   */
  GpuBuffer<T>& operator=(const GpuBuffer<T>& other) {
    allocate(other.size());
    if (size_ > 0) {
      checkCudaError(cudaMemcpyAsync(ptr_, other.ptr_, size_ * sizeof(T),
                                     cudaMemcpyDeviceToDevice,
                                     getCudaStream()));
    }
    return *this;
  }

  /**
   * Move assignment
   */
  GpuBuffer<T>& operator=(GpuBuffer<T>&& other) {
    recycle();
    ptr_ = other.ptr_;
    size_ = other.size_;
    other.ptr_ = nullptr;
    other.size_ = 0;
    return *this;
  }

  /**
   * Resizes the buffer.
   *
   * The buffer content will only be preserved if the new size matches the old
   * size.
   */
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

  /**
   * Empties the buffer and releases memory back into the pool.
   */
  void recycle() {
    if (ptr_)
      bufferPool.recycle((void**)&ptr_);
    ptr_ = nullptr;
    size_ = 0;
  }

  /**
   * Returns the size (number of elements) of the buffer.
   */
  size_t size() const { return size_; }
  T* get() const { return ptr_; }

 private:
  T* ptr_ = nullptr;
  size_t size_ = 0;
};