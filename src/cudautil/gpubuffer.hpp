#pragma once

#include <vector>

#include "cudaerror.hpp"
#include "cudastream.hpp"
#include "gpumemorypool.hpp"

template <typename T>
class GpuBuffer {
 public:
  // _______________________________________________________________CONSTRUCTORS

  GpuBuffer() {}                     /** Empty gpu buffer. */
  explicit GpuBuffer(size_t N);      /** Gpu buffer for N objects */
  GpuBuffer(const GpuBuffer& other); /** Copy constructor */
  GpuBuffer(GpuBuffer&& other);      /** Move constructor */

  /**
   * Construct a gpu buffer from a std::vector.
   * The size of the gpu buffer matches the size of the vector.
   * The values are copied from the vector on the host to the gpu.
   */
  explicit GpuBuffer(const std::vector<T>& other);

  /** Gpubuffer with size N, and copy N values from data pointer. */
  GpuBuffer(int N, T* data);

  // ________________________________________________________________DESTRUCTION

  /** Destroy and release allocated memory back to the memory pool. */
  ~GpuBuffer() { recycle(); }

  // ____________________________________________________________________GETTERS

  size_t size() const { return size_; } /** Number of elements in buffer. */
  T* get() const { return ptr_; }       /** Device ptr of the buffer. */
  std::vector<T> getData() const;       /** Get copy of data on the host. */

  /**
   * Copy the data from the gpubuffer to a newly allocated array on the
   * host. WARNING: This function returns a raw pointer, the user is
   * responsible to free the memory when done!
   */
  T* getHostCopy() const;

  // ___________________________________________________________________MODIFIERS

  GpuBuffer<T>& operator=(const GpuBuffer<T>& other); /** Copy assignment */
  GpuBuffer<T>& operator=(GpuBuffer<T>&& other);      /** Move assignment */

  void recycle(); /** Empty the buffer and releases memory to the pool. */

  /**
   * Resizes the buffer. The buffer content will only be preserved if the new
   * size matches the old size.
   */
  void allocate(size_t size);

  // _________________________________________________________PRIVATE_DATAMEMBERS
 private:
  T* ptr_ = nullptr;
  size_t size_ = 0;
};

//-------------------------------------------------------------------------------
// IMPLEMENTATION OF GPUBUFFER METHODS
//-------------------------------------------------------------------------------

template <class T>
inline GpuBuffer<T>::GpuBuffer(size_t N) {
  allocate(N);
}

template <class T>
inline GpuBuffer<T>::GpuBuffer(const std::vector<T>& other) {
  allocate(other.size());
  if (size_ > 0) {
    checkCudaError(cudaMemcpyAsync(ptr_, other.data(), size_ * sizeof(T),
                                   cudaMemcpyHostToDevice, getCudaStream()));
  }
}

template <class T>
inline GpuBuffer<T>::GpuBuffer(int N, T* data) {
  allocate(N);
  if (size_ > 0) {
    checkCudaError(cudaMemcpyAsync(ptr_, data, size_ * sizeof(T),
                                   cudaMemcpyHostToDevice, getCudaStream()));
  }
}

template <class T>
inline GpuBuffer<T>::GpuBuffer(const GpuBuffer& other) {
  allocate(other.size());
  if (size_ == 0)
    return;
  checkCudaError(cudaMemcpyAsync(ptr_, other.ptr_, size_ * sizeof(T),
                                 cudaMemcpyDeviceToDevice, getCudaStream()));
}

template <class T>
inline GpuBuffer<T>::GpuBuffer(GpuBuffer&& other) {
  recycle();
  ptr_ = other.ptr_;
  size_ = other.size_;
  other.ptr_ = nullptr;
  other.size_ = 0;
}

template <class T>
inline GpuBuffer<T>& GpuBuffer<T>::operator=(const GpuBuffer<T>& other) {
  allocate(other.size());
  if (size_ > 0) {
    checkCudaError(cudaMemcpyAsync(ptr_, other.ptr_, size_ * sizeof(T),
                                   cudaMemcpyDeviceToDevice, getCudaStream()));
  }
  return *this;
}

template <class T>
inline GpuBuffer<T>& GpuBuffer<T>::operator=(GpuBuffer<T>&& other) {
  recycle();
  ptr_ = other.ptr_;
  size_ = other.size_;
  other.ptr_ = nullptr;
  other.size_ = 0;
  return *this;
}

template <class T>
inline void GpuBuffer<T>::allocate(size_t size) {
  if (size_ == size)
    return;

  recycle();

  if (size > 0) {
    ptr_ = reinterpret_cast<T*>(memoryPool.allocate(size * sizeof(T)));
  } else {
    ptr_ = nullptr;
  }

  size_ = size;
}

template <class T>
inline void GpuBuffer<T>::recycle() {
  if (ptr_)
    memoryPool.recycle(reinterpret_cast<void**>(&ptr_));
  ptr_ = nullptr;
  size_ = 0;
}

template <class T>
inline T* GpuBuffer<T>::getHostCopy() const {
  T* data = new T[size_];
  if (size_ > 0) {
    checkCudaError(cudaMemcpyAsync(data, ptr_, size_ * sizeof(T),
                                   cudaMemcpyDeviceToHost, getCudaStream()));
  }
  return data;
}

template <class T>
inline std::vector<T> GpuBuffer<T>::getData() const {
  std::vector<T> vec(size_);
  if (size_ > 0) {
    checkCudaError(cudaMemcpyAsync(vec.data(), ptr_, size_ * sizeof(T),
                                   cudaMemcpyDeviceToHost, getCudaStream()));
  }
  return vec;
}

// Explicit bool specialization for GpuBuffer<bool>::getData().
// This specialization is needed because unlike othter element types,
// std::vector<bool> does not contain a contiguous array of elements. Hence, the
// boolean array data on the gpu can not directly copied into an
// std::vector<bool>. We need a temporary bool buffer to copy into.
template <>
inline std::vector<bool> GpuBuffer<bool>::getData() const {
  bool* buf = getHostCopy();
  std::vector<bool> data(size_);
  for (int i = 0; i < size_; i++) {
    data[i] = buf[i];
  }
  delete[] buf;
  return data;
}
