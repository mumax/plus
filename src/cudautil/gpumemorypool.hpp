#pragma once
#include <map>
#include <vector>

/**
 * Gpu memory pool.
 *
 * The allocation of large memory blocks on the gpu is an heavy operation.
 * Using a memory pool avoids the need the allocate new memory on the gpu over
 * and over again by recycling the memory blocks.
 *
 * The GpuMemoryPool user is responsible for recycling or freeing the allocated
 * memory after use. If you need a gpu buffer, consider to use a GpuBuffer
 * object which does this kind of memory management for you.
 */
class GpuMemoryPool {
 public:
  GpuMemoryPool() = default;
  GpuMemoryPool(GpuMemoryPool const&) = delete;
  void operator=(GpuMemoryPool const&) = delete;

  ~GpuMemoryPool();

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
   * done by calling the 'free' or 'recycle' method on the GpuMemoryPool which
   * is used to allocate the gpu memory.
   */
  void* allocate(size_t size);

  /**
   * Free gpu memory allocated by this memory pool.
   */
  void free(void**);

  /**
   * Recycle gpu memory allocated by this memory pool.
   * The allocated memory will become available again in the memory pool.
   */
  void recycle(void**);

  /**
   * Prints the number of used and available allocate memory by this memory
   * pool. This function is useful to detect memory leaks.
   */
  void printInfo() const;

 private:
  std::map<size_t, std::vector<void*>> pool_;
  std::map<void*, size_t> inUse_;
};

/**
 * The global GpuMemoryPool instance
 */
extern GpuMemoryPool memoryPool;
