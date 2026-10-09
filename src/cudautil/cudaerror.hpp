#pragma once

#include <cuda_runtime_api.h>
#include <stdio.h>

#ifndef GPU_ABORT_DEFAULT
#define GPU_ABORT_DEFAULT true
#endif

#define checkCudaError(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code,
                      const char* file,
                      int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    const char* v = std::getenv("GPU_ABORT");
    bool quit = abort && !(v && v[0] == '0');
    if (quit)
      exit(code);
  }
}
