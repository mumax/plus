// clang-format off
#pragma once

#include <cuda_runtime_api.h>
#include <stdio.h>
#include <string>

#define checkCudaError(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    const char* v = std::getenv("GPU_ABORT");
    abort = abort && !(v && std::string(v) == "OFF");
    if (abort)
      exit(code);
  }
}
