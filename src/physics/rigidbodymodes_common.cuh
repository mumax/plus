#include "cudalaunch.hpp"
#include "rigidbodymodes_common.hpp"

// Multiplier on the occupancy-derived block cap (numSMs * maxActiveBlocksPerSM).
constexpr int REDUCTION_OCCUPANCY_MULTIPLIER = 1;

template <typename Kernel>
int occupancyMaxBlocks(Kernel kernel, int blockDim, size_t sharedMemBytes = 0) {
  static int cached = -1;
  if (cached < 0) {
    int device;
    checkCudaError(cudaGetDevice(&device));
    int smCount;
    checkCudaError(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));

    int maxActiveBlocksPerSm = 0;
    checkCudaError(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerSm, kernel, blockDim, sharedMemBytes));
    cached = smCount * maxActiveBlocksPerSm * REDUCTION_OCCUPANCY_MULTIPLIER;
  }
  return cached;
}

//sensible choice of number of blocks
template <typename Kernel>
int numReductionBlocks(int ncells, Kernel kernel) {
  int n = (ncells + BLOCKDIM - 1) / BLOCKDIM;
  if (n < 1){
    n = 1;
  }
  int maxBlocks = occupancyMaxBlocks(kernel, BLOCKDIM);
  if (maxBlocks > 0 && n > maxBlocks){ 
    n = maxBlocks;
  }
  return n;
}


// T is deduced from the sdata array's element type
template <int NFIELDS, typename T>
__device__ inline void blockReduceSum(T sdata, int tid) {
  for (unsigned int s = BLOCKDIM / 2; s > 0; s >>= 1) {
    if (tid < s) {
      #pragma unroll
      for (int f = 0; f < NFIELDS; f++)
        sdata[f][tid] += sdata[f][tid + s];
    }
    __syncthreads();
  }
}

// Warp-shuffle, no risk of divergence
//warps are 32 threads on all current CUDA GPUs
constexpr int warp_size = 32;
template <typename T, int WIDTH = 32>
__device__ __forceinline__ T warpReduceSum(T val) {
  static_assert(WIDTH > 0 && (WIDTH & (WIDTH - 1)) == 0, "WIDTH must be a power of two");
  #pragma unroll
  for (int offset = WIDTH / 2; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffffu, val, offset, WIDTH);
  return val;
}

template <typename T>
__device__ __forceinline__ void kahanAdd(T& sum, T& c, T x) {
  T y = x - c;
  T t = sum + y;
  c = (t - sum) - y;
  sum = t;
}

// single block 2-level reduction, 8 warps reduce once, then warp0 reduces again
template <typename T, int N_accums>
__device__ __forceinline__ void blockReduceSum(T (&accumulators)[N_accums]) {
  #pragma unroll
  for (int accumIdx = 0; accumIdx < N_accums; accumIdx++)
    accumulators[accumIdx] = warpReduceSum<T, warp_size>(accumulators[accumIdx]);

  static_assert(BLOCKDIM % warp_size == 0, "assumes full warps");
  constexpr int NUM_WARPS = BLOCKDIM / warp_size;
  __shared__ T sdata[N_accums][NUM_WARPS];

  int lane = threadIdx.x % warpSize;
  int warpId = threadIdx.x / warpSize;
  if (lane == 0) {
    #pragma unroll
    for (int f = 0; f < N_accums; f++)
      sdata[f][warpId] = accumulators[f];
  }
  __syncthreads();

  if (warpId == 0) {
    #pragma unroll
    for (int f = 0; f < N_accums; f++) {
      T v = (lane < NUM_WARPS) ? sdata[f][lane] : T(0);
      accumulators[f] = warpReduceSum<T, NUM_WARPS>(v);
    }
  }
}

//pass 1: center of mass (COM)
// level 1: grid-stride load + two-level warp-shuffle tree reduce, one partial per block.
// level 2: single block combines the level-1 partials in double, same warp-shuffle tree.

struct ComAccum { real wrx, wry, wrz, w; };
struct ComAccumD { double wrx_d, wry_d, wrz_d, w_d; };

// level 1
static __global__ void k_comSumsPartial(ComAccum* blockPartials, CuSystem system,
                                 CuParameter rho, bool unweighted) {
  int ncells = system.grid.ncells();
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  real accum4[4] = {0, 0, 0, 0};  // w*r.x, w*r.y, w*r.z, w
  real comp4[4] = {0, 0, 0, 0};   // Kahan compensation terms

  for (int i = gid; i < ncells; i += stride) {
    if (!system.inGeometry(i)){
      continue;
    }
    real w;
    if (unweighted) {
      w = real(1.0);
    } else {
      w = rho.valueAt(i);
    }
    real3 r = cellPositionDevice(system, i);
    kahanAdd(accum4[0], comp4[0], w * r.x);
    kahanAdd(accum4[1], comp4[1], w * r.y);
    kahanAdd(accum4[2], comp4[2], w * r.z);
    kahanAdd(accum4[3], comp4[3], w);
  }

  blockReduceSum<real, 4>(accum4);

  if (threadIdx.x == 0){
    blockPartials[blockIdx.x] = ComAccum{accum4[0], accum4[1], accum4[2], accum4[3]};
  }
}

// level 2: single block, double accumulation
static __global__ void k_comSumsCombineDouble(ComAccumD* out, const ComAccum* blockPartials,
                                       int numPartials) {
  double accum4_out[4] = {0, 0, 0, 0};

  for (int i = threadIdx.x; i < numPartials; i += blockDim.x) {
    ComAccum p = blockPartials[i];
    accum4_out[0] += p.wrx; accum4_out[1] += p.wry; accum4_out[2] += p.wrz; accum4_out[3] += p.w;
  }

  blockReduceSum<double, 4>(accum4_out);

  if (threadIdx.x == 0){
    *out = ComAccumD{accum4_out[0], accum4_out[1], accum4_out[2], accum4_out[3]};
  }
}

// Host launcher
inline ComAccumD computeComSums(const CuSystem& cusys, const CuParameter& rho, int ncells, bool unweighted) {
  int numBlocks = numReductionBlocks(ncells, k_comSumsPartial);

  GpuBuffer<ComAccum> d_partials(numBlocks);
  GpuBuffer<ComAccumD> d_accum(1);

  k_comSumsPartial<<<numBlocks, BLOCKDIM, 0, getCudaStream()>>>(d_partials.get(), cusys, rho, unweighted);
  checkCudaError(cudaPeekAtLastError());

  k_comSumsCombineDouble<<<1, BLOCKDIM, 0, getCudaStream()>>>(d_accum.get(), d_partials.get(), numBlocks);
  checkCudaError(cudaPeekAtLastError());

  ComAccumD result;
  checkCudaError(cudaMemcpyAsync(&result, d_accum.get(), sizeof(ComAccumD),cudaMemcpyDeviceToHost, getCudaStream()));
  checkCudaError(cudaStreamSynchronize(getCudaStream()));
  return result;
}

inline ComResult computeCom(const CuSystem& cusys, const CuParameter& rho, int ncells, bool unweighted) {
  ComAccumD com_sums = computeComSums(cusys, rho, ncells, unweighted);

  //actually summed rho
  real weight = real(com_sums.w_d);
  if (weight <= real(0.0)){
    throw std::runtime_error("computeRigidBodyGeometry: total mass (sum of rho) in geometry is zero or negative.");
  }

  return ComResult{real3{real(com_sums.wrx_d), real(com_sums.wry_d), real(com_sums.wrz_d)} / weight, weight};
}
