#include "cudaerror.hpp"
#include "cudalaunch.hpp"
#include "cudastream.hpp"
#include "field.hpp"
#include "gpubuffer.hpp"
#include "shift.hpp"

__global__ void k_calculateShiftDirection(int* result, CuField f) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx == 0) {
        Grid grid = f.system.grid;
        int3 center_idx = {
            grid.size().x / 2,
            grid.size().y / 2,
            grid.size().z / 2
        };
        real3 centerValue = f.vectorAt(center_idx);
        int dir = (centerValue.x < 0) ? -1 : 1;
        *result = dir;
    }
}

__global__ void k_shift_field(CuField field,
                              int dir,
                              real3 leftValue,
                              real3 rightValue) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (!field.cellInGeometry(idx)) { return; }

    Grid grid = field.system.grid;
    int3 src = grid.index2coord(idx);

    real3 val;
    src.x += dir;

    if (src.x >= 0 && src.x < grid.size().x && field.cellInGeometry(grid.coord2index(src)))
        val = field.vectorAt(src);
    else
        val = (dir == 1) ? leftValue : rightValue;
    field.setVectorInCell(idx, val);
}

template <typename T>
__global__ void k_shift_buffer(T* data, int ncells, int dir, T leftValue, T rightValue) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ncells) { return; }

    T val;
    int srcIdx = idx + dir;

    if (srcIdx >= 0 && srcIdx < ncells)
        val = data[srcIdx];
    else
        val = (dir == 1) ? leftValue : rightValue;
    data[idx] = val;
}

int calculateShiftDirection(const Field& field) {
    // TODO: this function only works properly if initial DW is centered.
    GpuBuffer<int> d_result(1);
    cudaLaunchReductionKernel(k_calculateShiftDirection, d_result.get(), field.cu());
    
    int result;
    checkCudaError(cudaMemcpyAsync(&result, d_result.get(), 1 * sizeof(int),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
    return result;
}

Field shift(const Field& field, int dir, int comp, real3 leftValue, real3 rightValue) {
    real av = field.average()[comp];
    real tolerance = 4 / field.grid().size().x;
    if (abs(av) > tolerance) {
        if (av > tolerance) { dir *= -1; }
        // TODO: remove these values:
        leftValue = real3{1,0,0};
        rightValue = real3{-1,0,0};
        cudaLaunch(field.grid().ncells(), k_shift_field, field.cu(), dir, leftValue, rightValue);
    }
    return field;
}

template <typename T>
GpuBuffer<T> shift(const GpuBuffer<T>& data, int direction, int ncells, T left, T right) {
    cudaLaunch(ncells, k_shift_buffer, data.get(), ncells, direction, left, right);   
    return data;
}

// Explicit instantiations
template GpuBuffer<bool> shift<bool>(const GpuBuffer<bool>&, int, int, bool, bool);
template GpuBuffer<unsigned int> shift<unsigned int>(const GpuBuffer<unsigned int>&, int, int, unsigned int, unsigned int);
