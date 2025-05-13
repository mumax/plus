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

__global__ void k_shift_field(CuField result,
                              CuField field,
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
        val = (dir == 1) ? rightValue : leftValue;
    result.setVectorInCell(idx, val);
}

template <typename T>
__global__ void k_shift_buffer(T* result, T* data, int ncells, int Nx, int Ny, int dir, T leftValue, T rightValue) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Nx * Ny) return;

    int row = idx / Nx;
    int col = idx % Nx;

    int srcCol = col + dir;
    int srcIdx = row * Nx + srcCol;
    int dstIdx = row * Nx + col;

    if (srcCol >= 0 && srcCol < Nx)
        result[dstIdx] = data[srcIdx];
    else
        result[dstIdx] = (dir == 1) ? leftValue : rightValue;
}

int calculateShiftDirection(const Field& field, int comp) {
    real av = field.average()[comp];
    real tolerance = 4.0 / field.grid().size().x;
    if (abs(av) > tolerance) {
        GpuBuffer<int> d_result(1);
        cudaLaunchReductionKernel(k_calculateShiftDirection, d_result.get(), field.cu());

        int result;
        checkCudaError(cudaMemcpyAsync(&result, d_result.get(), 1 * sizeof(int),
                                    cudaMemcpyDeviceToHost, getCudaStream()));
        if (av < tolerance) { return result * (-1); }
        return result;
    }
    return 0;
}

Field shift(const Field& field, int dir, int comp, real3 leftValue, real3 rightValue) {
    Field result(field.system(), 3);
    cudaLaunch(field.grid().ncells(), k_shift_field, result.cu(), field.cu(), dir, leftValue, rightValue);
    return result;
}

template <typename T>
GpuBuffer<T> shift(const GpuBuffer<T>& data, int direction, int ncells, int nx, int ny, T left, T right) {
    GpuBuffer<T> result(data.size());
    cudaLaunch(ncells, k_shift_buffer, result.get(), data.get(), ncells, nx, ny, direction, left, right);
    cudaStreamSynchronize(getCudaStream());

    return result;
}

// Explicit instantiations
template GpuBuffer<bool> shift<bool>(const GpuBuffer<bool>&, int, int, int, int, bool, bool);
template GpuBuffer<unsigned int> shift<unsigned int>(const GpuBuffer<unsigned int>&, int, int, int, int, unsigned int, unsigned int);
