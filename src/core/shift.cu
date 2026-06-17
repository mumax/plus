#include "cudaerror.hpp"
#include "cudalaunch.hpp"
#include "cudastream.hpp"
#include "field.hpp"
#include "shift.hpp"

__global__ void k_getSignOfOuterValue(int* result, CuField f, int comp, int axis) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx == 0) {
        Grid grid = f.system.grid;
        int3 coo = {grid.size().x / 2,
                    grid.size().y / 2,
                    grid.size().z / 2};
        (&coo.x)[axis] = 0;

        // TODO: check if coo in geometry? What if not?
        real3 value = f.vectorAt(coo);
        int sign = ((&value.x)[comp] < 0) ? -1 : 1;
        *result = sign;
    }
}

__global__ void k_shift_field(CuField result,
                              CuField field,
                              int dir,
                              int axis,
                              real3 leftValue,
                              real3 rightValue) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (!field.cellInGeometry(idx)) { return; }

    Grid grid = field.system.grid;
    int3 gridsize = grid.size();

    real3 value;

    const int3 src = grid.index2coord(idx);

    int3 direction{0, 0, 0};
    (&direction.x)[axis] = dir;

    int3 dst = src - direction;

    if ((&dst.x)[axis] >= 0 && (&dst.x)[axis] < (&gridsize.x)[axis] && field.cellInGeometry(grid.coord2index(dst)))
        value = field.vectorAt(dst);
    else {
        if (dir == 1)
            value = (leftValue != real3{0, 0, 0}) ? leftValue : field.vectorAt(src);
        else
            value = (rightValue != real3{0, 0, 0}) ? rightValue : field.vectorAt(src);
    }
    result.setVectorInCell(idx, value);
}

real sgn(real value) {
    return (value > 0.) ? 1. : -1.;
}

int calculateShiftDirection(const Field& field, real3 leftValue, real3 rightValue, int comp, int axis) {
    real av = field.average()[comp];
    int3 gridsize = field.grid().size();
    real tolerance = 4.0 / (&gridsize.x)[axis];

    // If left insertion value is absent, deduce sign of left domain
    if (leftValue == real3{0,0,0}) {
        GpuBuffer<int> d_result(1);
        cudaLaunchReductionKernel(k_getSignOfOuterValue, d_result.get(), field.cu(), comp, axis);

        int result;
        checkCudaError(cudaMemcpyAsync(&result, d_result.get(), 1 * sizeof(int),
                                       cudaMemcpyDeviceToHost, getCudaStream()));
        (&leftValue.x)[comp] = real(result);
    }
    if (abs(av) > tolerance)
        return - sgn((&leftValue.x)[comp]) * sgn(av);
    else
        return 0;
}

Field shift(const Field& field, int dir, int axis, int comp, real3 leftValue, real3 rightValue) {
    Field result(field.system(), 3);
    cudaLaunch(field.grid().ncells(), k_shift_field, result.cu(), field.cu(), dir, axis, leftValue, rightValue);
    return result;
}