#include "cudaerror.hpp"
#include "cudalaunch.hpp"
#include "cudastream.hpp"
#include "field.hpp"
#include "shift.hpp"

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

    const int3 dst = grid.index2coord(idx);

    int3 direction{0, 0, 0};
    (&direction.x)[axis] = dir;

    int3 src = dst - direction;

    // TODO: this only works because non-trivial geometries are not allowed at this point
    if (field.cellInGeometry(src))
        value = field.vectorAt(src);
    else {
        if (dir == 1)
            value = (leftValue != real3{0, 0, 0}) ? leftValue : field.vectorAt(dst);
        else
            value = (rightValue != real3{0, 0, 0}) ? rightValue : field.vectorAt(dst);
    }
    result.setVectorInCell(idx, value);
}

real sgn(real value) {
    return (value > 0.) ? 1. : -1.;
}

int calculateShiftDirection(const Field& field, int comp, int axis, real3 leftValue, real3 rightValue) {
    real av = field.average()[comp];
    int3 gridsize = field.grid().size();
    real tolerance = 4.0 / (&gridsize.x)[axis];

    if (abs(av) > tolerance) {
    // If left insertion value is absent, deduce sign of 'left' domain
        if (leftValue == real3{0,0,0}) {
            auto grid = field.grid();
            int3 coo = {grid.size().x / 2, grid.size().y / 2, grid.size().z / 2};
            (&coo.x)[axis] = 0;

            // TODO: What if coo not in geometry?
            int idx = grid.coord2index(coo);
            real value;
            checkCudaError(cudaMemcpy(&value, field.device_ptr(comp) + idx,
                                       sizeof(real), cudaMemcpyDeviceToHost));
            (&leftValue.x)[comp] = value;
        }
        return - sgn((&leftValue.x)[comp]) * sgn(av);
    }
    else {
        return 0;
    }
}

Field shift(const Field& field, int dir, int comp, int axis, real3 leftValue, real3 rightValue) {
    Field result(field.system(), 3);
    cudaLaunch(field.grid().ncells(), k_shift_field, result.cu(), field.cu(), dir, axis, leftValue, rightValue);
    return result;
}