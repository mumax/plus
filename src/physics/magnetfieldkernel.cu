#include "cudalaunch.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "magnetfieldkernel.hpp"
#include "newell.hpp"

MagnetFieldKernel::MagnetFieldKernel(Grid grid, real3 cellsize)
    : cellsize_(cellsize), grid_(grid) {
  kernel_ = new Field(grid_, 6);
  compute();
}

MagnetFieldKernel::MagnetFieldKernel(Grid dst, Grid src, real3 cellsize)
    : cellsize_(cellsize), grid_(kernelGrid(dst, src)) {
  kernel_ = new Field(grid_, 6);
  compute();
}

MagnetFieldKernel::~MagnetFieldKernel() {
  delete kernel_;
}

__global__ void k_magnetFieldKernel(CuField kernel, real3 cellsize) {
  // printf("Started k_magnetFieldKernel. threadIdx %d\n", threadIdx.x);
  // if (!threadIdx.x) printf("Params %d, %d, %d\n", gridDim.x, blockDim.x, blockIdx.x);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // printf("Calculated ID = %d\n", idx);
  if (!kernel.cellInGrid(idx))
    return;

  // printf("Line 30, thread %d\n", idx);
  int3 coo = kernel.grid.index2coord(idx);
  // printf("Line 33, thread %d\n", idx);
  kernel.setValueInCell(idx, 0, calcNewellNxx(coo, cellsize));
  // printf("Line 35, thread %d\n", idx);
  kernel.setValueInCell(idx, 1, calcNewellNyy(coo, cellsize));
  kernel.setValueInCell(idx, 2, calcNewellNzz(coo, cellsize));
  kernel.setValueInCell(idx, 3, calcNewellNxy(coo, cellsize));
  // if (!threadIdx.x) printf("Line 36\n");
  kernel.setValueInCell(idx, 4, calcNewellNxz(coo, cellsize));
  kernel.setValueInCell(idx, 5, calcNewellNyz(coo, cellsize));
  // if (!threadIdx.x) printf("k_magnetFieldKernel success!!!\n");
}

void MagnetFieldKernel::compute() {
  printf("I am launching from magnetfieldkernel.cu\n");
  // printf("Grid %d, cellsize_ %f, %d, %f\n", grid_.ncells(), cellsize_.x, cellsize_.y, cellsize_.z);
  real3 cell_test;
  cell_test.x = 3.90625E-09F;
  cell_test.y = 3.90625E-09F;
  cell_test.z = 3E-09F;
  cudaLaunch(grid_.ncells(), k_magnetFieldKernel, kernel_->cu(), cell_test);
  printf("k_magnetFieldKernel launch was successful.\n");
}

Grid MagnetFieldKernel::grid() const {
  return grid_;
}
real3 MagnetFieldKernel::cellsize() const {
  return cellsize_;
}

const Field& MagnetFieldKernel::field() const {
  return *kernel_;
}

Grid MagnetFieldKernel::kernelGrid(Grid dst, Grid src) {
  int3 size = src.size() + dst.size() - int3{1, 1, 1};
  int3 origin = dst.origin() - (src.origin() + src.size() - int3{1, 1, 1});

  // add padding to get even dimensions if size is larger than 5
  // this will make the fft on this grid much more efficient
  int3 padding{0, 0, 0};
  if (size.x > 5 && size.x % 2 == 1)
    padding.x = 1;
  if (size.y > 5 && size.y % 2 == 1)
    padding.y = 1;
  if (size.z > 5 && size.z % 2 == 1)
    padding.z = 1;

  size += padding;
  origin -= padding;  // pad in front, this makes it easier to unpad after the
                      // convolution

  return Grid(size, origin);
}