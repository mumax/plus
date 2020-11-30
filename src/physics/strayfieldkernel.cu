#include <memory>

#include "cudalaunch.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "newell.hpp"
#include "strayfieldkernel.hpp"
#include "system.hpp"
#include "world.hpp"

StrayFieldKernel::StrayFieldKernel(Grid grid, const World* world) {
  kernel_ = std::make_unique<Field>(std::make_shared<System>(world, grid), 6);
  compute();
}

StrayFieldKernel::StrayFieldKernel(Grid dst, Grid src, const World* world)
    : StrayFieldKernel(kernelGrid(dst, src), world) {}

StrayFieldKernel::~StrayFieldKernel() {}

std::shared_ptr<const System> StrayFieldKernel::kernelSystem() const {
  return kernel_->system();
}

__global__ void k_strayFieldKernel(CuField kernel, real3 cellsize) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!kernel.cellInGrid(idx))
    return;
  int3 coo = kernel.grid.index2coord(idx);
  kernel.setValueInCell(idx, 0, calcNewellNxx(coo, cellsize));
  kernel.setValueInCell(idx, 1, calcNewellNyy(coo, cellsize));
  kernel.setValueInCell(idx, 2, calcNewellNzz(coo, cellsize));
  kernel.setValueInCell(idx, 3, calcNewellNxy(coo, cellsize));
  kernel.setValueInCell(idx, 4, calcNewellNxz(coo, cellsize));
  kernel.setValueInCell(idx, 5, calcNewellNyz(coo, cellsize));
}

void StrayFieldKernel::compute() {
  cudaLaunch(grid().ncells(), k_strayFieldKernel, kernel_->cu(), cellsize());
}

Grid StrayFieldKernel::grid() const {
  return kernelSystem()->grid();
}
real3 StrayFieldKernel::cellsize() const {
  return kernelSystem()->cellsize();
}

const Field& StrayFieldKernel::field() const {
  return *kernel_;
}

Grid StrayFieldKernel::kernelGrid(Grid dst, Grid src) {
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
