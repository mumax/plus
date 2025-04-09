#include <memory>

#include "cudalaunch.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "newell.hpp"
#include "demagasymptotic.hpp"
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

__global__ void k_strayFieldKernel(CuField kernel, const Grid mastergrid,
                                   const int3 pbcRepetitions) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!kernel.cellInGrid(idx))
    return;

  const real3 cellsize = kernel.system.cellsize;
  int3 coo = kernel.system.grid.index2coord(idx);
  real Nxx = 0, Nyy = 0, Nzz = 0, Nxy = 0, Nxz = 0, Nyz = 0;
  
  // pbcRepetitions.c should not be > 0 if mastergrid.size().c == 0
  // but no sanity check here, World should keep track of such things
  for (int i = -pbcRepetitions.x; i <= pbcRepetitions.x; i++) {
    for (int j = -pbcRepetitions.y; j <= pbcRepetitions.y; j++) {
      for (int k = -pbcRepetitions.z; k <= pbcRepetitions.z; k++) {
        int3 coo_ = coo + (int3{i,j,k} * mastergrid.size());

        // Assume the following errors on the analytical and asymptotic result
        // E_analytic = eps R³/V
        // E_asymptotic = V R²/(5(R²-dmax²)) dmax^(n-3)/R^(n)
        // Here V is dx*dy*dz, dmax = max(dx,dy,dz), n is the order of asymptote
        // and eps is a constant determined by trial and error.
        // Use the analytical model when
        // E_analytic / E_asymptotic < 1

        // Todo: make the order variable?
        // Todo: switch to O(R⁻³) when far away?

        double x = coo_.x * cellsize.x;
        double y = coo_.y * cellsize.y;
        double z = coo_.z * cellsize.z;
        double R = sqrtf(x*x + y*y + z*z);
        double V = cellsize.x * cellsize.y * cellsize.z;
        double h = fmax(cellsize.x,fmax(cellsize.y,cellsize.z));
        
        if (5 * 1e-10 * (R*R - h*h)/(V*V) * pow(R,11+1)/pow(h,11-3) < 1) {
          Nxx += calcNewellNxx(coo_, cellsize);
          Nyy += calcNewellNyy(coo_, cellsize);
          Nzz += calcNewellNzz(coo_, cellsize);
          Nxy += calcNewellNxy(coo_, cellsize);
          Nxz += calcNewellNxz(coo_, cellsize);
          Nyz += calcNewellNyz(coo_, cellsize);
        } else {
          Nxx += calcAsymptoticNxx(coo_, cellsize);
          Nyy += calcAsymptoticNyy(coo_, cellsize);
          Nzz += calcAsymptoticNzz(coo_, cellsize);
          Nxy += calcAsymptoticNxy(coo_, cellsize);
          Nxz += calcAsymptoticNxz(coo_, cellsize);
          Nyz += calcAsymptoticNyz(coo_, cellsize);
        }
      }
    }
  }
  kernel.setValueInCell(idx, 0, Nxx);
  kernel.setValueInCell(idx, 1, Nyy);
  kernel.setValueInCell(idx, 2, Nzz);
  kernel.setValueInCell(idx, 3, Nxy);
  kernel.setValueInCell(idx, 4, Nxz);
  kernel.setValueInCell(idx, 5, Nyz);
}

void StrayFieldKernel::compute() {
  cudaLaunch(grid().ncells(), k_strayFieldKernel, kernel_->cu(),
             mastergrid(), pbcRepetitions());
}

Grid StrayFieldKernel::grid() const {
  return kernelSystem()->grid();
}
Grid StrayFieldKernel::mastergrid() const {
  return kernelSystem()->world()->mastergrid();
}
real3 StrayFieldKernel::cellsize() const {
  return kernelSystem()->cellsize();
}
const int3 StrayFieldKernel::pbcRepetitions() const {
  return kernelSystem()->world()->pbcRepetitions();
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
