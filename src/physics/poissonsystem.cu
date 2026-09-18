#include "conductivitytensor.hpp"
#include "cudalaunch.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "linsolver.hpp"
#include "linsystem.hpp"
#include "poissonsystem.hpp"
#include "quantityevaluator.hpp"
#include "reduce.hpp"

/** Represent a sparce matrix row with max N non zero elements. */
class Row {
  /** Maximal non zero values in a sparse row.
   *  1 center cell + 6 nearest neighbors + 12 next nearest neighbors
   */
  static const int N = 19;

 public:
  real b; /** Right hand sight of this row in Ax=b */

 private:
  int rowidx;     /** Row index of this sparse row  */
  real value_[N]; /** matrix values in this sparse row */
  int colidx_[N]; /** column indices of the matrix values in this sparse row */

 public:
  /** Construct a row of a sparse linear system of equations */
  __device__ Row(int rowidx, CuSystem system) : b(0), rowidx(rowidx) {
    const int3 coo = system.grid.index2coord(rowidx);
    for (int ix = -1; ix <= 1; ix++) {
      for (int iy = -1; iy <= 1; iy++) {
        for (int iz = -1; iz <= 1; iz++) {
          if (ix * ix + iy * iy + iz * iz > 2)  // only include (next) nearest
            continue;

          int3 rcoo{ix, iy, iz};
          value(rcoo) = 0.0;
          if (system.inGeometry(coo + rcoo)) {
            colidx(rcoo) = system.grid.coord2index(coo + rcoo);
          } else {
            colidx(rcoo) = -1;
          }
        }
      }
    }
  }

  /** Column index for neighbor with relative position rcoo.
   *  If neighbor is outside the geometry, return -1.
   */
  __device__ int& colidx(int3 rcoo) { return colidx_[neighborId(rcoo)]; }
  __device__ const int& colidx(int3 rcoo) const { return colidx_[neighborId(rcoo)]; }

  /** Matrix value for neighbor with relative position rcoo. */
  __device__ real& value(int3 rcoo) { return value_[neighborId(rcoo)]; }
  __device__ const real& value(int3 rcoo) const { return value_[neighborId(rcoo)]; }

  /** Return true if neighbor is in geometry. */
  __device__ bool inGeometry(int3 rcoo) const { return colidx_[neighborId(rcoo)] >= 0; }

  /** Add finite difference in the matrix row. */
  __device__ void addDiff(int3 rcoo1, int3 rcoo2, real val) {
    if (inGeometry(rcoo1) && inGeometry(rcoo2)) {
      value(rcoo1) += val;
      value(rcoo2) -= val;
    }
  }

  /** Fill in the row in the linear system linsys. */
  __device__ void writeRowInLinearSystem(LinearSystem::CuData* linsys) const {
    // Set the right hand side of the row in the system of linear equations
    linsys->b[rowidx] = b;

    // Fill in the non zero matrix values
    int c = 0;
    for (int k = 0; k < N; k++) {
      if (value_[k] != 0.0) {  // element is non zero
        linsys->matrixIdx(rowidx, c) = colidx_[k];
        linsys->matrixVal(rowidx, c) = value_[k] / value({0, 0, 0});
        c++;
      }
    }

    // Invalidate other available places by setting an invalid column index.
    for (; c < linsys->nnz; c++)
      linsys->matrixIdx(rowidx, c) = -1;
  }

 private:
  /** Return local index of neighbor (-1 if rcoo does not point to neighbor). */
  __device__ int constexpr neighborId(int3 rcoo) const {
    // clang-format off
    int id = -1;
    if      (rcoo.x == 0 && rcoo.y == 0 && rcoo.z == 0 ) id =  0; // center
    else if (rcoo.x ==-1 && rcoo.y == 0 && rcoo.z == 0 ) id =  1; // nearest
    else if (rcoo.x == 1 && rcoo.y == 0 && rcoo.z == 0 ) id =  2; //  |
    else if (rcoo.x == 0 && rcoo.y ==-1 && rcoo.z == 0 ) id =  3; //  |
    else if (rcoo.x == 0 && rcoo.y == 1 && rcoo.z == 0 ) id =  4; //  |
    else if (rcoo.x == 0 && rcoo.y == 0 && rcoo.z ==-1 ) id =  5; //  |
    else if (rcoo.x == 0 && rcoo.y == 0 && rcoo.z == 1 ) id =  6; //  |
    else if (rcoo.x ==-1 && rcoo.y ==-1 && rcoo.z == 0 ) id =  7; // next nearest
    else if (rcoo.x ==-1 && rcoo.y == 1 && rcoo.z == 0 ) id =  8; //  |
    else if (rcoo.x == 1 && rcoo.y ==-1 && rcoo.z == 0 ) id =  9; //  |
    else if (rcoo.x == 1 && rcoo.y == 1 && rcoo.z == 0 ) id = 10; //  |
    else if (rcoo.x ==-1 && rcoo.y == 0 && rcoo.z ==-1 ) id = 11; //  |
    else if (rcoo.x ==-1 && rcoo.y == 0 && rcoo.z == 1 ) id = 12; //  |
    else if (rcoo.x == 1 && rcoo.y == 0 && rcoo.z ==-1 ) id = 13; //  |
    else if (rcoo.x == 1 && rcoo.y == 0 && rcoo.z == 1 ) id = 14; //  |
    else if (rcoo.x == 0 && rcoo.y ==-1 && rcoo.z ==-1 ) id = 15; //  |
    else if (rcoo.x == 0 && rcoo.y ==-1 && rcoo.z == 1 ) id = 16; //  |
    else if (rcoo.x == 0 && rcoo.y == 1 && rcoo.z ==-1 ) id = 17; //  |
    else if (rcoo.x == 0 && rcoo.y == 1 && rcoo.z == 1 ) id = 18; //  |
    return id;
    // clang-format on
  }
};

//------------------------------------------------------------------------------

PoissonSystem::PoissonSystem(const Ferromagnet* magnet) : magnet_(magnet), solver_() {}

void PoissonSystem::init() {
  solver_.setSystem(construct());
}

__global__ static void k_construct(LinearSystem::CuData linsys,
                                   const CuField conductivity,
                                   const CuParameter pot) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  CuSystem system = pot.system;

  // if outside the grid, then there is nothing to do, and we can return early
  if (!system.grid.cellInGrid(idx))
    return;

  Row row(idx, system);

  if (!system.inGeometry(idx) || conductivity.valueAt(idx) == real(0.0) ||
      !isnan(pot.valueAt(idx))) {
    // Put 1 on the diagonal
    row.colidx({0, 0, 0}) = idx;
    row.value({0, 0, 0}) = 1.0;
    // set applied potential as rhs (or 0 if there is no applied potential)
    row.b = isnan(pot.valueAt(idx)) ? 0.0 : pot.valueAt(idx);

  } else {  // NON TRIVIAL MATRIX ROW

    bool anisotropic = conductivity.ncomp > 1;

    // Return the average conductivity of two cells
    auto avgConductivity = [&conductivity](int idx1, int idx2, int comp) {
      real c1 = conductivity.valueAt(idx1, comp);
      real c2 = conductivity.valueAt(idx2, comp);
      return sqrt(c1 * c2);
    };

    const real dx = system.cellsize.x;
    const real dy = system.cellsize.y;
    const real dz = system.cellsize.z;

    // current along x direction
    for (int ix : {-1, 1}) {
      if (row.inGeometry({ix, 0, 0})) {
        int idx_ = row.colidx({ix, 0, 0});
        real crosssection = dy * dz;

        {  // Isotropic part
          real conductivityXX = avgConductivity(idx, idx_, 0);
          real fac = crosssection * conductivityXX / dx;
          row.addDiff({0, 0, 0}, {ix, 0, 0}, fac);
        }

        if (anisotropic) {
          real conductivityXY = avgConductivity(idx, idx_, 3);
          real fac_dy = -ix * crosssection * conductivityXY / dy / 4;
          // clang-format off
          row.addDiff({ 0,  0,  0}, { 0, -1,  0}, fac_dy);
          row.addDiff({ix,  0,  0}, {ix, -1,  0}, fac_dy);
          row.addDiff({ 0,  1,  0}, { 0,  0,  0}, fac_dy);
          row.addDiff({ix,  1,  0}, {ix,  0,  0}, fac_dy);
          // clang-format on

          real conductivityXZ = avgConductivity(idx, idx_, 4);
          real fac_dz = -ix * crosssection * conductivityXZ / dz / 4;
          // clang-format off
          row.addDiff({ 0,  0,  0}, { 0,  0, -1}, fac_dz);
          row.addDiff({ix,  0,  0}, {ix,  0, -1}, fac_dz);
          row.addDiff({ 0,  0,  1}, { 0,  0,  0}, fac_dz);
          row.addDiff({ix,  0,  1}, {ix,  0,  0}, fac_dz);
          // clang-format on
        }
      }
    }

    // current along y direction
    for (int iy : {-1, 1}) {
      if (row.inGeometry({0, iy, 0})) {
        int idx_ = row.colidx({0, iy, 0});
        real crosssection = dx * dz;

        {  // Isotropic part
          real conductivityYY = avgConductivity(idx, idx_, anisotropic ? 1 : 0);
          real fac = crosssection * conductivityYY / dy;
          row.addDiff({0, 0, 0}, {0, iy, 0}, fac);
        }

        if (anisotropic) {  // clang-format off
          real conductivityXY = avgConductivity(idx, idx_, 3);
          real fac_dx = -iy * crosssection * conductivityXY  / dx / 4;
          row.addDiff({ 0,  0,  0}, {-1,  0,  0}, fac_dx);
          row.addDiff({ 0, iy,  0}, {-1, iy,  0}, fac_dx);
          row.addDiff({ 1,  0,  0}, { 0,  0,  0}, fac_dx);
          row.addDiff({ 1, iy,  0}, { 0, iy,  0}, fac_dx);

          real conductivityYZ = avgConductivity(idx, idx_, 5);
          real fac_dz = -iy * crosssection * conductivityYZ / dz / 4;
          row.addDiff({ 0,  0,  0}, { 0,  0, -1}, fac_dz);
          row.addDiff({ 0, iy,  0}, { 0, iy, -1}, fac_dz);
          row.addDiff({ 0,  0,  1}, { 0,  0,  0}, fac_dz);
          row.addDiff({ 0, iy,  1}, { 0, iy,  0}, fac_dz);
        }  // clang-format on
      }
    }

    // current along z direction
    for (int iz : {-1, 1}) {
      if (row.inGeometry({0, 0, iz})) {
        int idx_ = row.colidx({0, 0, iz});
        real crosssection = dx * dy;

        {  // Isotropic part
          real conductivityZZ = avgConductivity(idx, idx_, anisotropic ? 2 : 0);
          real fac = crosssection * conductivityZZ / dz;
          row.addDiff({0, 0, 0}, {0, 0, iz}, fac);
        }

        if (anisotropic) {  // clang-format off
          real conductivityXZ = avgConductivity(idx, idx_, 4);
          real fac_dx = -iz * crosssection * conductivityXZ  / dx / 4;
          row.addDiff({ 0,  0,  0}, {-1,  0,  0}, fac_dx);
          row.addDiff({ 0,  0, iz}, {-1,  0, iz}, fac_dx);
          row.addDiff({ 1,  0,  0}, { 0,  0,  0}, fac_dx);
          row.addDiff({ 1,  0, iz}, { 0,  0, iz}, fac_dx);

          real conductivityYZ = avgConductivity(idx, idx_, 5);
          real fac_dy = -iz * crosssection * conductivityYZ / dy / 4;
          row.addDiff({ 0,  0,  0}, { 0, -1,  0}, fac_dy);
          row.addDiff({ 0,  0, iz}, { 0, -1, iz}, fac_dy);
          row.addDiff({ 0,  1,  0}, { 0,  0,  0}, fac_dy);
          row.addDiff({ 0,  1, iz}, { 0,  0, iz}, fac_dy);
        }  // clang-format on
      }
    }

    row.b = 0.0;
  }

  row.writeRowInLinearSystem(&linsys);
}

LinearSystem PoissonSystem::construct() const {
  Grid grid = magnet_->grid();
  bool threeDimenstional = grid.size().z > 1;
  bool anisotropic = !magnet_->amrRatio.assuredZero();

  int maxNonZeros;
  Field conductivity;

  if (anisotropic) {
    conductivity = evalConductivityTensor(magnet_);  // 6 components!
    maxNonZeros = threeDimenstional ? 19 : 9;
  } else {
    conductivity = magnet_->conductivity.eval();  // only 1 component
    maxNonZeros = threeDimenstional ? 7 : 5;
  }

  LinearSystem system(grid.ncells(), maxNonZeros);

  cudaLaunch(grid.ncells(), k_construct, system.cu(), conductivity.cu(),
             magnet_->appliedPotential.cu());

  return system;
}

__global__ static void k_putSolutionInField(CuField f, lsReal* y) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!f.cellInGeometry(idx))
    return;
  f.setValueInCell(idx, 0, (real)y[idx]);
}

Field PoissonSystem::solve() {
  init();
  GVec y = solver_.solve();
  Field pot(magnet_->system(), 1);
  cudaLaunch(pot.grid().ncells(), k_putSolutionInField, pot.cu(), y.get());
  return pot;
}
