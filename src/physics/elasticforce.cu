#include "elastodynamics.hpp"
#include "elasticforce.hpp"
#include "cudalaunch.hpp"
#include "magnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "stresstensor.hpp"


__device__ int tensorComp(int row, int col) {
  return (row == col) ? row : row+col+2;
}

__device__ int3 tensorRowComps(int row) {
  return int3{tensorComp(row, 0), tensorComp(row, 1), tensorComp(row, 2)};
}

// Numerical divergence of stress with central five-point stencil in bulk material.
// Lower order accuracy (three-point stencil) central difference is used in bulk
// or 1 cell away from boundary.
// At the boundary, traction-free bondary conditions are implemented, inspired
// by a three-point stencil with step size h equal to half cellsize.
__global__ void k_elasticForce(CuField fField,
                               const CuField stressTensor,
                               const real3 w,  // 1 / 2*cellsize
                               const Grid mastergrid) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = fField.system;
  const Grid grid = system.grid;

  // When outside the geometry, set to zero and return early
  if (!system.inGeometry(idx)) {
    if (grid.cellInGrid(idx)) {
      fField.setVectorInCell(idx, real3{0, 0, 0});
    }
    return;
  }

  // array instead of real3 to get indexing [i]
  const real ws[3] = {w.x, w.y, w.z};
  const int3 im2_arr[3] = {int3{-2, 0, 0}, int3{0,-2, 0}, int3{0, 0,-2}};
  const int3 im1_arr[3] = {int3{-1, 0, 0}, int3{0,-1, 0}, int3{0, 0,-1}};
  const int3 ip1_arr[3] = {int3{ 1, 0, 0}, int3{0, 1, 0}, int3{0, 0, 1}};
  const int3 ip2_arr[3] = {int3{ 2, 0, 0}, int3{0, 2, 0}, int3{0, 0, 2}};
  const int3 coo = grid.index2coord(idx);
    
  real3 f = {0, 0, 0};  // elastic force vector
  for (int i = 0; i < 3; i++) {
    // i is {x, y, z} derivative direction and stress tensor row
    // f_j = ∂i σ_ij

    int3 stressRow = tensorRowComps(i);

    // translate in direction i
    int3 im2 = im2_arr[i], im1 = im1_arr[i];  // transl in direction -i
    int3 ip1 = ip1_arr[i], ip2 = ip2_arr[i];  // transl in direction +i

    int3 coo_im2 = mastergrid.wrap(coo + im2);
    int3 coo_im1 = mastergrid.wrap(coo + im1);
    int3 coo_ip1 = mastergrid.wrap(coo + ip1);
    int3 coo_ip2 = mastergrid.wrap(coo + ip2);

    bool im2_inGeo = system.inGeometry(coo_im2);
    bool im1_inGeo = system.inGeometry(coo_im1);
    bool ip1_inGeo = system.inGeometry(coo_ip1);
    bool ip2_inGeo = system.inGeometry(coo_ip2);
    if (!im1_inGeo && !ip1_inGeo) {
      // --1-- zero
      continue;
    } else if (!im1_inGeo) {
      // --11- left boundary, apply central difference + traction free BC
      f += ws[i] * (stressTensor.vectorAt(coo_ip1, stressRow) +
                    stressTensor.vectorAt(idx, stressRow));
    } else if (!ip1_inGeo) {
      // -11-- right boundary, apply central difference + traction free BC
      f -= ws[i] * (stressTensor.vectorAt(idx, stressRow) +
                    stressTensor.vectorAt(coo_im1, stressRow));
    } else if (!im2_inGeo || !ip2_inGeo) {
      // -111-, 1111-, -1111 central difference,  ε ~ h^2
      f += ws[i] * (stressTensor.vectorAt(coo_ip1, stressRow) -
                    stressTensor.vectorAt(coo_im1, stressRow));
    } else {  // all 5 points are safe for sure
      // 11111 central difference,  ε ~ h^4
      f += ws[i] * ((4.0/3.0) * (stressTensor.vectorAt(coo_ip1, stressRow) -
                                  stressTensor.vectorAt(coo_im1, stressRow)) + 
                    (1.0/6.0) * (stressTensor.vectorAt(coo_im2, stressRow) -
                                 stressTensor.vectorAt(coo_ip2, stressRow)));
    }
  }

  fField.setVectorInCell(idx, f);
}


Field evalElasticForce(const Magnet* magnet) {

  Field fField(magnet->system(), 3);
  if (elasticityAssuredZero(magnet)) {
    fField.makeZero();
    return fField;
  }

  int ncells = fField.grid().ncells();
  Field stressTensor = evalStressTensor(magnet);
  real3 w = 0.5 / magnet->cellsize();  // 1 / 2*cellsize
  Grid mastergrid = magnet->world()->mastergrid();

  cudaLaunch(ncells, k_elasticForce, fField.cu(), stressTensor.cu(), w, mastergrid);

  return fField;
}

M_FieldQuantity elasticForceQuantity(const Magnet* magnet) {
  return M_FieldQuantity(magnet, evalElasticForce, 3, "elastic_force", "N/m3");
}
