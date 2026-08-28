#include "cudalaunch.hpp"
#include "magnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "straintensor.hpp"


bool strainTensorAssuredZero(const Magnet* magnet) {
  return !magnet->enableElastodynamics();
}


__global__ void k_strainTensor(CuField strain,
                               const CuField u,
                               const real3 w,  // w = 1/cellsize
                               const Grid mastergrid) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = strain.system;
  const Grid grid = system.grid;

  // When outside the geometry, set to zero and return early
  if (!system.inGeometry(idx)) {
    if (grid.cellInGrid(idx)) {
      for (int i = 0; i < strain.ncomp; i++)
        strain.setValueInCell(idx, i, 0);
    }
    return;
  }

  const real ws[3] = {w.x, w.y, w.z};
  const int3 im2_arr[3] = {int3{-2, 0, 0}, int3{0,-2, 0}, int3{0, 0,-2}};
  const int3 im1_arr[3] = {int3{-1, 0, 0}, int3{0,-1, 0}, int3{0, 0,-1}};
  const int3 ip1_arr[3] = {int3{ 1, 0, 0}, int3{0, 1, 0}, int3{0, 0, 1}};
  const int3 ip2_arr[3] = {int3{ 2, 0, 0}, int3{0, 2, 0}, int3{0, 0, 2}};
  const int3 coo = grid.index2coord(idx);

  real der[3][3] = {{0,0,0}, {0,0,0}, {0,0,0}};  // derivatives ∂i(mj)
  real3 u_0 = u.vectorAt(idx);
#pragma unroll
  for (int i = 0; i < 3; i++) {  // i is a {x, y, z} direction
    // take translation in i direction
    real wi = ws[i]; 
    int3 im2 = im2_arr[i], im1 = im1_arr[i];  // transl in direction -i
    int3 ip1 = ip1_arr[i], ip2 = ip2_arr[i];  // transl in direction +i

    int3 coo_im2 = mastergrid.wrap(coo + im2);
    int3 coo_im1 = mastergrid.wrap(coo + im1);
    int3 coo_ip1 = mastergrid.wrap(coo + ip1);
    int3 coo_ip2 = mastergrid.wrap(coo + ip2);

    // determine a derivative ∂i(m)
    real3 dudi;
    if (!system.inGeometry(coo_im1) && !system.inGeometry(coo_ip1)) {
      // --1-- zero
      dudi = real3{0, 0, 0};
    } else if ((!system.inGeometry(coo_im2) || !system.inGeometry(coo_ip2)) &&
                system.inGeometry(coo_im1) && system.inGeometry(coo_ip1)) {
      // -111-, 1111-, -1111 central difference,  ε ~ h^2
      dudi = 0.5 * (u.vectorAt(coo_ip1) - u.vectorAt(coo_im1));
    } else if (!system.inGeometry(coo_im2) && !system.inGeometry(coo_ip1)) {
      // -11-- backward difference, ε ~ h^1
      dudi =  (u_0 - u.vectorAt(coo_im1));
    } else if (!system.inGeometry(coo_im1) && !system.inGeometry(coo_ip2)) {
      // --11- forward difference,  ε ~ h^1
      dudi = (-u_0 + u.vectorAt(coo_ip1));
    } else if (system.inGeometry(coo_im2) && !system.inGeometry(coo_ip1)) {
      // 111-- backward difference, ε ~ h^2
      dudi =  (0.5 * u.vectorAt(coo_im2) - 2.0 * u.vectorAt(coo_im1) + 1.5 * u_0);
    } else if (!system.inGeometry(coo_im1) && system.inGeometry(coo_ip1)) {
      // --111 forward difference,  ε ~ h^2
      dudi = (-0.5 * u.vectorAt(coo_ip2) + 2.0 * u.vectorAt(coo_ip1) - 1.5 * u_0);
    } else {
      // 11111 central difference,  ε ~ h^4
      dudi = ((2.0/3.0)  * (u.vectorAt(coo_ip1) - u.vectorAt(coo_im1)) + 
              (1.0/12.0) * (u.vectorAt(coo_im2) - u.vectorAt(coo_ip2)));
    }
    dudi *= wi;

    der[i][0] = dudi.x;
    der[i][1] = dudi.y;
    der[i][2] = dudi.z;
  }

  // create the strain tensor
  for (int i = 0; i < 3; i++){
    for (int j = i; j < 3; j++){
      if (i == j) {  // diagonals
        strain.setValueInCell(idx, i, der[i][j]);
      }
      else {  // off-diagonal
        strain.setValueInCell(idx, i+j+2,
                              0.5 * (der[i][j] + der[j][i]));
      }
    }
  }
}


Field evalStrainTensor(const Magnet* magnet) {
  Field strain(magnet->system(), 6);
  if (strainTensorAssuredZero(magnet)) {
    strain.makeZero();
    return strain;
  }

  int ncells = strain.grid().ncells();
  CuField u = magnet->elasticDisplacement()->field().cu();
  real3 w = 1 / magnet->cellsize();
  Grid mastergrid = magnet->world()->mastergrid();

  cudaLaunch(ncells, k_strainTensor, strain.cu(), u, w, mastergrid);
  return strain;
}


M_FieldQuantity strainTensorQuantity(const Magnet* magnet) {
  return M_FieldQuantity(magnet, evalStrainTensor, 6, "strain_tensor", "");
}

// --------------------
// Strain Rate

Field evalStrainRate(const Magnet* magnet) {
  Field strainRate(magnet->system(), 6);  // symmetric 3x3 tensor
  if (strainTensorAssuredZero(magnet)) {  // same condition
    strainRate.makeZero();
    return strainRate;
  }

  int ncells = strainRate.grid().ncells();
  CuField v = magnet->elasticVelocity()->field().cu();
  real3 w = 1/ magnet->cellsize();
  Grid mastergrid = magnet->world()->mastergrid();

  // The math for strain rate is exactly the same as for strain tensor,
  // but applied to velocity instead of displacement.
  cudaLaunch(ncells, k_strainTensor, strainRate.cu(), v, w, mastergrid);

  return strainRate;
}

M_FieldQuantity strainRateQuantity(const Magnet* magnet) {
  return M_FieldQuantity(magnet, evalStrainRate, 6, "strain_rate", "1/s");
}
