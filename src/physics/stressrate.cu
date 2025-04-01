#include "cudalaunch.hpp"
#include "elastodynamics.hpp"
#include "field.hpp"
#include "magnet.hpp"
#include "parameter.hpp"
#include "stressrate.hpp"


// ==================== Norm Stress Rate ====================

__global__ void k_stressRate(CuField srField,
                             const CuField v,
                             const CuParameter C11,
                             const CuParameter C12,
                             const CuParameter C44,
                             const real3 w,  // 1/cellsize
                             const Grid mastergrid) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = srField.system;
  const Grid grid = system.grid;

  // When outside the geometry, set to zero and return early
  if (!system.inGeometry(idx)) {
    if (grid.cellInGrid(idx)) {
      for (int i = 0; i < srField.ncomp; i++)
        srField.setValueInCell(idx, i, 0);
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

  real der[3][3] = {{0,0,0}, {0,0,0}, {0,0,0}};  // derivatives ∂i(vj)
  real3 v_0 = v.vectorAt(idx);
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

    // determine a derivative ∂i(v)
    real3 dvdi;
    if (!system.inGeometry(coo_im1) && !system.inGeometry(coo_ip1)) {
      // --1-- zero
      dvdi = real3{0, 0, 0};
    } else if ((!system.inGeometry(coo_im2) || !system.inGeometry(coo_ip2)) &&
                system.inGeometry(coo_im1) && system.inGeometry(coo_ip1)) {
      // -111-, 1111-, -1111 central difference,  ε ~ h^2
      dvdi = 0.5 * (v.vectorAt(coo_ip1) - v.vectorAt(coo_im1));
    } else if (!system.inGeometry(coo_im2) && !system.inGeometry(coo_ip1)) {
      // -11-- backward difference, ε ~ h^1
      dvdi =  (v_0 - v.vectorAt(coo_im1));
    } else if (!system.inGeometry(coo_im1) && !system.inGeometry(coo_ip2)) {
      // --11- forward difference,  ε ~ h^1
      dvdi = (-v_0 + v.vectorAt(coo_ip1));
    } else if (system.inGeometry(coo_im2) && !system.inGeometry(coo_ip1)) {
      // 111-- backward difference, ε ~ h^2
      dvdi =  (0.5 * v.vectorAt(coo_im2) - 2.0 * v.vectorAt(coo_im1) + 1.5 * v_0);
    } else if (!system.inGeometry(coo_im1) && system.inGeometry(coo_ip1)) {
      // --111 forward difference,  ε ~ h^2
      dvdi = (-0.5 * v.vectorAt(coo_ip2) + 2.0 * v.vectorAt(coo_ip1) - 1.5 * v_0);
    } else {
      // 11111 central difference,  ε ~ h^4
      dvdi = ((2./3.)  * (v.vectorAt(coo_ip1) - v.vectorAt(coo_im1)) + 
              (1./12.) * (v.vectorAt(coo_im2) - v.vectorAt(coo_ip2)));
    }
    dvdi *= wi;

    der[i][0] = dvdi.x;
    der[i][1] = dvdi.y;
    der[i][2] = dvdi.z;
  }

  // create the stress rate tensor
  for (int i = 0; i < 3; i++){
    for (int j = i; j < 3; j++){
      if (i == j) {  // diagonals
        srField.setValueInCell(idx, i,
          C11.valueAt(idx) * der[i][i] +
          C12.valueAt(idx) * (der[(i+1)%3][(i+1)%3] + der[(i+2)%3][(i+2)%3])
        );
      } else {  // off-diagonal
        srField.setValueInCell(idx, i+j+2,
                               C44.valueAt(idx) * (der[i][j] + der[j][i]));
      }
    }
  }
}


Field evalStressRate(const Magnet* magnet) {

  Field srField(magnet->system(), 6);  // symmetric 3x3 tensor
  if (elasticityAssuredZero(magnet)) {
    srField.makeZero();
    return srField;
  }

  int ncells = srField.grid().ncells();
  CuField vField = magnet->elasticVelocity()->field().cu();
  CuParameter C11 = magnet->C11.cu();
  CuParameter C12 = magnet->C12.cu();
  CuParameter C44 = magnet->C44.cu();
  real3 w = 1/ magnet->cellsize();
  Grid mastergrid = magnet->world()->mastergrid();

  cudaLaunch(ncells, k_stressRate, srField.cu(), vField, C11, C12, C44, w, mastergrid);

  return srField;
}

M_FieldQuantity stressRateQuantity(const Magnet* magnet) {
  return M_FieldQuantity(magnet, evalStressRate, 6, "stress_rate", "Pa/s");
}
