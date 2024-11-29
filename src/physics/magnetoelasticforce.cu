#include "cudalaunch.hpp"
#include "elastodynamics.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "magnetoelasticfield.hpp"  // magnetoelasticAssuredZero
#include "magnetoelasticforce.hpp"
#include "parameter.hpp"


__global__ void k_magnetoelasticForce(CuField fField,
                                      const CuField m,
                                      const CuParameter B1,
                                      const CuParameter B2,
                                      const real3 w,  // w = 1/cellsize
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
  const real ws[3] = {w.x, w.y, w.z};
  const int3 im2_arr[3] = {int3{-2, 0, 0}, int3{0,-2, 0}, int3{0, 0,-2}};
  const int3 im1_arr[3] = {int3{-1, 0, 0}, int3{0,-1, 0}, int3{0, 0,-1}};
  const int3 ip1_arr[3] = {int3{ 1, 0, 0}, int3{0, 1, 0}, int3{0, 0, 1}};
  const int3 ip2_arr[3] = {int3{ 2, 0, 0}, int3{0, 2, 0}, int3{0, 0, 2}};
  const int3 coo = grid.index2coord(idx);

  real der[3][3] = {{0,0,0}, {0,0,0}, {0,0,0}};  // derivatives ∂i(mj)
  real3 m_0 = m.vectorAt(idx);
#pragma unroll
  for (int i=0; i<3; i++) {  // i is a {x, y, z} direction
    // take translation in i direction
    real wi = ws[i]; 
    int3 im2 = im2_arr[i], im1 = im1_arr[i];  // transl in direction -i
    int3 ip1 = ip1_arr[i], ip2 = ip2_arr[i];  // transl in direction +i
    
    int3 coo_im2 = mastergrid.wrap(coo + im2);
    int3 coo_im1 = mastergrid.wrap(coo + im1);
    int3 coo_ip1 = mastergrid.wrap(coo + ip1);
    int3 coo_ip2 = mastergrid.wrap(coo + ip2);
    
    // determine a derivative ∂i(m)
    real3 dmdi;
    if (!system.inGeometry(coo_im1) && !system.inGeometry(coo_ip1)) {
      // --1-- zero
      dmdi = real3{0, 0, 0};
    } else if ((!system.inGeometry(coo_im2) || !system.inGeometry(coo_ip2)) &&
                system.inGeometry(coo_im1) && system.inGeometry(coo_ip1)) {
      // -111-, 1111-, -1111 central difference,  ε ~ h^2
      dmdi = 0.5 * (m.vectorAt(coo_ip1) - m.vectorAt(coo_im1));
    } else if (!system.inGeometry(coo_im2) && !system.inGeometry(coo_ip1)) {
      // -11-- backward difference, ε ~ h^1
      dmdi =  (m_0 - m.vectorAt(coo_im1));
    } else if (!system.inGeometry(coo_im1) && !system.inGeometry(coo_ip2)) {
      // --11- forward difference,  ε ~ h^1
      dmdi = (-m_0 + m.vectorAt(coo_ip1));
    } else if (system.inGeometry(coo_im2) && !system.inGeometry(coo_ip1)) {
      // 111-- backward difference, ε ~ h^2
      dmdi =  (0.5 * m.vectorAt(coo_im2) - 2.0 * m.vectorAt(coo_im1) + 1.5 * m_0);
    } else if (!system.inGeometry(coo_im1) && system.inGeometry(coo_ip1)) {
      // --111 forward difference,  ε ~ h^2
      dmdi = (-0.5 * m.vectorAt(coo_ip2) + 2.0 * m.vectorAt(coo_ip1) - 1.5 * m_0);
    } else {
      // 11111 central difference,  ε ~ h^4
      dmdi = ((2./3.)  * (m.vectorAt(coo_ip1) - m.vectorAt(coo_im1)) + 
              (1./12.) * (m.vectorAt(coo_im2) - m.vectorAt(coo_ip2)));
    }
    dmdi *= wi;

    der[i][0] = dmdi.x;
    der[i][1] = dmdi.y;
    der[i][2] = dmdi.z;
  }

  real m_here[3] = {m.valueAt(idx, 0), m.valueAt(idx, 1), m.valueAt(idx, 2)};
  int ip1, ip2;
#pragma unroll
  for (int i=0; i<3; i++) {
    ip1 = i+1; ip2 = i+2;
    // If they exceed 3, loop around
    if (ip1 >= 3) ip1 -= 3;
    if (ip2 >= 3) ip2 -= 3;

    real f_i = 2 * B1.valueAt(idx) * m_here[i] * der[i][i];
    f_i += B2.valueAt(idx) * m_here[i] * (der[ip1][ip1] + der[ip2][ip2]);
    f_i += B2.valueAt(idx) * (m_here[ip1] * der[ip1][i] + m_here[ip2] * der[ip2][i]);
    fField.setValueInCell(idx, i, f_i);
  }
}


Field evalMagnetoelasticForce(const Ferromagnet* magnet) {
  Field fField(magnet->system(), 3);
  if (magnetoelasticAssuredZero(magnet)) {
    fField.makeZero();
    return fField;
  }
  int ncells = fField.grid().ncells();
  CuField m = magnet->magnetization()->field().cu();
  CuParameter B1 = magnet->B1.cu();
  CuParameter B2 = magnet->B2.cu();
  real3 w = 1 / magnet->cellsize();
  Grid mastergrid = magnet->world()->mastergrid();
  cudaLaunch(ncells, k_magnetoelasticForce, fField.cu(), m, B1, B2, w, mastergrid);
  return fField;
}


FM_FieldQuantity magnetoelasticForceQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalMagnetoelasticForce, 3,
                          "magnetoelastic_force", "N/m3");
}
