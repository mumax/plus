#include "elastodynamics.hpp"
#include "elasticforce.hpp"
#include "cudalaunch.hpp"
#include "magnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "stresstensor.hpp"


/** Returns coo+relcoo if it is inside the geometry, otherwise it returns coo
 * itself (assumed to be safe). This mimics open boundary conditions.
*/
__device__ int3 safeCoord(int3 coo, int3 relcoo,
                          const CuSystem& system, const Grid& mastergrid) {
  int3 coo_ = mastergrid.wrap(coo + relcoo);
  if (system.inGeometry(coo_)) {  // don't convert to index if outside grid!
    return coo_;
  }
  return coo;
}

/**
 * Elastic force kernel translated directly from the mumax³ implementation.
 * https://github.com/Fredericvdv/Magnetoelasticity_MuMax3/blob/magnetoelastic/cuda/elas_free_bndry.cu
 * This calculation assumes a convex geometry and a uniform displacement in the
 * z-direction, ideal for xy-plane rectangles, and it applies traction-free
 * boundary conditions, as described in
 * https://doi.org/10.12688/openreseurope.13302.1
 */
__global__ void k_elasticForce(CuField fField,
                               const CuField uField,
                               const CuParameter C11,
                               const CuParameter C12,
                               const CuParameter C44,
                               const real3 w,  // 1 / cellsize
                               const Grid mg  // mastergrid
                               ) {
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

  // Central cell
  const int3 coo = grid.index2coord(idx);
  real3 u0 = uField.vectorAt(idx);
  real3 cc;
  real C11_0 = C11.valueAt(coo);
  real C12_0 = C12.valueAt(coo);
  real C44_0 = C44.valueAt(coo);
  
  // Neighbor cell
  int3 coo_;
  real3 u_, d_, cc_;
  real d__, cc__;

  // Result
  real3 f{0., 0., 0.};

  // x-interface
  if (!system.inGeometry(mg.wrap(coo + int3{-1, 0, 0}))) {  // Left
    if (!system.inGeometry(mg.wrap(coo + int3{0, -1, 0}))) {
      // Left-down corner

      coo_ = mg.wrap(coo + int3{1, 0, 0});
      cc__ = 0.5 * (C11_0 + C11.valueAt(coo_));
      f.x += cc__*w.x*w.x * (uField.valueAt(coo_, 0) - u0.x);
      
      coo_ = mg.wrap(coo + int3{0, 1, 0});
      cc__ = C12.valueAt(coo_);
      d__ = uField.valueAt(coo_, 1);
      coo_ = mg.wrap(coo + int3{1, 1, 0});
      cc__ += C12.valueAt(coo_);
      d__ += uField.valueAt(coo_, 1);
      // coo_ = mg.wrap(coo + int3{0, 0, 0});
      cc__ += C12_0;
      d__ -= u0.y;
      coo_ = mg.wrap(coo + int3{1, 0, 0});
      cc__ += C12.valueAt(coo_);
      d__ -= uField.valueAt(coo_, 1);
      cc__ *= 0.25;
      f.x += cc__*w.x*w.y*0.5*d__;

      coo_ = mg.wrap(coo + int3{0, 1, 0});
      cc__ = 0.5 * (C11_0 + C11.valueAt(coo_));
      f.y += cc__*w.y*w.y * (uField.valueAt(coo_, 1) - u0.y);

      coo_ = mg.wrap(coo + int3{1, 0, 0});
      cc__ = C12.valueAt(coo_);
      d__ = uField.valueAt(coo_, 0);
      coo_ = mg.wrap(coo + int3{1, 1, 0});
      cc__ += C12.valueAt(coo_);
      d__ += uField.valueAt(coo_, 0);
      // coo_ = mg.wrap(coo + int3{0, 0, 0});
      cc__ += C12_0;
      d__ -= u0.x;
      coo_ = mg.wrap(coo + int3{0, 1, 0});
      cc__ += C12.valueAt(coo_);
      d__ -= uField.valueAt(coo_, 0);
      cc__ *= 0.25;
      f.x += cc__*w.x*w.y*0.5*d__;
    
    } else if (!system.inGeometry(mg.wrap(coo + int3{0, 1, 0}))) {
      // Left-upper corner
      
      coo_ = mg.wrap(coo + int3{1, 0, 0});
      cc__ = 0.5 * (C11_0 + C11.valueAt(coo_));
      f.x += cc__*w.x*w.x * (uField.valueAt(coo_, 0) - u0.x);

      // coo_ = mg.wrap(coo + int{0, 0, 0});
      cc__ = C12_0;
      d__ = u0.y;
      coo_ = mg.wrap(coo + int3{1, 0, 0});
      cc__ += C12.valueAt(coo_);
      d__ += uField.valueAt(coo_, 1);
      coo_ = mg.wrap(coo + int3{0, -1, 0});
      cc__ += C12.valueAt(coo_);
      d__ -= uField.valueAt(coo_, 1);
      coo_ = mg.wrap(coo + int3{1, -1, 0});
      cc__ += C12.valueAt(coo_);
      d__ -= uField.valueAt(coo_, 1);
      cc__ *= 0.25;
      f.x += cc__*w.x*w.y*0.5*d__;

      coo_ = mg.wrap(coo + int3{0, -1, 0});
      cc__ = 0.5 * (C11_0 + C11.valueAt(coo_));
      f.y += cc__*w.y*w.y * (uField.valueAt(coo_, 1) - u0.y);

      coo_ = mg.wrap(coo + int3{1, 0, 0});
      cc__ = C12.valueAt(coo_);
      d__ = uField.valueAt(coo_, 0);
      coo_ = mg.wrap(coo + int3{1, -1, 0});
      cc__ += C12.valueAt(coo_);
      d__ += uField.valueAt(coo_, 0);
      // coo_ = mg.wrap(coo + int{0, 0, 0});
      cc__ += C12_0;
      d__ -= u0.x;
      coo_ = mg.wrap(coo + int3{0, -1, 0});
      cc__ += C12.valueAt(coo_);
      d__ -= uField.valueAt(coo_, 0);
      cc__ *= 0.25;
      f.x += cc__*w.x*w.y*0.5*d__;

    } else {
      // Left interface

      coo_ = mg.wrap(coo + int3{1, 0, 0});
      cc__ = 0.5 * (C11_0 + C11.valueAt(coo_));
      f.x += cc__*w.x*w.x * (uField.valueAt(coo_, 0) - u0.x);
          
      coo_ = mg.wrap(coo + int3{0, 1, 0});
      cc__ = C12.valueAt(coo_);
      d__ = uField.valueAt(coo_, 1);
      coo_ = mg.wrap(coo + int3{1, 1, 0});
      cc__ += C12.valueAt(coo_);
      d__ += uField.valueAt(coo_, 1);
      coo_ = mg.wrap(coo + int3{0, -1, 0});
      cc__ += C12.valueAt(coo_);
      d__ -= uField.valueAt(coo_, 1);
      coo_ = mg.wrap(coo + int3{1, -1, 0});
      cc__ += C12.valueAt(coo_);
      d__ -= uField.valueAt(coo_, 1);
      cc__ *= 0.25;
      f.x += cc__*w.x*w.y*0.25*d__;
      
      coo_ = mg.wrap(coo + int3{0, 1, 0});
      cc__ = C44.valueAt(coo_);
      d__ = uField.valueAt(coo_, 0);
      coo_ = mg.wrap(coo + int3{1, 1, 0});
      cc__ += C44.valueAt(coo_);
      d__ += uField.valueAt(coo_, 0);
      coo_ = mg.wrap(coo + int3{0, -1, 0});
      cc__ += C44.valueAt(coo_);
      d__ -= uField.valueAt(coo_, 0);
      coo_ = mg.wrap(coo + int3{1, -1, 0});
      cc__ += C44.valueAt(coo_);
      d__ -= uField.valueAt(coo_, 0);
      cc__ *= 0.25;
      f.y += cc__*w.x*w.y*0.25*d__;

      coo_ = mg.wrap(coo + int3{1, 0, 0});
      cc__ = 0.5 * (C44_0 + C44.valueAt(coo_));
      f.y += cc__*w.x*w.x * (uField.valueAt(coo_, 1) - u0.y);

      coo_ = mg.wrap(coo + int3{0, 1, 0});
      cc__ = C11.valueAt(coo_);
      d__ = uField.valueAt(coo_, 1);
      coo_ = mg.wrap(coo + int3{0, -1, 0});
      cc__ += C11.valueAt(coo_);
      d__ += uField.valueAt(coo_, 1);
      cc__ += 2 * C11_0;
      d__ -= 2 * u0.y;
      cc__ *= 0.25;
      f.y += cc__*w.y*w.y*d__;

      coo_ = mg.wrap(coo + int3{1, 1, 0});
      cc__ = C12.valueAt(coo_);
      d__ = uField.valueAt(coo_, 0);
      coo_ = mg.wrap(coo + int3{0, 1, 0});
      cc__ += C12.valueAt(coo_);
      d__ -= uField.valueAt(coo_, 0);
      coo_ = mg.wrap(coo + int3{1, -1, 0});
      cc__ += C12.valueAt(coo_);
      d__ -= uField.valueAt(coo_, 0);
      coo_ = mg.wrap(coo + int3{0, -1, 0});
      cc__ += C12.valueAt(coo_);
      d__ += uField.valueAt(coo_, 0);
      cc__ *= 0.25;
      f.y += cc__*w.x*w.y*0.5*d__;

    }
  } else if (!system.inGeometry(mg.wrap(coo + int3{1, 0, 0}))) {  // Right
    if (!system.inGeometry(mg.wrap(coo + int3{0, -1, 0}))) {
      // Right-down corner
            
      coo_ = mg.wrap(coo + int3{-1, 0, 0});
      cc__ = 0.5 * (C11_0 + C11.valueAt(coo_));
      f.x += cc__*w.x*w.x * (uField.valueAt(coo_, 0) - u0.x);

      coo_ = mg.wrap(coo + int3{0, 1, 0});
      cc__ = C12.valueAt(coo_);
      d__ = uField.valueAt(coo_, 1);
      coo_ = mg.wrap(coo + int3{-1, 1, 0});
      cc__ += C12.valueAt(coo_);
      d__ += uField.valueAt(coo_, 1);
      // coo_ = mg.wrap(coo + int3{0, 0, 0});
      cc__ += C12_0;
      d__ -= u0.y;
      coo_ = mg.wrap(coo + int3{-1, 0, 0});
      cc__ += C12.valueAt(coo_);
      d__ -= uField.valueAt(coo_, 1);
      cc__ *= 0.25;
      f.x += -cc__*w.x*w.y*0.5*d__;

      coo_ = mg.wrap(coo + int3{0, 1, 0});
      cc__ = 0.5 * (C11_0 + C11.valueAt(coo_));
      f.y += cc__*w.y*w.y * (uField.valueAt(coo_, 1) - u0.y);

      // coo_ = mg.wrap(coo + int3{0, 0, 0});
      cc__ = C12_0;
      d__ = u0.x;
      coo_ = mg.wrap(coo + int3{0, 1, 0});
      cc__ += C12.valueAt(coo_);
      d__ += uField.valueAt(coo_, 0);
      coo_ = mg.wrap(coo + int3{-1, 0, 0});
      cc__ += C12.valueAt(coo_);
      d__ -= uField.valueAt(coo_, 0);
      coo_ = mg.wrap(coo + int3{-1, 1, 0});
      cc__ += C12.valueAt(coo_);
      d__ -= uField.valueAt(coo_, 0);
      cc__ *= 0.25;
      f.x += -cc__*w.x*w.y*0.5*d__;

    } else if (!system.inGeometry(mg.wrap(coo + int3{0, 1, 0}))) {
      // Right-upper corner

      coo_ = mg.wrap(coo + int3{-1, 0, 0});
      cc__ = 0.5 * (C11_0 + C11.valueAt(coo_));
      f.x += -cc__*w.x*w.x * (u0.x - uField.valueAt(coo_, 0));

      // coo_ = mg.wrap(coo + int3{0, 0, 0});
      cc__ = C12_0;
      d__ = u0.y;
      coo_ = mg.wrap(coo + int3{-1, 0, 0});
      cc__ += C12.valueAt(coo_);
      d__ += uField.valueAt(coo_, 1);
      coo_ = mg.wrap(coo + int3{0, -1, 0});
      cc__ += C12.valueAt(coo_);
      d__ -= uField.valueAt(coo_, 1);
      coo_ = mg.wrap(coo + int3{-1, -1, 0});
      cc__ += C12.valueAt(coo_);
      d__ -= uField.valueAt(coo_, 1);
      cc__ *= 0.25;
      f.x += -cc__*w.x*w.y*0.5*d__;

      coo_ = mg.wrap(coo + int3{0, -1, 0});
      cc__ = 0.5 * (C11_0 + C11.valueAt(coo_));
      f.y += -cc__*w.y*w.y * (u0.y - uField.valueAt(coo_, 1));

      // coo_ = mg.wrap(coo + int3{0, 0, 0});
      cc__ = C12_0;
      d__ = u0.x;
      coo_ = mg.wrap(coo + int3{0, -1, 0});
      cc__ += C12.valueAt(coo_);
      d__ += uField.valueAt(coo_, 0);
      coo_ = mg.wrap(coo + int3{-1, 0, 0});
      cc__ += C12.valueAt(coo_);
      d__ -= uField.valueAt(coo_, 0);
      coo_ = mg.wrap(coo + int3{-1, -1, 0});
      cc__ += C12.valueAt(coo_);
      d__ -= uField.valueAt(coo_, 0);
      cc__ *= 0.25;
      f.x += -cc__*w.x*w.y*0.5*d__;

    } else {
      // Right interface

      coo_ = mg.wrap(coo + int3{-1, 0, 0});
      cc__ = 0.5 * (C11_0 + C11.valueAt(coo_));
      f.x += cc__*w.x*w.x * (uField.valueAt(coo_, 0) - u0.x);

      coo_ = mg.wrap(coo + int3{0, 1, 0}); 
      cc__ = C12.valueAt(coo_);
      d__ = uField.valueAt(coo_, 1);
      coo_ = mg.wrap(coo + int3{-1, 1, 0}); 
      cc__ += C12.valueAt(coo_);
      d__ += uField.valueAt(coo_, 1);
      coo_ = mg.wrap(coo + int3{0, -1, 0}); 
      cc__ += C12.valueAt(coo_);
      d__ -= uField.valueAt(coo_, 1);
      coo_ = mg.wrap(coo + int3{-1, -1, 0}); 
      cc__ += C12.valueAt(coo_);
      d__ -= uField.valueAt(coo_, 1);
      cc__ *= 0.25;
      f.x += -cc__*w.x*w.y*0.25*d__;

      coo_ = mg.wrap(coo + int3{0, 1, 0});
      cc__ = C44.valueAt(coo_);
      d__ = uField.valueAt(coo_, 0);
      coo_ = mg.wrap(coo + int3{-1, 1, 0});
      cc__ += C44.valueAt(coo_);
      d__ += uField.valueAt(coo_, 0);
      coo_ = mg.wrap(coo + int3{0, -1, 0});
      cc__ += C44.valueAt(coo_);
      d__ -= uField.valueAt(coo_, 0);
      coo_ = mg.wrap(coo + int3{-1, -1, 0});
      cc__ += C44.valueAt(coo_);
      d__ -= uField.valueAt(coo_, 0);
      cc__ *= 0.25;
      f.y += -cc__*w.x*w.y*0.25*d__;

      coo_ = mg.wrap(coo + int3{-1, 0, 0});
      cc__ = 0.5 * (C44_0 + C44.valueAt(coo_));
      f.y += cc__*w.x*w.x * (uField.valueAt(coo_, 1) - u0.y);

      coo_ = mg.wrap(coo + int3{0, 1, 0});
      cc__ = C11.valueAt(coo_);
      d__ = uField.valueAt(coo_, 1);
      coo_ = mg.wrap(coo + int3{0, -1, 0});
      cc__ += C11.valueAt(coo_);
      d__ += uField.valueAt(coo_, 1);
      cc__ += 2 * C11_0;
      d__ -= 2 * u0.y;
      cc__ *= 0.25;
      f.y += cc__*w.y*w.y*d__;

      coo_ = mg.wrap(coo + int3{0, 1, 0});
      cc__ = C12.valueAt(coo_);
      d__ = uField.valueAt(coo_, 0);
      coo_ = mg.wrap(coo + int3{-1, 1, 0});
      cc__ += C12.valueAt(coo_);
      d__ -= uField.valueAt(coo_, 0);
      coo_ = mg.wrap(coo + int3{0, -1, 0});
      cc__ += C12.valueAt(coo_);
      d__ -= uField.valueAt(coo_, 0);
      coo_ = mg.wrap(coo + int3{-1, -1, 0});
      cc__ += C12.valueAt(coo_);
      d__ += uField.valueAt(coo_, 0);
      cc__ *= 0.25;
      f.y += cc__*w.x*w.y*0.5*d__;

    }
  // y-interface
  } else if (!system.inGeometry(mg.wrap(coo + int3{0, -1, 0}))) {
    // Down interface

    coo_ = mg.wrap(coo + int3{1, 0, 0});
    cc__ = C11.valueAt(coo_);
    d__ = uField.valueAt(coo_, 0);
    coo_ = mg.wrap(coo + int3{-1, 0, 0});
    cc__ += C11.valueAt(coo_);
    d__ += uField.valueAt(coo_, 0);
    cc__ += 2 * C11_0;
    d__ -= 2 * u0.x;
    cc__ *= 0.25;
    f.x += cc__*w.x*w.x*d__;

    coo_ = mg.wrap(coo + int3{1, 1, 0});
    cc__ = C12.valueAt(coo_);
    d__ = uField.valueAt(coo_, 1);
    coo_ = mg.wrap(coo + int3{1, 0, 0});
    cc__ += C12.valueAt(coo_);
    d__ -= uField.valueAt(coo_, 1);
    coo_ = mg.wrap(coo + int3{-1, 1, 0});
    cc__ += C12.valueAt(coo_);
    d__ -= uField.valueAt(coo_, 1);
    coo_ = mg.wrap(coo + int3{-1, 0, 0});
    cc__ += C12.valueAt(coo_);
    d__ += uField.valueAt(coo_, 1);
    cc__ *= 0.25;
    f.x += cc__*w.x*w.y*0.5*d__;

    coo_ = mg.wrap(coo + int3{1, 0, 0});
    cc__ = C44.valueAt(coo_);
    d__ = uField.valueAt(coo_, 1);
    coo_ = mg.wrap(coo + int3{1, 1, 0});
    cc__ += C44.valueAt(coo_);
    d__ += uField.valueAt(coo_, 1);
    coo_ = mg.wrap(coo + int3{-1, 0, 0});
    cc__ += C44.valueAt(coo_);
    d__ -= uField.valueAt(coo_, 1);
    coo_ = mg.wrap(coo + int3{-1, 1, 0});
    cc__ += C44.valueAt(coo_);
    d__ -= uField.valueAt(coo_, 1);
    cc__ *= 0.25;
    f.x += cc__*w.x*w.y*0.25*d__;

    coo_ = mg.wrap(coo + int3{0, 1, 0});
    cc__ = 0.5 * (C44_0 + C44.valueAt(coo_));
    f.x += cc__*w.y*w.y * (uField.valueAt(coo_, 0) - u0.x);

    // coo_ = mg.wrap(coo + int3{0, 1, 0});
    cc__ = 0.5 * (C11_0 + C11.valueAt(coo_));
    f.y += cc__*w.y*w.y * (uField.valueAt(coo_, 1) - u0.y);

    coo_ = mg.wrap(coo + int3{1, 0, 0});
    cc__ = C12.valueAt(coo_);
    d__ = uField.valueAt(coo_, 0);
    coo_ = mg.wrap(coo + int3{1, 1, 0});
    cc__ += C12.valueAt(coo_);
    d__ += uField.valueAt(coo_, 0);
    coo_ = mg.wrap(coo + int3{-1, 0, 0});
    cc__ += C12.valueAt(coo_);
    d__ -= uField.valueAt(coo_, 0);
    coo_ = mg.wrap(coo + int3{-1, 1, 0});
    cc__ += C12.valueAt(coo_);
    d__ -= uField.valueAt(coo_, 0);
    cc__ *= 0.25;
    f.y += cc__*w.x*w.y*0.25*d__;

  } else if (!system.inGeometry(mg.wrap(coo + int3{0, 1, 0}))) {
    // Upper interface

    coo_ = mg.wrap(coo + int3{1, 0, 0});
    cc__ = C11.valueAt(coo_);
    d__ = uField.valueAt(coo_, 0);
    coo_ = mg.wrap(coo + int3{-1, 0, 0});
    cc__ += C11.valueAt(coo_);
    d__ += uField.valueAt(coo_, 0);
    cc__ += 2 * C11_0;
    d__ -= 2 * u0.x;
    cc__ *= 0.25;
    f.x += cc__*w.x*w.x*d__;

    coo_ = mg.wrap(coo + int3{1, 0, 0});
    cc__ = C12.valueAt(coo_);
    d__ = uField.valueAt(coo_, 1);
    coo_ = mg.wrap(coo + int3{1, -1, 0});
    cc__ += C12.valueAt(coo_);
    d__ -= uField.valueAt(coo_, 1);
    coo_ = mg.wrap(coo + int3{-1, 0, 0});
    cc__ += C12.valueAt(coo_);
    d__ -= uField.valueAt(coo_, 1);
    coo_ = mg.wrap(coo + int3{-1, -1, 0});
    cc__ += C12.valueAt(coo_);
    d__ += uField.valueAt(coo_, 1);
    cc__ *= 0.25;
    f.x +=  cc__*w.x*w.y*0.5*d__;

    coo_ = mg.wrap(coo + int3{1, 0, 0});
    cc__ = C44.valueAt(coo_);
    d__ = uField.valueAt(coo_, 1);
    coo_ = mg.wrap(coo + int3{1, -1, 0});
    cc__ += C44.valueAt(coo_);
    d__ += uField.valueAt(coo_, 1);
    coo_ = mg.wrap(coo + int3{-1, 0, 0});
    cc__ += C44.valueAt(coo_);
    d__ -= uField.valueAt(coo_, 1);
    coo_ = mg.wrap(coo + int3{-1, -1, 0});
    cc__ += C44.valueAt(coo_);
    d__ -= uField.valueAt(coo_, 1);
    cc__ *= 0.25;
    f.x += -cc__*w.x*w.y*0.25*d__;

    coo_ = mg.wrap(coo + int3{0, -1, 0});
    cc__ = 0.5 * (C44_0 + C44.valueAt(coo_));
    f.x += cc__*w.y*w.y * (uField.valueAt(coo_, 0) - u0.x);

    coo_ = mg.wrap(coo + int3{0, -1, 0});
    cc__ = 0.5 * (C11_0 + C11.valueAt(coo_));
    f.y += cc__*w.y*w.y * (uField.valueAt(coo_, 1) - u0.y);

    coo_ = mg.wrap(coo + int3{1, 0, 0});
    cc__ = C12.valueAt(coo_);
    d__ = uField.valueAt(coo_, 0);
    coo_ = mg.wrap(coo + int3{1, -1, 0});
    cc__ += C12.valueAt(coo_);
    d__ += uField.valueAt(coo_, 0);
    coo_ = mg.wrap(coo + int3{-1, 0, 0});
    cc__ += C12.valueAt(coo_);
    d__ -= uField.valueAt(coo_, 0);
    coo_ = mg.wrap(coo + int3{-1, -1, 0});
    cc__ += C12.valueAt(coo_);
    d__ -= uField.valueAt(coo_, 0);
    cc__ *= 0.25;
    f.y += -cc__*w.x*w.y*0.25*d__;

  } else {
    // Bulk

    // Double derivative in x-direction
    // d_ = real3{0., 0., 0.};
    cc = real3{C11_0, C44_0, C44_0};
    // Right neighbor
    coo_ = mg.wrap(coo + int3{1, 0, 0});
    u_ = uField.vectorAt(coo_);
    cc_ = real3{C11.valueAt(coo_), C44.valueAt(coo_), C44.valueAt(coo_)};
    cc_ = 0.5 * (cc + cc_);
    d_ = w.x*w.x * cc_ * (u_ - u0);
    // Left neighbor
    coo_ = mg.wrap(coo + int3{-1, 0, 0});
    u_ = uField.vectorAt(coo_);
    cc_ = real3{C11.valueAt(coo_), C44.valueAt(coo_), C44.valueAt(coo_)};
    cc_ = 0.5 * (cc + cc_);
    d_ += w.x*w.x * cc_ * (u_ - u0);

    f.x += d_.x;
    f.y += d_.y;
    // f.z += 0;

    // Double derivative in y-direction
    // d_ = real3{0., 0., 0.};
    cc = real3{C44_0, C11_0, C44_0};
    // Right neighbor
    coo_ = mg.wrap(coo + int3{0, 1, 0});
    u_ = uField.vectorAt(coo_);
    cc_ = real3{C44.valueAt(coo_), C11.valueAt(coo_), C44.valueAt(coo_)};
    cc_ = 0.5 * (cc + cc_);
    d_ = w.y*w.y * cc_ * (u_ - u0);
    //Left neighbour
    coo_ = mg.wrap(coo + int3{0, -1, 0});
    u_ = uField.vectorAt(coo_);
    cc_ = real3{C44.valueAt(coo_), C11.valueAt(coo_), C44.valueAt(coo_)};
    cc_ = 0.5 * (cc + cc_);
    d_ += w.y*w.y * cc_ * (u_ - u0);

    f.x += d_.x;
    f.y += d_.y;
    // f.z += 0;

    //dxy without boundaries
    d_ = real3{0., 0., 0.};
    cc__ = 0.;
    // (i+1,j+1)
    coo_ = mg.wrap(coo + int3{1, 1, 0});
    d_ += uField.vectorAt(coo_);
    cc__ += C12.valueAt(coo_) + C44.valueAt(coo_);
    // (i-1,j-1)
    coo_ = mg.wrap(coo + int3{-1, -1, 0});
    d_ += uField.vectorAt(coo_);
    cc__ += C12.valueAt(coo_) + C44.valueAt(coo_);
    // (i+1,j-1)
    coo_ = mg.wrap(coo + int3{1, -1, 0});
    d_ -= uField.vectorAt(coo_);
    cc__ += C12.valueAt(coo_) + C44.valueAt(coo_);
    // (i-1,j+1)
    coo_ = mg.wrap(coo + int3{-1, 1, 0});
    d_ -= uField.vectorAt(coo_);
    cc__ += C12.valueAt(coo_) + C44.valueAt(coo_);

    cc__ += 4 * (C12_0 + C44_0);
    cc__ *= 0.125;
    d_ = cc__*d_*0.25*w.x*w.y;

    f.x += d_.y;
    f.y += d_.x;
    // f.z += 0;

  }
  // Out of plane component has just Neumann boundary conditions

  // Double derivative in x-direction
  // Right neighbor
  coo_ = safeCoord(coo, int3{1, 0, 0}, system, mg);
  cc__ = 0.5 * (C44.valueAt(coo_) + C44_0);
  d__ = w.x*w.x*cc__ * (uField.valueAt(coo_, 2) - u0.z);
  f.z += d__;
  // Left neighbor
  coo_ = safeCoord(coo, int3{-1, 0, 0}, system, mg);
  cc__ = 0.5 * (C44.valueAt(coo_) + C44_0);
  d__ = w.x*w.x*cc__ * (uField.valueAt(coo_, 2) - u0.z);
  f.z += d__;

  // Double derivative in y-direction
  // Top neighbor
  coo_ = safeCoord(coo, int3{0, 1, 0}, system, mg);
  cc__ = 0.5 * (C44.valueAt(coo_) + C44_0);
  d__ = w.y*w.y*cc__ * (uField.valueAt(coo_, 2) - u0.z);
  f.z += d__;
  // Bottom neighbor
  coo_ = safeCoord(coo, int3{0, -1, 0}, system, mg);
  cc__ = 0.5 * (C44.valueAt(coo_) + C44_0);
  d__ = w.y*w.y*cc__ * (uField.valueAt(coo_, 2) - u0.z);
  f.z += d__;


  fField.setVectorInCell(idx, f);
}


Field evalElasticForce(const Magnet* magnet) {

  Field fField(magnet->system(), 3);
  if (elasticityAssuredZero(magnet)) {
    fField.makeZero();
    return fField;
  }

  int ncells = fField.grid().ncells();
  CuField uField = magnet->elasticDisplacement()->field().cu();
  CuParameter C11 = magnet->C11.cu();
  CuParameter C12 = magnet->C12.cu();
  CuParameter C44 = magnet->C44.cu();
  real3 w = 1 / magnet->cellsize();
  Grid mastergrid = magnet->world()->mastergrid();

  cudaLaunch(ncells, k_elasticForce, fField.cu(), uField, C11, C12, C44, w, mastergrid);

  return fField;
}

M_FieldQuantity elasticForceQuantity(const Magnet* magnet) {
  return M_FieldQuantity(magnet, evalElasticForce, 3, "elastic_force", "N/m3");
}
