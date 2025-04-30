#include "constants.hpp"
#include "cudalaunch.hpp"
#include "fullmag.hpp"
#include "mfm.hpp"

/** This code calculates an MFM kernel
  * Need to calculate dF/dz = sum M . d²B/dz²
  * The sum runs over each cell in a magnet.
  * M is the magnetization in that cell.
  * B is the stray field from the tip evaluated in the cell.
  * Source: The design and verification of MuMax3.
  */
__global__ void k_magneticForceMicroscopy(CuField kernel,
                                          CuField magnetization,
                                          const Grid mastergrid,
                                          const int3 pbcRepetitions,
                                          CuParameter lift,
                                          CuParameter tipsize,
                                          const real V) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (!kernel.cellInGrid(idx))
        return;

    real pi = 3.1415926535897931f;
    
    // The cell-coordinate of the tip (without lift)
    const real3 cellsize = kernel.system.cellsize;
    int3 coo = kernel.system.grid.index2coord(idx);
    real x0 = coo.x * cellsize.x;
    real y0 = coo.y * cellsize.y;
    real z0 = coo.z * cellsize.z;

    real prefactor = 1/(4*pi*MU0);  // charge at the tip is 1/µ0
    real delta = 1e-9;       // tip oscillation, take 2nd derivative over this distance
    // Size of the grid
    int xmax = kernel.system.grid.size().x;
    int ymax = kernel.system.grid.size().y;
    int zmax = kernel.system.grid.size().z;

    real dFdz = 0.;

    // Loop over pbc
    for (int Nz = -pbcRepetitions.z; Nz <= pbcRepetitions.z; Nz++) {
        real zpbc = Nz * mastergrid.size().z;
        for (int Ny = -pbcRepetitions.y; Ny <= pbcRepetitions.y; Ny++) {
            real ypbc = Ny * mastergrid.size().y;
            for (int Nx = -pbcRepetitions.x; Nx <= pbcRepetitions.x; Nx++) {
                real xpbc = Nx * mastergrid.size().x;            

                // Loop over cells in the magnet
                for (int iz = 0; iz < zmax; iz++) {
                    real z = (iz+zpbc) * cellsize.z;
                    for (int iy = 0; iy < ymax; iy++) {
                        real y = (iy+ypbc) * cellsize.y;
                        for (int ix = 0; ix < xmax; ix++) {
                            real x = (ix+xpbc) * cellsize.x;

                            real3 m = magnetization.vectorAt(int3{ix, iy, iz});
                            real E[3];  // Energy of 3 tip positions

                            // Get 3 different tip heights
                            for (int i = -1; i <= 1; i++) {
                                // First pole position and field
                                real3 R = {x0-x,
                                           y0-y,
                                           z0 + z - (lift.valueAt(idx) + i*delta)};
                                real r = sqrt(R.x*R.x + R.y*R.y + R.z*R.z);
                                real3 B = R * prefactor/(4*pi*r*r*r);
                                
                                // Second pole position and field
                                R.z -= tipsize.valueAt(idx);
                                r = sqrt(R.x*R.x + R.y*R.y + R.z*R.z);
                                B -= R * prefactor/(4*pi*r*r*r);
                                
                                // Energy (B.M) * V
                                E[i+1] = (B.x * m.x + B.y * m.y + B.z * m.z) * V;
                            }

                            // dF/dz = d²E/dz²
                            dFdz += ((E[0] - E[1]) + (E[2] - E[1])) / (delta*delta);
                        }
                    }
                }
            }
        }
    }
    kernel.setValueInCell(idx, 0, dFdz);
}

Field evalMagneticForceMicroscopy(const Magnet* magnet) {
    Field mfm(magnet->system(), 1, 0.0);
    Grid mastergrid = magnet->world()->mastergrid();
    int3 pbcRepetitions = magnet->world()->pbcRepetitions();
    CuParameter lift = magnet->lift.cu();
    CuParameter tipsize = magnet->tipsize.cu();
    const real V = magnet->world()->cellVolume();
    int ncells = mfm.grid().ncells();

    if (const Ferromagnet* mag = magnet->asFM()) {
        const CuField magnetization = fullMagnetizationQuantity(mag).eval().cu();
        cudaLaunch(ncells, k_magneticForceMicroscopy, mfm.cu(), magnetization, mastergrid, pbcRepetitions, lift, tipsize, V);
    }
    else if (const Antiferromagnet* mag = magnet->asAFM()) {
        const CuField magnetization = fullMagnetizationQuantity(mag).eval().cu();
        cudaLaunch(ncells, k_magneticForceMicroscopy, mfm.cu(), magnetization, mastergrid, pbcRepetitions, lift, tipsize, V);
    }
    else {
    throw std::invalid_argument("Cannot calculate MFM of instance which"
                                "is no Ferromagnet or Antiferromagnet.");
    }
    
    return mfm;
}

FM_FieldQuantity magneticForceMicroscopyQuantity(const Ferromagnet* magnet) {
    return FM_FieldQuantity(magnet, evalMagneticForceMicroscopy, 1, "magnetic_force_microscopy", "J");
}

AFM_FieldQuantity magneticForceMicroscopyQuantity(const Antiferromagnet* magnet) {
    return AFM_FieldQuantity(magnet, evalMagneticForceMicroscopy, 1, "magnetic_force_microscopy", "J");
}