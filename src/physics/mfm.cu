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
__global__ void k_magneticForceMicroscopy(CuField kernel, CuField magnetization,
                                          const Grid mastergrid, const int3 pbcRepetitions,
                                          CuParameter lift, CuParameter tipsize) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (!kernel.cellInGrid(idx))
        return;
    
    // The cell-coordinate of the tip (without lift)
    const real3 cellsize = kernel.system.cellsize;
    int3 coo = kernel.system.grid.index2coord(idx);
    real x0 = coo.x * cellsize.x;
    real y0 = coo.y * cellsize.y;
    real z0 = coo.z * cellsize.z;

    real tipCharge = 1/MU0;  // charge at the tip
    real delta = 1e-9;       // tip oscillation, take 2nd derivative over this distance
    real V = cellsize.x * cellsize.y * cellsize.z;  // volume of a cell
    real pi = 3.1415926535897931f;

    // Size of the grid
    int xmax = kernel.system.grid.size().x;
    int ymax = kernel.system.grid.size().y;
    int zmax = kernel.system.grid.size().z;

    real dFdz = 0.;

    // Loop over pbc
    for (int Nx = -pbcRepetitions.x; Nx <= pbcRepetitions.x; Nx++) {
        real xpbc = Nx * mastergrid.size().x;
        for (int Ny = -pbcRepetitions.y; Ny <= pbcRepetitions.y; Ny++) {
            real ypbc = Ny * mastergrid.size().y;
            for (int Nz = -pbcRepetitions.z; Nz <= pbcRepetitions.z; Nz++) {
                real zpbc = Nz * mastergrid.size().z;

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
                                real3 B = R * tipCharge/(4*pi*r*r*r);
                                
                                // Second pole position and field
                                R.z = z0 + z - (lift.valueAt(idx) + tipsize.valueAt(idx) + i*delta);
                                r = sqrt(R.x*R.x + R.y*R.y + R.z*R.z);
                                B.x -= R.x * tipCharge/(4*pi*r*r*r);
                                B.y -= R.y * tipCharge/(4*pi*r*r*r);
                                B.z -= R.z * tipCharge/(4*pi*r*r*r);
                                
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

Field evalMagneticForceMicroscopy(const Ferromagnet* magnet) {
    Field mfm(magnet->system(), 1, 0.0);
    const CuField magnetization = fullMagnetizationQuantity(magnet).eval().cu();
    Grid mastergrid = magnet->world()->mastergrid();
    int3 pbcRepetitions = magnet->world()->pbcRepetitions();
    CuParameter lift = magnet->lift.cu();
    CuParameter tipsize = magnet->tipsize.cu();
    int ncells = mfm.grid().ncells();
    cudaLaunch(ncells, k_magneticForceMicroscopy, mfm.cu(), magnetization, mastergrid, pbcRepetitions, lift, tipsize);
    return mfm;
}

Field evalMagneticForceMicroscopyAFM(const Antiferromagnet* magnet) {
    Field mfm(magnet->system(), 1, 0.0);
    const CuField magnetization = fullMagnetizationQuantity(magnet).eval().cu();
    Grid mastergrid = magnet->world()->mastergrid();
    int3 pbcRepetitions = magnet->world()->pbcRepetitions();
    CuParameter lift = magnet->lift.cu();
    CuParameter tipsize = magnet->tipsize.cu();
    int ncells = mfm.grid().ncells();
    cudaLaunch(ncells, k_magneticForceMicroscopy, mfm.cu(), magnetization, mastergrid, pbcRepetitions, lift, tipsize);
    return mfm;
}

FM_FieldQuantity magneticForceMicroscopyQuantity(const Ferromagnet* magnet) {
    return FM_FieldQuantity(magnet, evalMagneticForceMicroscopy, 1, "magnetic_force_microscopy", "J");
}

AFM_FieldQuantity magneticForceMicroscopyAFMQuantity(const Antiferromagnet* magnet) {
    return AFM_FieldQuantity(magnet, evalMagneticForceMicroscopyAFM, 1, "magnetic_force_microscopy", "J");
}