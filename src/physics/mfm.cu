#include "constants.hpp"
#include "cudalaunch.hpp"
#include "fullmag.hpp"
#include "mfm.hpp"
#include "system.hpp"

#include <iostream>
#include <stdio.h>

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
                                          real lift,
                                          real tipsize,
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
    real delta = 1e-9;  // tip oscillation, take 2nd derivative over this distance
    // Size of the grid
    int xmax = magnetization.system.grid.size().x;
    int ymax = magnetization.system.grid.size().y;
    int zmax = magnetization.system.grid.size().z;

    real dFdz = 0.;
    // Loop over valid pbc
    for (int Ny = -pbcRepetitions.y; Ny <= pbcRepetitions.y; Ny++) {
        real ypbc = Ny * mastergrid.size().y;
        for (int Nx = -pbcRepetitions.x; Nx <= pbcRepetitions.x; Nx++) {
            real xpbc = Nx * mastergrid.size().x;            

            // Loop over cells in the magnet
            for (int iz = 0; iz < zmax; iz++) {
                real z = iz * cellsize.z;
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
                                       z0 + z - (lift + i*delta)};
                            real r = sqrt(R.x*R.x + R.y*R.y + R.z*R.z);
                            real3 B = R * prefactor/(4*pi*r*r*r);
                            
                            // Second pole position and field
                            R.z -= tipsize;
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
    kernel.setValueInCell(idx, 0, dFdz);
}

MFM::MFM(const Magnet* magnet,
         const Grid grid)
    : magnet_(magnet),
      grid_(grid),
      system_(std::make_shared<System>(magnet->world(), grid_)),
      tipsize(1e-3) {
    setLift(10e-9);
    if (grid_.size().z > 1) {
        throw std::invalid_argument("MFM should scan a 2D surface. Reduce"
                                    "the number of z-cells to 1.");
    }

    if (magnet->world()->pbcRepetitions().z > 0) {
        throw std::invalid_argument("Cannot take MFM picture of PBC in the"
                                    "z-direction.");
    }
}

Field MFM::eval() const {
    Field mfm(system_, 1, 0.0);
    Grid mastergrid = magnet_->world()->mastergrid();
    int3 pbcRepetitions = magnet_->world()->pbcRepetitions();
    const real V = magnet_->world()->cellVolume();
    int ncells = grid_.ncells();
    Field magnetization;
    if (const Ferromagnet* mag = magnet_->asFM()) {
        magnetization = fullMagnetizationQuantity(mag).eval();
    } else if (const Antiferromagnet* mag = magnet_->asAFM()) {
        magnetization = fullMagnetizationQuantity(mag).eval();
    } else {
        throw std::invalid_argument("Cannot calculate MFM of instance which"
                                    "is no Ferromagnet or Antiferromagnet.");
    }

    cudaLaunch(ncells, k_magneticForceMicroscopy, mfm.cu(), magnetization.cu(), mastergrid, pbcRepetitions, lift_, tipsize, V);

    return mfm;
}

int MFM::ncomp() const {
    return 1;
}

std::shared_ptr<const System> MFM::system() const {
    return system_;
}

void MFM::setLift(real value) {
    // z_grid * cz + lift - delta < (z_magnet + nz) * cz - cz/2
    if (grid_.origin().z * magnet_->world()->cellsize().z + lift_ - 1e-9 <= (magnet_->grid().origin().z + magnet_->grid().size().z) * magnet_->world()->cellsize().z - magnet_->world()->cellsize().z /2) {
        throw std::invalid_argument("Tip crashed into the sample. increase"
                                    "the lift or the origin of the MFM grid.");
    }
    lift_ = value;
}