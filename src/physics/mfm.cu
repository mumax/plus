#include "constants.hpp"
#include "cudalaunch.hpp"
#include "datatypes.hpp"
#include "fullmag.hpp"
#include "gpubuffer.hpp"
#include "mfm.hpp"
#include "system.hpp"
#include <map>
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
                                          const real V,
                                          bool* crashedResult) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (!kernel.cellInGrid(idx))
        return;

    real pi = 3.1415926535897931;
    
    // The cell-coordinate of the tip (without lift)
    const real3 cellsize = kernel.system.cellsize;
    int3 coo = kernel.system.grid.index2coord(idx);
    real x0 = coo.x * cellsize.x;
    real y0 = coo.y * cellsize.y;
    real z0 = coo.z * cellsize.z;

    real prefactor = 1/(4*pi*MU0);  // charge at the tip is 1/µ0
    real delta = 1e-9;  // tip oscillation, take 2nd derivative over this distance
    // Size of the magnet's grid
    int magnetNCells = magnetization.system.grid.ncells();

    real dFdz = 0.;
    // Loop over valid pbc
    for (int Ny = -pbcRepetitions.y; Ny <= pbcRepetitions.y; Ny++) {
        real ypbc = Ny * mastergrid.size().y;
        for (int Nx = -pbcRepetitions.x; Nx <= pbcRepetitions.x; Nx++) {
            real xpbc = Nx * mastergrid.size().x;            

            // Loop over cells in the magnet
            for (int n = 0; n < magnetNCells; n++) {
                if (!magnetization.cellInGeometry(n))
                    continue;

                int3 magnetCoo = magnetization.system.grid.index2coord(n);
                int ix = magnetCoo.x;
                int iy = magnetCoo.y;
                int iz = magnetCoo.z;

                real x = (ix + xpbc) * cellsize.x;
                real y = (iy + ypbc) * cellsize.y;
                real z = (iz) * cellsize.z;

                if (coo.x == ix &&
                    coo.y == iy &&
                    z0 + lift - delta <= z + 0.5 * cellsize.z) {
                        *crashedResult = true; return;}

                real3 m = magnetization.vectorAt(int3{ix, iy, iz});
                real E[3];  // Energy of 3 tip positions

                // Get 3 different tip heights
                for (int i = -1; i <= 1; i++) {
                    // First pole position and field
                    real3 R = {x0-x,
                                y0-y,
                                z0 - z + (lift + i*delta)};
                    real r = norm(R);
                    real3 B = R * prefactor/(r*r*r);
                    
                    // Second pole position and field
                    R.z += tipsize;
                    r = norm(R);
                    B -= R * prefactor/(r*r*r);
                    
                    // Energy (B.M) * V
                    E[i + 1] = dot(B, m) * V;
                }

                // dF/dz = d²E/dz²
                dFdz += ((E[0] - E[1]) + (E[2] - E[1])) / (delta*delta);
            }
        }
    }
    kernel.setValueInCell(idx, 0, dFdz);
}

MFM::MFM(Magnet* magnet,
         const Grid grid)
    : grid_(grid),
      system_(std::make_shared<System>(magnet->world(), grid_)),
      lift(10e-9),
      tipsize(1e-3) {
    magnets_[magnet->name()] = magnet;
    if (grid_.size().z > 1) {
        throw std::invalid_argument("MFM should scan a 2D surface. Reduce "
                                    "the number of z-cells to 1.");
    }

    if (magnet->world()->pbcRepetitions().z > 0) {
        throw std::invalid_argument("Cannot take MFM picture of PBC in the "
                                    "z-direction.");
    }

}

MFM::MFM(const MumaxWorld* world,
         const Grid grid)
    : magnets_(world->magnets()),
      grid_(grid),
      system_(std::make_shared<System>(world, grid_)),
      lift(10e-9),
      tipsize(1e-3) {   
    if (grid_.size().z > 1) {
        throw std::invalid_argument("MFM should scan a 2D surface. Reduce the "
                                    "number of cells in the z direction to 1.");
    }

    if (world->pbcRepetitions().z > 0) {
        throw std::invalid_argument("MFM picture cannot be taken if PBC are "
                                    "enabled in the z-direction");
    }

}

Field MFM::eval() const {
    GpuBuffer<bool> crashed(1);  // Parameter to check if the tip hit the sample
    bool init = false;
    checkCudaError(cudaMemcpyAsync(crashed.get(), &init, sizeof(bool),
                                   cudaMemcpyHostToDevice, getCudaStream()));
    Field mfmTotal(system_, 1, 0.0);

    // loop over all magnets
    for (const auto& pair : magnets_) {
        Magnet* magnet = pair.second;

        Field mfm(system_, 1, 0.0);
        Grid mastergrid = magnet->world()->mastergrid();
        int3 pbcRepetitions = magnet->world()->pbcRepetitions();
        const real V = magnet->world()->cellVolume();
        int ncells = grid_.ncells();
        Field magnetization;

        if (const Ferromagnet* mag = magnet->asFM()) {
            magnetization = fullMagnetizationQuantity(mag).eval();
        } else if (const Antiferromagnet* mag = magnet->asAFM()) {
            magnetization = fullMagnetizationQuantity(mag).eval();
        } else {
            throw std::invalid_argument("Cannot calculate MFM of instance which "
                                        "is no Ferromagnet or Antiferromagnet.");
        }

        cudaLaunch(ncells, k_magneticForceMicroscopy, mfm.cu(), magnetization.cu(), mastergrid, pbcRepetitions, lift, tipsize, V, crashed.get());
        
        bool crashedResult;
        checkCudaError(cudaMemcpyAsync(&crashedResult, crashed.get(), sizeof(bool), cudaMemcpyDeviceToHost, getCudaStream()));

        if (crashedResult) {
            throw std::invalid_argument("Tip crashed into the sample. increase "
                                        "the lift or the z component of the "
                                        "origin of the MFM grid.");
        }
        mfmTotal += mfm;
    }
    return mfmTotal;
}

int MFM::ncomp() const {
    return 1;
}

std::shared_ptr<const System> MFM::system() const {
    return system_;
}